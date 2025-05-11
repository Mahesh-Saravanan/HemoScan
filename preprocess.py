import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from skimage.exposure import match_histograms
from config import Config
# ========== Configuration Class ==========

config = Config()


# ========== Configuration ==========
DATASET_PATH = config.data_dir
IMAGE_COUNT = config.no_of_images
OUTPUT_SIZE = config.image_size 
DEVICE =config.device

# ========== Image Preprocessing Functions ==========
def colorProcess(rgb_image, reference_rgb=None, clahe_clip=2.0, clahe_tile_grid=(8, 8), gamma=1.0):
    rgb_image = np.array(rgb_image, dtype=np.uint8)
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile_grid)
    v = clahe.apply(v)

    if reference_rgb is not None:
        ref_hsv = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2HSV)
        v_ref = ref_hsv[..., 2]
        v = match_histograms(v, v_ref, channel_axis=None).astype(np.uint8)

    v = np.clip(255 * ((v / 255.0) ** (1.0 / gamma)), 0, 255).astype(np.uint8)

    hsv_enhanced = cv2.merge([h, s, v])
    rgb_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)

    return Image.fromarray(rgb_enhanced)

# ========== DeepLabV3 Person Mask Dimming ==========
def DeepLabV3(image_batch, model, transform, device='mps'):
    processed_images = []

    model.to(device)
    model.eval()

    input_tensors = torch.stack([transform(img) for img in image_batch]).to(device)

    with torch.no_grad():
        outputs = model(input_tensors)['out']

    PERSON_CLASS = 15

    for idx, output in enumerate(outputs):
        prediction = output.argmax(0)
        mask = prediction == PERSON_CLASS

        orig_np = np.array(image_batch[idx]).astype(np.float32)
        mask_resized = cv2.resize(mask.cpu().numpy().astype(np.uint8), (orig_np.shape[1], orig_np.shape[0]))

        dimmed = orig_np * 1#0.4
        dimmed[mask_resized == 1] = orig_np[mask_resized == 1]

        processed = np.clip(dimmed, 0, 255).astype(np.uint8)
        processed_images.append(Image.fromarray(processed.astype(np.uint8)))

    return processed_images

# ========== Main Processing ==========
def main(model):
    transform = transforms.Compose([
    transforms.Resize((256, 256)),  
    
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

 
    for idi in tqdm(range(1, IMAGE_COUNT + 1), desc="Processing Images", unit="pair"):
        id_str = f"{idi:04d}"
        folder_path = os.path.join(DATASET_PATH, id_str)
        pf_path = os.path.join(folder_path, "PF.jpg")
        pb_path = os.path.join(folder_path, "PB.jpg")

        if not os.path.exists(pf_path) or not os.path.exists(pb_path):
            continue

        
        img_pf = Image.open(pf_path).convert("RGB")
        img_pb = Image.open(pb_path).convert("RGB")

        proc_pf = colorProcess(img_pf)
        proc_pb = colorProcess(img_pb)

        processed_batch = DeepLabV3([proc_pf, proc_pb], model, transform, device=DEVICE)

        for name, processed_img in zip(["PF", "PB"], processed_batch):
            resized = processed_img.resize(OUTPUT_SIZE)
            save_path = os.path.join(folder_path, f"{name}_processed.jpg")
            resized.save(save_path)

if __name__ == "__main__":
    from torchvision.models.segmentation import deeplabv3_resnet101
    model = deeplabv3_resnet101(pretrained=True)
    main(model)
