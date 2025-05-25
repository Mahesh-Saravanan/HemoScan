from config import Config
from model import ModelV1
import torch
from PIL import Image
import numpy as np

def to_pil_image(img):
    if isinstance(img, Image.Image):
        return img
    elif isinstance(img, np.ndarray):
        return Image.fromarray(img.astype('uint8'))
    else:
        raise TypeError("Input must be a PIL Image or a NumPy array.")

class Inference:
    def __init__(self,y_mean = 12.047,y_std = 2.163):
        config = Config()
        self.model = ModelV1(freeze_backbone=True, unfreeze_from_layer='layer4')
        self.model.load_state_dict(torch.load(config.model_load_path))

        self.transform = config.transform
        self.device = config.device
        self.y_mean = y_mean
        self.y_std = y_std
        self.model.to(self.device)
        print(f"Model and its weight loaded and pushed to device:{self.device}")

    def predict(self,image_palm,image_nail):
        self.model.eval()
        image_palm = to_pil_image(image_palm)
        image_nail = to_pil_image(image_nail)

        img_pf = self.transform(image_palm)
        img_pb = self.transform(image_nail)

        img_pf = img_pf.unsqueeze(0).to(self.device)
        img_pb = img_pb.unsqueeze(0).to(self.device)

        output = self.model(img_pf, img_pb)
        output = output.detach().cpu().numpy()[0]
        output = output * self.y_std +  self.y_mean
        output = round(float(output),2)
        return output

