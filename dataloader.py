import os
import json
import torch
from torch.utils.data import Dataset, random_split
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
class CustomDataloader(Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.hb_mean = 0.0
        self.hb_std = 0.0

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        
        self.pf_paths = []
        self.pj_paths = []
        self.targets = []

        for folder_name in sorted(os.listdir(self.data_dir)):
            folder_path = os.path.join(self.data_dir, folder_name)
        
            if not os.path.isdir(folder_path):
                continue 
            
            pf_file = os.path.join(folder_path, 'PF_processed.JPG')
            pb_file = os.path.join(folder_path, 'PB_processed.JPG')
            json_file = os.path.join(folder_path, f"y{folder_name}.json")

            
            if not (os.path.isfile(pf_file) and os.path.isfile(pb_file) and os.path.isfile(json_file)):
                continue 
            
            self.pf_paths.append(pf_file)
            self.pj_paths.append(pb_file)

            
            with open(json_file, 'r') as f:
                data = json.load(f)
               
                target_value = data.get("hb", None)
                if target_value is None:
                    raise KeyError(f"'hb' not found in {json_file}")
                self.targets.append(float(target_value))
        self.hb_mean = np.mean(np.array(self.targets))
        self.hb_std = np.std(np.array(self.targets))


    def hsv2tensor(self, hsv_array, size=(224, 224), normalize=True):
   
        
        hsv_resized = cv2.resize(hsv_array, size)
      

        # Convert to float and scale to [0, 1]
        hsv_tensor = torch.from_numpy(hsv_resized).float().permute(2, 0, 1) / 255.0

        if normalize:
            hsv_tensor = hsv_tensor / 255.0 
            mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
            hsv_tensor = (hsv_tensor - mean) / std

        return hsv_tensor

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        
        pf_path = self.pf_paths[idx]
        pj_path = self.pj_paths[idx]
        target_value = self.targets[idx]

        img_pf = Image.open(pf_path).convert("RGB")
        img_pj = Image.open(pj_path).convert("RGB")

        img_pf = cv2.cvtColor(np.array(img_pf), cv2.COLOR_RGB2HSV)
        img_pj = cv2.cvtColor(np.array(img_pj), cv2.COLOR_RGB2HSV)
       

        img_pf = self.hsv2tensor(img_pf)
        img_pj = self.hsv2tensor(img_pj)

        
        target_value = (target_value - self.hb_mean) / self.hb_std
        target_tensor = torch.tensor(target_value, dtype=torch.float)

        return img_pf, img_pj, target_tensor


def train_test_split(dataset, test_split=0.2):
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset