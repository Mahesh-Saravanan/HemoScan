import os
import json
import torch
from torch.utils.data import Dataset, random_split
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

# Helper function: Extract H from HSV and L, A from LAB
def extract_hla(image):
    image = np.array(image)  # PIL → numpy (H, W, 3)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    h = hsv[:, :, 0:1]    # Hue channel
    l = lab[:, :, 0:1]    # Lightness
    a = lab[:, :, 1:2]    # A (red-green)
    
    hla = np.concatenate([h, l, a], axis=2)  # Shape: (H, W, 3)
    hla = Image.fromarray(hla.squeeze().astype(np.uint8)) if hla.shape[2] == 1 else Image.fromarray(hla.astype(np.uint8))
    return hla

class CustomDataloader(Dataset):
    def __init__(self, data_dir, transform=None, is_train=True):
        super().__init__()
        self.data_dir = data_dir
        self.is_train = is_train
        self.hb_mean = 0.0
        self.hb_std = 0.0

        # Define transforms
        if transform is None:
            if self.is_train:
                self.transform = transforms.Compose([
                    transforms.Lambda(lambda img: extract_hla(img)),
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])  # Normalize H, L, A roughly to [-1, 1]
                ])
            
            else:
                self.transform = transforms.Compose([
                    transforms.Lambda(lambda img: extract_hla(img)),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
                ])
        else:
            self.transform = transform

        # Load data
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

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        pf_path = self.pf_paths[idx]
        pj_path = self.pj_paths[idx]
        target_value = self.targets[idx]

        img_pf = Image.open(pf_path).convert("RGB")
        img_pj = Image.open(pj_path).convert("RGB")

        if self.transform:
            img_pf = self.transform(img_pf)
            img_pj = self.transform(img_pj)

        # Normalize target
        target_value = (target_value - self.hb_mean) / self.hb_std
        target_tensor = torch.tensor(target_value, dtype=torch.float)

        return img_pf, img_pj, target_tensor

# Train/val split
def train_test_split(dataset, test_split=0.2, seed=42):
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - test_size
    return random_split(dataset, [train_size, test_size],
                        generator=torch.Generator().manual_seed(seed))

# Create loaders
def create_dataloaders(data_dir, batch_size=32, test_split=0.2):
    # Full dataset first
    full_dataset = CustomDataloader(data_dir=data_dir, is_train=True)

    # Split
    train_dataset, val_dataset = train_test_split(full_dataset, test_split=test_split)

    # Set correct transform for val dataset
    val_dataset.dataset.is_train = False
    val_dataset.dataset.transform = transforms.Compose([
        transforms.Lambda(lambda img: extract_hla(img)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = None if len(val_loader)==0 else val_loader

    return train_loader, val_loader, full_dataset.hb_mean, full_dataset.hb_std


