from torchvision import transforms
import torch
from utils import extract_hla
import os

class Config:
    def __init__(self):
        self.data_dir = "/Users/maheshsaravanan/Documents/HemoScan/Dataset"
        self.no_of_images = 462
        self.image_size = (256, 256)
        self.transform = transforms.Compose([
                            transforms.Lambda(lambda img: extract_hla(img)),
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                std=[0.5, 0.5, 0.5])])
        

        self.batch_size = 2
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.shuffle = True
        self.test_split = 0.2
        
        self.criterion = torch.nn.MSELoss()
        self.num_epochs = 10
        self.learning_rate = 1e-5



        self.model_save_path = r"/Users/maheshsaravanan/Documents/HemoScan/ProjectHS/Models"
        self.save_interval = -1
        self.save_best_model = False
        self.load_model = False
        self.model_load_path = os.path.join(self.model_save_path,"model_weights_2505.pth")
        self.log_path = r"/Users/maheshsaravanan/Documents/HemoScan/ProjectHS/log"
        
