from torchvision import transforms
import torch

class Config:
    def __init__(self):
        self.data_dir = "/Users/maheshsaravanan/Documents/HemoScan/Dataset"
        self.transform = transform = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])
        self.batch_size = 2
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.shuffle = True
        self.test_split = 0.80
        
        self.criterion = torch.nn.MSELoss()
        self.num_epochs = 2
        self.learning_rate = 1e-4



        self.model_save_path = r"/Users/maheshsaravanan/Documents/HemoScan/ProjectHS/Models"
        self.save_interval = 5
        self.load_model = False
        self.model_load_path = r"/Users/maheshsaravanan/Documents/HemoScan/ProjectHS/Models/model_epoch_5.pth"
        self.log_path = r"/Users/maheshsaravanan/Documents/HemoScan/ProjectHS/log"
        
