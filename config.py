from torchvision import transforms
import torch

class Config:
    def __init__(self):
        self.data_dir = "/Users/maheshsaravanan/Documents/HemoScan/Dataset"
        self.no_of_images = 462
        self.image_size = (256, 256)
        self.transform = None
        # self.transform = transform = transforms.Compose([
        #                             transforms.Resize((224, 224)),
        #                             transforms.ToTensor(),
        #                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                                 std=[0.229, 0.224, 0.225])])
        self.batch_size = 2
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.shuffle = True
        self.test_split = 0.10
        
        self.criterion = torch.nn.MSELoss()
        self.num_epochs = 10
        self.learning_rate = 1e-4



        self.model_save_path = r"/Users/maheshsaravanan/Documents/HemoScan/ProjectHS/Models"
        self.save_interval = 5
        self.load_model = True
        self.model_load_path = r"/Users/maheshsaravanan/Documents/HemoScan/ProjectHS/Models/model_epoch_best.pth"
        self.log_path = r"/Users/maheshsaravanan/Documents/HemoScan/ProjectHS/log"
        
