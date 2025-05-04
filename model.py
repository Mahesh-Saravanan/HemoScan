import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights



class ModelV1(nn.Module):
    def __init__(self, freeze_backbone=True, unfreeze_from_layer='layer4'):
        super(ModelV1, self).__init__()
        
      
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity() 
        

        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
            
 
            unfreeze = False
            for name, module in self.resnet.named_children():
                if name == unfreeze_from_layer:
                    unfreeze = True
                if unfreeze:
                    for param in module.parameters():
                        param.requires_grad = True

     
        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, img1, img2):
        x1 = self.resnet(img1)
        x2 = self.resnet(img2)
        x = torch.cat((x1, x2), dim=1)
        output = self.fc(x)
        return output


def initialize_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.dim() > 1:
                init.kaiming_normal_(param)  
            else:
                init.zeros_(param)
