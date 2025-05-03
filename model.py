import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights



class ModelV1(nn.Module):
    def __init__(self):
        super(ModelV1, self).__init__()

        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1]) 

        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 256),  
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, img_pf, img_pj):

        f1 = self.feature_extractor(img_pf)  
        f2 = self.feature_extractor(img_pj)


        f1 = f1.view(f1.size(0), -1) 
        f2 = f2.view(f2.size(0), -1)


        combined = torch.cat((f1, f2), dim=1) 

        output = self.fc(combined)  
        return output.squeeze(1)  