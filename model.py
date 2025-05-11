import torch
import torch.nn as nn
import torch.nn.init as init


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Expecting 3-channel HSV input, 224x224
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # (B, 16, 224, 224)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B, 16, 112, 112)

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # (B, 32, 112, 112)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B, 32, 56, 56)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (B, 64, 56, 56)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B, 64, 28, 28)

            nn.AdaptiveAvgPool2d((1, 1))  # (B, 64, 1, 1)
        )

        # Final fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(64 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, img1, img2):
        x1 = self.cnn(img1).view(img1.size(0), -1)  # Flatten to (B, 64)
        x2 = self.cnn(img2).view(img2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)  # (B, 128)
        output = self.fc(x)  # (B, 1)
        return output


def initialize_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.dim() > 1:
                init.kaiming_normal_(param)  
            else:
                init.zeros_(param)