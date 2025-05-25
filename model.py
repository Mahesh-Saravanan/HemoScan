import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=2, dropout=0.1):  # smaller heads
        super(CrossAttentionBlock, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),  # reduced FF size
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query, key_value):
        attn_output, _ = self.cross_attention(query, key_value, key_value)
        attn_output = self.dropout(attn_output)
        x = self.norm1(attn_output + query)

        ff_output = self.feedforward(x)
        ff_output = self.dropout(ff_output)
        output = self.norm2(ff_output + x)

        return output

class ModelV1(nn.Module):
    def __init__(self, freeze_backbone=True, unfreeze_from_layer='layer4'):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Modify input if needed
        self.resnet.conv1 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=self.resnet.conv1.out_channels,
            kernel_size=self.resnet.conv1.kernel_size,
            stride=self.resnet.conv1.stride,
            padding=self.resnet.conv1.padding,
            bias=False
        )
        torch.nn.init.kaiming_normal_(self.resnet.conv1.weight, mode='fan_out', nonlinearity='relu')

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
        
        # BIG CHANGE
        self.project = nn.Linear(512, 256)  # reduce from 512 → 256

        # Smaller Cross Attention
        self.cross_attention1 = CrossAttentionBlock(embed_dim=256, num_heads=2)
        self.cross_attention2 = CrossAttentionBlock(embed_dim=256, num_heads=2)

        # Lighter final head
        self.fc = nn.Sequential(
            nn.Linear(256 * 2, 128),  # because concat
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, img1, img2):
        feat1 = self.resnet(img1)
        feat2 = self.resnet(img2)
        
        feat1 = self.project(feat1)
        feat2 = self.project(feat2)
        
        feat1 = feat1.unsqueeze(1)  # (B, 1, 256)
        feat2 = feat2.unsqueeze(1)

        attn1 = self.cross_attention1(feat1, feat2)
        attn2 = self.cross_attention2(feat2, feat1)

        attn1 = attn1.squeeze(1)
        attn2 = attn2.squeeze(1)

        combined = torch.cat([attn1, attn2], dim=1)

        output = self.fc(combined)
        return output

def initialize_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.dim() > 1:
                init.kaiming_normal_(param)
            else:
                init.zeros_(param)