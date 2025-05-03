import numpy as np
import os
import json
import torch

def denormalize(tensor, mean, std):
    
    tensor = tensor * torch.tensor(std).view(1, 3, 1, 1) + torch.tensor(mean).view(1, 3, 1, 1)
    return tensor