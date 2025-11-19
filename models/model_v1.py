# File: models/model_v1.py

import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes):
    """
    Returns a ResNet18 model with a custom final layer.
    
    Args:
        num_classes (int): The number of classes in your dataset.
                           (e.g., 38 for PlantVillage)
    """
    # Load the pre-trained ResNet18 model
    # weights='IMAGENET1K_V1' is the modern syntax for pretrained=True
    model = models.resnet18(weights='IMAGENET1K_V1')
    
    # Replace the final Fully Connected (fc) layer to match our dataset size
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model