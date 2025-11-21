import torch
import torch.nn as nn
from torchvision import models

def get_model_v2(num_classes):
    """
    Model V2: MobileNetV2
    - Improvement: Much lighter and faster architecture than ResNet18.
    - Customization: Increased Dropout (0.4) for better regularization.
    """
    # 1. Load pre-trained MobileNetV2
    # MobileNet is optimized for speed and efficiency (perfect for "New Architecture" bonus)
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    
    # 2. (Optional) Freeze feature extractor to speed up training
    # for param in model.features.parameters():
    #     param.requires_grad = False
    
    # 3. Replace the Classifier Head
    # MobileNetV2's classifier is a Sequential block. We replace it to match our classes.
    # We access the number of input features from the last layer of the classifier
    num_ftrs = model.last_channel
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),  # Adding stronger Dropout as per assignment Step 5
        nn.Linear(num_ftrs, num_classes)
    )
    
    return model