"""
Model architectures for plant disease classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def get_resnet_model(num_classes):
    """
    Get ResNet18 model with pretrained ImageNet weights
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        ResNet18 model with frozen backbone and trainable classifier
    """
    model = models.resnet18(weights="IMAGENET1K_V1")

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the final classifier layer
    for param in model.fc.parameters():
        param.requires_grad = True

    # Replace classifier head
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model