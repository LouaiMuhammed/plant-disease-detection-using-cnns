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


def get_mobilenet_model(num_classes, version='v2', pretrained=True, dropout=0.2):
    """
    Get MobileNet model with pretrained weights
    
    Args:
        num_classes: Number of output classes
        version: 'v2', 'v3_small', or 'v3_large'
        pretrained: Use pretrained ImageNet weights
        dropout: Dropout rate for classifier
        
    Returns:
        MobileNet model with frozen backbone and trainable classifier
    """
    if version == 'v2':
        weights = "IMAGENET1K_V1" if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier
        for param in model.classifier.parameters():
            param.requires_grad = True
        
        # Replace classifier
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
            
    elif version == 'v3_small':
        weights = "IMAGENET1K_V1" if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier
        for param in model.classifier.parameters():
            param.requires_grad = True
        
        # Replace classifier
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
            
    elif version == 'v3_large':
        weights = "IMAGENET1K_V1" if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier
        for param in model.classifier.parameters():
            param.requires_grad = True
        
        # Replace classifier
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    
    else:
        raise ValueError(f"Unknown version: {version}. Choose from 'v2', 'v3_small', 'v3_large'")
    
    return model