import torch.nn as nn
from torchvision import models

def get_resnet18(num_classes, unfreeze_layer4=True):
    model = models.resnet18(weights="IMAGENET1K_V1")

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    for param in model.fc.parameters():
        param.requires_grad = True

    if unfreeze_layer4:
        for param in model.layer4.parameters():
            param.requires_grad = True

    return model
