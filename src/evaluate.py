"""
Model evaluation utilities
"""
import torch
from sklearn.metrics import classification_report, confusion_matrix


def evaluate(model, dataloader, device):
    """
    Evaluate model on a dataset
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
    """
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    return y_true, y_pred