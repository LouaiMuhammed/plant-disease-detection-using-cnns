"""
Training utilities and training loop
"""
import torch
import torch.nn as nn
import numpy as np

from .config import (
    NUM_EPOCHS, LEARNING_RATE_HEAD, LEARNING_RATE_BACKBONE,
    MOBILENET_NUM_EPOCHS, MOBILENET_LR_HEAD, MOBILENET_LR_BACKBONE,
    MODEL_SAVE_PATH, MOBILENET_MODEL_SAVE_PATH,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA
)
from .early_stopping import EarlyStopping


def setup_training_resnet(model, train_labels, num_classes, device):
    """
    Setup loss function and optimizer for ResNet
    
    Args:
        model: ResNet model
        train_labels: List of training labels
        num_classes: Number of classes
        device: Device to use
    """
    # Compute class weights (inverse frequency)
    class_counts = np.bincount(train_labels, minlength=num_classes)
    weights = 1.0 / (class_counts + 1e-6)
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # Optimizer with different learning rates for head and backbone
    optimizer = torch.optim.Adam([
        {'params': model.fc.parameters(), 'lr': LEARNING_RATE_HEAD},
        {'params': model.layer4.parameters(), 'lr': LEARNING_RATE_BACKBONE}
    ])
    
    return criterion, optimizer


def setup_training_mobilenet(model, train_labels, num_classes, device):
    """
    Setup loss function and optimizer for MobileNet
    
    Args:
        model: MobileNet model
        train_labels: List of training labels
        num_classes: Number of classes
        device: Device to use
    """
    # Compute class weights from training labels
    class_counts = np.bincount(train_labels, minlength=num_classes)
    weights = 1.0 / (class_counts + 1e-6)
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # MobileNet-specific optimizer
    # Optimize classifier (head) and last feature block (backbone)
    optimizer = torch.optim.Adam([
        {'params': model.classifier.parameters(), 'lr': MOBILENET_LR_HEAD},
        {'params': model.features[-1].parameters(), 'lr': MOBILENET_LR_BACKBONE}
    ])
    
    return criterion, optimizer


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    train_loss = running_loss / total
    train_acc = correct / total
    
    return train_loss, train_acc


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    
    val_loss = val_running_loss / val_total
    val_acc = val_correct / val_total
    
    return val_loss, val_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                num_epochs=NUM_EPOCHS, model_save_path=MODEL_SAVE_PATH):
    """
    Full training loop with early stopping
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        num_epochs: Number of epochs to train
        model_save_path: Path to save best model
    """
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
        verbose=True
    )
    
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(num_epochs):
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }