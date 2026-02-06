import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import DATA_DIR
from utils import (
    load_dataset, split_dataset, get_class_distribution,
    identify_rare_classes, create_weighted_sampler, prepare_datasets,
    create_dataloaders, print_dataset_info, print_class_distribution,
    print_rare_classes, print_directory_counts
)
from models import get_resnet_model
from train import setup_training, train_model
from evaluate import evaluate


def main():
    """
    Main execution function
    """
    print("=" * 70)
    print("PLANT DISEASE CLASSIFICATION")
    print("=" * 70)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print(f"Data directory: {DATA_DIR}")
    
    # Load dataset
    print("\n" + "=" * 70)
    print("LOADING DATASET")
    print("=" * 70)
    full_dataset = load_dataset()
    print_dataset_info(full_dataset)
    
    # Split dataset
    print("\n" + "=" * 70)
    print("SPLITTING DATASET")
    print("=" * 70)
    train_dataset, val_dataset = split_dataset(full_dataset)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Get class distribution
    class_counts, train_labels = get_class_distribution(full_dataset, train_dataset)
    print_class_distribution(full_dataset, class_counts)
    
    # Identify rare classes
    print("\n" + "=" * 70)
    print("IDENTIFYING RARE CLASSES")
    print("=" * 70)
    rare_classes = identify_rare_classes(class_counts)
    print_rare_classes(full_dataset, rare_classes)
    
    # Print directory counts
    print_directory_counts(full_dataset)
    
    # Create weighted sampler
    print("\n" + "=" * 70)
    print("CREATING WEIGHTED SAMPLER")
    print("=" * 70)
    num_classes = len(full_dataset.classes)
    sampler = create_weighted_sampler(class_counts, train_labels, num_classes)
    print("Weighted sampler created for balanced training")
    
    # Prepare datasets with transforms
    print("\n" + "=" * 70)
    print("APPLYING TRANSFORMS")
    print("=" * 70)
    train_dataset, val_dataset = prepare_datasets(
        full_dataset, train_dataset, val_dataset, rare_classes
    )
    print("Transforms applied (class-aware augmentation)")
    
    # Create dataloaders
    print("\n" + "=" * 70)
    print("CREATING DATALOADERS")
    print("=" * 70)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, sampler)
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)
    model = get_resnet_model(num_classes).to(device)
    print("ResNet18 model created with pretrained ImageNet weights")
    
    # Setup training
    print("\n" + "=" * 70)
    print("SETTING UP TRAINING")
    print("=" * 70)
    criterion, optimizer = setup_training(model, full_dataset, device)
    print("Loss function: CrossEntropyLoss with class weights")
    print("Optimizer: Adam with differential learning rates")
    
    # Train model
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    history = train_model(model, train_loader, val_loader, criterion, optimizer, device)
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    print("\nLoading best model for evaluation...")
    model.load_state_dict(torch.load("ResNet_model.pth"))
    evaluate(model, val_loader, device)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()