"""
Data loading and preparation utilities
"""
import os
import numpy as np
import torch
from collections import Counter
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from config import (
    DATA_DIR, BATCH_SIZE, TRAIN_SPLIT, RARE_THRESHOLD,
    NUM_WORKERS_TRAIN, NUM_WORKERS_VAL, PIN_MEMORY
)
from transforms import get_light_transform, get_strong_transform, get_val_transform
from dataset import ClassAwareDataset


def load_dataset():
    """
    Load the full dataset from disk
    
    Returns:
        ImageFolder dataset
    """
    full_dataset = ImageFolder(
        root=DATA_DIR,
        transform=None
    )
    return full_dataset


def split_dataset(full_dataset):
    """
    Split dataset into train and validation sets
    
    Args:
        full_dataset: ImageFolder dataset
        
    Returns:
        train_dataset, val_dataset (both Subset objects)
    """
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size]
    )
    
    return train_dataset, val_dataset


def get_class_distribution(full_dataset, train_dataset):
    """
    Compute class distribution in training set
    
    Args:
        full_dataset: Full ImageFolder dataset
        train_dataset: Training subset
        
    Returns:
        class_counts: Counter object with class counts
    """
    train_labels = [full_dataset.targets[i] for i in train_dataset.indices]
    class_counts = Counter(train_labels)
    return class_counts, train_labels


def identify_rare_classes(class_counts, threshold=RARE_THRESHOLD):
    """
    Identify rare classes based on threshold
    
    Args:
        class_counts: Counter with class counts
        threshold: Minimum samples for a class to be considered common
        
    Returns:
        Set of rare class indices
    """
    rare_classes = {
        cls_idx for cls_idx, count in class_counts.items()
        if count < threshold
    }
    return rare_classes


def create_weighted_sampler(class_counts, train_labels, num_classes):
    """
    Create weighted sampler for balanced training
    
    Args:
        class_counts: Counter with class counts
        train_labels: List of training labels
        num_classes: Total number of classes
        
    Returns:
        WeightedRandomSampler
    """
    # Compute class weights (inverse frequency)
    class_weights = np.zeros(num_classes)
    for cls_idx, count in class_counts.items():
        class_weights[cls_idx] = 1.0 / count
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum()
    
    # Assign weight to each sample
    sample_weights = [
        class_weights[label] for label in train_labels
    ]
    
    sample_weights = torch.DoubleTensor(sample_weights)
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


def prepare_datasets(full_dataset, train_dataset, val_dataset, rare_classes):
    """
    Apply transforms to datasets
    
    Args:
        full_dataset: Full ImageFolder dataset
        train_dataset: Training subset
        val_dataset: Validation subset
        rare_classes: Set of rare class indices
        
    Returns:
        Transformed train and validation datasets
    """
    # Get transforms
    light_transform = get_light_transform()
    strong_transform = get_strong_transform()
    val_transform = get_val_transform()
    
    # Create class-aware training dataset
    train_dataset = ClassAwareDataset(
        train_dataset,
        rare_classes,
        light_transform,
        strong_transform
    )
    
    # Apply validation transform
    val_dataset.dataset.transform = val_transform
    
    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, sampler):
    """
    Create data loaders for training and validation
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        sampler: Weighted sampler for training
        
    Returns:
        train_loader, val_loader
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS_TRAIN,
        pin_memory=PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS_VAL,
        pin_memory=PIN_MEMORY
    )
    
    return train_loader, val_loader


def print_dataset_info(full_dataset):
    """
    Print dataset information
    """
    print("Classes:", full_dataset.classes)
    print("Number of images:", len(full_dataset))


def print_class_distribution(full_dataset, class_counts):
    """
    Print class distribution in training set
    """
    print("\nTraining class distribution:")
    for cls_idx, count in class_counts.items():
        print(f"{full_dataset.classes[cls_idx]:35s}: {count}")


def print_rare_classes(full_dataset, rare_classes):
    """
    Print identified rare classes
    """
    print("\nRare classes:")
    for cls_idx in rare_classes:
        print(full_dataset.classes[cls_idx])


def print_directory_counts(full_dataset):
    """
    Print image counts per class directory
    """
    print("\nImages per class directory:")
    for idx, class_name in enumerate(full_dataset.classes):
        class_path = os.path.join(DATA_DIR, class_name)
        count = len(os.listdir(class_path)) if os.path.isdir(class_path) else 0
        print(f"{class_name:35} : {count:5d}")