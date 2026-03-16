"""
Data loading and preparation utilities for pre-split datasets
"""
import os
import numpy as np
import torch
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from pathlib import Path


from .config import (
    TRAIN_DIR, VAL_DIR, BATCH_SIZE, RARE_THRESHOLD,
    NUM_WORKERS_TRAIN, NUM_WORKERS_VAL, PIN_MEMORY,
    OVERSAMPLING_ENABLED, OVERSAMPLING_POWER,
    OVERSAMPLING_MIN_MULTIPLIER, OVERSAMPLING_MAX_MULTIPLIER,
    OVERSAMPLING_EPOCH_MULTIPLIER
)
from .transforms import get_light_transform, get_strong_transform, get_val_transform, get_train_transform
from .datasets import ClassAwareDataset


def load_datasets():
    """
    Load datasets from pre-split train/val directories
    
    Returns:
        train_dataset_base, val_dataset_base: ImageFolder datasets without transforms
        classes: List of class names
        num_classes: Number of classes
    """
    train_dataset_base = ImageFolder(root=TRAIN_DIR, transform=None)
    val_dataset_base = ImageFolder(root=VAL_DIR, transform=None)
    
    classes = train_dataset_base.classes
    num_classes = len(classes)
    
    return train_dataset_base, val_dataset_base, classes, num_classes


def get_class_distribution(train_dataset_base):
    """
    Get class distribution from training dataset
    
    Args:
        train_dataset_base: Training ImageFolder dataset
        
    Returns:
        class_counts: Counter object with class counts
        train_labels: List of all training labels
    """
    train_labels = [label for _, label in train_dataset_base.imgs]
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
    # Build per-class multipliers so minority classes are sampled more often.
    class_weights = np.ones(num_classes, dtype=np.float64)
    if OVERSAMPLING_ENABLED:
        max_count = max(class_counts.values())
        for cls_idx in range(num_classes):
            count = class_counts.get(cls_idx, 1)
            multiplier = (max_count / count) ** OVERSAMPLING_POWER
            multiplier = np.clip(
                multiplier,
                OVERSAMPLING_MIN_MULTIPLIER,
                OVERSAMPLING_MAX_MULTIPLIER
            )
            class_weights[cls_idx] = multiplier

    # Assign a weight to each sample according to its class multiplier.
    sample_weights = torch.DoubleTensor([class_weights[label] for label in train_labels])
    num_samples = max(1, int(len(sample_weights) * OVERSAMPLING_EPOCH_MULTIPLIER))

    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True
    )
    
    return sampler


def prepare_datasets(train_dataset_base, val_dataset_base, rare_classes):
    """
    Apply transforms to datasets
    
    Args:
        train_dataset_base: Training ImageFolder dataset
        val_dataset_base: Validation ImageFolder dataset
        rare_classes: Set of rare class indices
        
    Returns:
        Transformed train and validation datasets
    """
    # Training augmentation can be unified through get_train_transform().
    # Validation must stay deterministic so metrics remain comparable.
    train_transform = get_train_transform()
    strong_transform = get_strong_transform()
    val_transform = get_val_transform()

    # Create class-aware training dataset
    train_dataset = ClassAwareDataset(
        train_dataset_base,
        rare_classes,
        train_transform,
        strong_transform
    )
    
    # Create validation dataset with transform
    val_dataset = ImageFolder(root=VAL_DIR, transform=val_transform)
    
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
        num_workers=0,
        pin_memory=PIN_MEMORY
    )
    
    return train_loader, val_loader


def print_dataset_info(classes, train_dataset, val_dataset):
    """Print dataset information"""
    print("Classes:", classes)
    print("Number of training images:", len(train_dataset))
    print("Number of validation images:", len(val_dataset))


def print_class_distribution(classes, class_counts):
    """Print class distribution in training set"""
    print("\nTraining class distribution:")
    for cls_idx, count in class_counts.items():
        print(f"{classes[cls_idx]:35s}: {count}")


def print_rare_classes(classes, rare_classes):
    """Print identified rare classes"""
    print(f"\nRare classes (< {RARE_THRESHOLD} samples):")
    for cls_idx in rare_classes:
        print(classes[cls_idx])


def print_directory_counts(train_dir, val_dir, classes):
    """Print image counts per class directory"""
    print("\nTraining set class counts:")
    for idx, class_name in enumerate(classes):
        class_path = os.path.join(train_dir, class_name)
        count = len(os.listdir(class_path)) if os.path.isdir(class_path) else 0
        print(f"{class_name:35} : {count:5d}")
    
    print("\nValidation set class counts:")
    for idx, class_name in enumerate(classes):
        class_path = os.path.join(val_dir, class_name)
        count = len(os.listdir(class_path)) if os.path.isdir(class_path) else 0
        print(f"{class_name:35} : {count:5d}")

def load_checkpoint(checkpoint_path, device):
    """Load model and classes from a checkpoint file"""
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(ckpt, dict) and 'idx_to_class' in ckpt:
        classes = list(ckpt['idx_to_class'])
    elif isinstance(ckpt, dict) and 'class_to_idx' in ckpt:
        inv = {v: k for k, v in ckpt['class_to_idx'].items()}
        classes = [inv[i] for i in range(len(inv))]
    else:
        raise RuntimeError('Checkpoint missing class mapping')
    
    return ckpt, classes


def resolve_checkpoint(candidates):
    """Return first existing checkpoint from a list of paths"""
    for p in candidates:
        if Path(p).exists():
            return Path(p)
    raise FileNotFoundError(f'No checkpoint found in: {candidates}')
