"""
Data augmentation and transformation utilities
"""
from torchvision import transforms
from .config import IMAGE_SIZE, DATASET_MEAN, DATASET_STD


def get_light_transform():
    """
    Light augmentation for common classes
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=DATASET_MEAN,
            std=DATASET_STD
        ),
        transforms.RandomErasing(
            p=0.3,
            scale=(0.02, 0.2),
            ratio=(0.3, 3.3),
            value="random"
        )
    ])


def get_strong_transform():
    """
    Strong augmentation for rare classes
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=DATASET_MEAN,
            std=DATASET_STD
        ),
        #transforms.RandomErasing(
        #    p=0.5,
        #    scale=(0.02, 0.33),
        #    ratio=(0.3, 3.3),
        #    value="random"
        #)
    ])


def get_val_transform():
    """
    Validation transform (no augmentation)
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=DATASET_MEAN,
            std=DATASET_STD
        )
    ])


def get_train_transform():
    """
    Unified training transform to force generalization.
    We must break the 'shape memory' of the leaf.
    """
    return transforms.Compose([
        # 1. THE MOST IMPORTANT FIX:
        # Instead of 0.7, we go down to 0.4. 
        # This forces the model to look at patches of leaf, not the whole shape.
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.4, 1.0)), 
        
        # 2. Add Vertical Flip (Leaves don't have a 'up' in 3D space)
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        
        # 3. Aggressive Rotation
        transforms.RandomRotation(90), 
        
        # 4. Color Jitter is vital for segmented leaves 
        # because the black background is constant, but leaf green varies.
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        
        transforms.ToTensor(),
        
        # 5. Normalize
        transforms.Normalize(
            mean=DATASET_MEAN,
            std=DATASET_STD
        ),
        
        # 6. UNCOMMENT AND USE RandomErasing
        # This is 'Dropout for pixels'. It is your best friend right now.
        transforms.RandomErasing(
            p=0.5,
            scale=(0.02, 0.25),
            value="random" # Fills with noise, not black
        )
    ])
