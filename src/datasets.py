"""
Custom dataset for class-aware augmentation
"""
from torch.utils.data import Dataset


class ClassAwareDataset(Dataset):
    """
    Dataset that applies different transformations based on class rarity
    """
    def __init__(self, subset, rare_classes, light_transform, strong_transform):
        """
        Args:
            subset: torch.utils.data.Subset object from random_split
            rare_classes: set of class indices considered rare
            light_transform: transform for common classes
            strong_transform: transform for rare classes
        """
        self.subset = subset
        self.rare_classes = rare_classes
        self.light_transform = light_transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Get the underlying ImageFolder dataset
        dataset = self.subset.dataset
        # Get the real index in the original dataset
        real_idx = self.subset.indices[idx]
        
        # Get image path and label
        path, label = dataset.samples[real_idx]
        # Load image (always returns PIL Image)
        img = dataset.loader(path)
        
        # Apply different transforms based on class rarity
        if label in self.rare_classes:
            img = self.strong_transform(img)
        else:
            img = self.light_transform(img)
        
        return img, label