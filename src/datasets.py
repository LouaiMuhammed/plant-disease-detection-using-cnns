"""
Custom dataset for class-aware augmentation
"""
from torch.utils.data import Dataset


class ClassAwareDataset(Dataset):
    """
    Dataset that applies different transformations based on class rarity.
    Works with both ImageFolder (pre-split) and Subset (random split).
    """
    def __init__(self, base_dataset, rare_classes, light_transform, strong_transform):
        """
        Args:
            base_dataset: Either ImageFolder or Subset object
            rare_classes: set of class indices considered rare
            light_transform: transform for common classes
            strong_transform: transform for rare classes
        """
        self.base_dataset = base_dataset
        self.rare_classes = rare_classes
        self.light_transform = light_transform
        self.strong_transform = strong_transform
        
        # Check if base_dataset is a Subset or ImageFolder
        self.is_subset = hasattr(base_dataset, 'indices')

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        if self.is_subset:
            # For Subset (random split)
            dataset = self.base_dataset.dataset
            real_idx = self.base_dataset.indices[idx]
            path, label = dataset.samples[real_idx]
            img = dataset.loader(path)
        else:
            # For ImageFolder (pre-split)
            path, label = self.base_dataset.samples[idx]
            img = self.base_dataset.loader(path)
        
        # Apply different transforms based on class rarity
        if label in self.rare_classes:
            img = self.strong_transform(img)
        else:
            img = self.light_transform(img)
        
        return img, label