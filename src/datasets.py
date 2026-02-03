from torch.utils.data import Dataset, random_split
from torchvision.datasets import ImageFolder
from collections import Counter

def create_datasets(data_dir, light_transform, strong_transform, val_transform):
    full_dataset = ImageFolder(root=data_dir, transform=None)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    train_labels = [full_dataset.targets[i] for i in train_subset.indices]
    class_counts = Counter(train_labels)

    rare_classes = {
        cls for cls, count in class_counts.items() if count < 200
    }

    train_dataset = ClassAwareDataset(
        train_subset, rare_classes, light_transform, strong_transform
    )

    val_subset.dataset.transform = val_transform

    return train_dataset, val_subset, full_dataset.classes
