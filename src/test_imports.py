"""
Test script to verify all source modules can be imported correctly
Run this to check for any import errors or missing dependencies
"""

import sys
sys.path.append('./src')

print("Testing imports...")
print("-" * 50)

try:
    from config import *
    print("✓ config.py imported successfully")
except Exception as e:
    print(f"✗ config.py failed: {e}")

try:
    from datasets import ClassAwareDataset
    print("✓ datasets.py imported successfully")
except Exception as e:
    print(f"✗ datasets.py failed: {e}")

try:
    from early_stopping import EarlyStopping
    print("✓ early_stopping.py imported successfully")
except Exception as e:
    print(f"✗ early_stopping.py failed: {e}")

try:
    from evaluate import evaluate
    print("✓ evaluate.py imported successfully")
except Exception as e:
    print(f"✗ evaluate.py failed: {e}")

try:
    from models import get_resnet_model, get_mobilenet_model
    print("✓ models.py imported successfully")
except Exception as e:
    print(f"✗ models.py failed: {e}")

try:
    from train import (
        setup_training_resnet,
        setup_training_mobilenet,
        train_epoch,
        validate_epoch,
        train_model
    )
    print("✓ train.py imported successfully")
except Exception as e:
    print(f"✗ train.py failed: {e}")

try:
    from transforms import (
        get_light_transform,
        get_strong_transform,
        get_val_transform
    )
    print("✓ transforms.py imported successfully")
except Exception as e:
    print(f"✗ transforms.py failed: {e}")

try:
    from utils import (
        load_datasets,
        get_class_distribution,
        identify_rare_classes,
        create_weighted_sampler,
        prepare_datasets,
        create_dataloaders,
        print_dataset_info,
        print_class_distribution,
        print_rare_classes,
        print_directory_counts
    )
    print("✓ utils.py imported successfully")
except Exception as e:
    print(f"✗ utils.py failed: {e}")

print("-" * 50)
print("\nTesting key functionality...")
print("-" * 50)

try:
    import torch
    print(f"✓ PyTorch {torch.__version__} available")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"✗ PyTorch not available: {e}")

try:
    from torchvision import models, transforms
    print("✓ torchvision available")
except Exception as e:
    print(f"✗ torchvision not available: {e}")

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__} available")
except Exception as e:
    print(f"✗ NumPy not available: {e}")

try:
    from sklearn.metrics import classification_report
    print("✓ scikit-learn available")
except Exception as e:
    print(f"✗ scikit-learn not available: {e}")

try:
    import matplotlib.pyplot as plt
    print("✓ matplotlib available")
except Exception as e:
    print(f"✗ matplotlib not available: {e}")

try:
    import seaborn as sns
    print("✓ seaborn available")
except Exception as e:
    print(f"✗ seaborn not available: {e}")

print("-" * 50)
print("\nTesting model creation...")
print("-" * 50)

try:
    import torch
    model = get_resnet_model(num_classes=38)
    print(f"✓ ResNet model created successfully")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
except Exception as e:
    print(f"✗ ResNet model creation failed: {e}")

try:
    model = get_mobilenet_model(num_classes=38, version='v2')
    print(f"✓ MobileNetV2 model created successfully")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
except Exception as e:
    print(f"✗ MobileNet model creation failed: {e}")

print("-" * 50)
print("\nAll tests completed!")
print("If all checks passed, you're ready to run the notebook.")