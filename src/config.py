"""
Configuration file for plant disease classification
"""

# Data settings
DATA_DIR = "data"
BATCH_SIZE = 64
TRAIN_SPLIT = 0.8

# Class imbalance settings
RARE_THRESHOLD = 200

# Training settings
NUM_EPOCHS = 15
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_BACKBONE = 1e-4

# Early stopping settings
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.001

# Model settings
IMAGE_SIZE = 224
MODEL_SAVE_PATH = "ResNet_model.pth"

# DataLoader settings
NUM_WORKERS_TRAIN = 0
NUM_WORKERS_VAL = 4
PIN_MEMORY = True

# Image normalization (ImageNet statistics)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]