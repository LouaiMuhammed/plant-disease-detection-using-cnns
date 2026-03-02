"""
Configuration file for plant disease classification
"""

# Data settings
DATA_DIR = "../data"
TRAIN_DIR = "../data/train"
VAL_DIR = "../data/val"
BATCH_SIZE = 64
TRAIN_SPLIT = 0.8

# Class imbalance settings
RARE_THRESHOLD = 200
OVERSAMPLING_ENABLED = True
# Strength of minority oversampling. 1.0 = inverse-frequency, >1.0 = more aggressive.
OVERSAMPLING_POWER = 0.8
# Clamp oversampling multipliers to avoid unstable training.
OVERSAMPLING_MIN_MULTIPLIER = 1.0
OVERSAMPLING_MAX_MULTIPLIER = 8.0
# Controls how many samples are drawn per training epoch via WeightedRandomSampler.
OVERSAMPLING_EPOCH_MULTIPLIER = 1.0

# Training settings - ResNet
NUM_EPOCHS = 15
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_BACKBONE = 1e-4

# Training settings - MobileNet
MOBILENET_NUM_EPOCHS = 10
MOBILENET_LR_HEAD = 1e-3
MOBILENET_LR_BACKBONE = 1e-4

# Early stopping settings
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.001

# Model settings
IMAGE_SIZE = 224
MODEL_SAVE_PATH = "../models/ResNet18_model.pth"
RESNET_MODEL_SAVE_PATH = "../models/ResNet18_model.pth"
MOBILENET_MODEL_SAVE_PATH = "../models/MobileNet_model.pth"

# DataLoader settings
NUM_WORKERS_TRAIN = 0
NUM_WORKERS_VAL = 4
PIN_MEMORY = True

# Image normalization (ImageNet statistics)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
