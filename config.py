"""
Configuration settings for DenWide-Net MRI ASD Detection Project.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"

# Create directories if they do not exist
for p in [DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR, STATIC_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Training Hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = "cuda"  # Auto-falls back to "cpu" if cuda unavailable
SEED = 42

# Image Specifications
IMAGE_SIZE = (224, 224)
IN_CHANNELS = 3  # RGB slices or replicated grayscale MRI slices
NUM_CLASSES = 2  # 0: Typical Control (TC), 1: Autism Spectrum Disorder (ASD)
CLASS_NAMES = ["Typical Control (TC)", "Autism Spectrum Disorder (ASD)"]

# DenWide-Net Architecture Configurations
GROWTH_RATE = 24       # DenseNet growth rate (features added per dense layer)
WIDEN_FACTOR = 2       # WideResNet channel widening multiplier (k)
COMPRESSION_FACTOR = 0.5  # Transition layer channel compression (theta)
BLOCK_CONFIG = (4, 6, 8, 4)  # Number of layers in each of the 4 Dense-Wide stages
DROPOUT_RATE = 0.2
USE_ATTENTION = True

# Best Model Weights Path
BEST_MODEL_PATH = CHECKPOINT_DIR / "denwidenet_best.pth"
LAST_MODEL_PATH = CHECKPOINT_DIR / "denwidenet_last.pth"
