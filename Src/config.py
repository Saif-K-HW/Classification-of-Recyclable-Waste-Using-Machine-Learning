"""
Project Configuration
Defines shared constants, paths, and reproducibility helpers
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

# Core training and data settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
SPLIT_RATIOS = (0.7, 0.15, 0.15)
BASE_LR = 1e-4
FINE_TUNE_LR = 1e-5
EPOCHS_FROZEN = 20
EPOCHS_FINE = 8

# Canonical model names and supported input image extensions
MODEL_NAMES = ("baseline_cnn", "mobilenet_frozen", "mobilenet_finetuned")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}

# Project paths used across all pipeline scripts
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_GLOBAL_DIR = PLOTS_DIR / "global"
METRICS_DIR = RESULTS_DIR / "metrics"
METRICS_GLOBAL_DIR = METRICS_DIR / "global"
LOGS_DIR = RESULTS_DIR / "logs"
MISCLASSIFIED_DIR = RESULTS_DIR / "misclassified"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def get_model_metrics_dir(model_name: str) -> Path:
    return METRICS_DIR / model_name


def get_model_plots_dir(model_name: str) -> Path:
    return PLOTS_DIR / model_name


def ensure_directories() -> None:
    """Create output folders so every script can save artifacts safely."""
    # Base output folders that always need to exist
    base_paths = [
        SPLITS_DIR,
        MODELS_DIR,
        PLOTS_DIR,
        PLOTS_GLOBAL_DIR,
        METRICS_DIR,
        METRICS_GLOBAL_DIR,
        LOGS_DIR,
        MISCLASSIFIED_DIR,
        EXPERIMENTS_DIR,
    ]

    for path in base_paths:
        path.mkdir(parents=True, exist_ok=True)

    # Model-specific metrics and plots folders
    for model_name in MODEL_NAMES:
        get_model_metrics_dir(model_name).mkdir(parents=True, exist_ok=True)
        get_model_plots_dir(model_name).mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = SEED) -> None:
    """Seed every random source so runs are reproducible end to end."""
    # Keep randomness aligned across Python, NumPy, and TensorFlow
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
