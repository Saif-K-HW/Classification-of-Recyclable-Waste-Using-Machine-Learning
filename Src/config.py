"""
Project Configuration
Defines shared constants, paths, and reproducibility helpers
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict

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

# Runtime-configurable training settings
BACKBONE = "resnet50"
AUGMENTATION = "none"
LOSS = "cross_entropy"
CLASS_WEIGHT_CAP: float | None = None
FINETUNE_UNFREEZE_LAYERS = 20
FINETUNE_LR = FINE_TUNE_LR
FINETUNE_EPOCHS = EPOCHS_FINE
LABEL_SMOOTHING = 0.0

# Keep a stable default map so experiments can override only what they need.
DEFAULT_OVERRIDES: Dict[str, Any] = {
    "backbone": BACKBONE,
    "augmentation": AUGMENTATION,
    "loss": LOSS,
    "class_weight_cap": CLASS_WEIGHT_CAP,
    "finetune_unfreeze_layers": FINETUNE_UNFREEZE_LAYERS,
    "finetune_lr": FINETUNE_LR,
    "finetune_epochs": FINETUNE_EPOCHS,
    "label_smoothing": LABEL_SMOOTHING,
    "results_dir": "results",
}

# Canonical model names and supported input image extensions
MODEL_NAMES = ("baseline_cnn", "resnet50_frozen", "resnet50_finetuned")
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


def _set_results_root(results_dir: Path) -> None:
    # Updating these globals lets legacy code keep using the same path symbols.
    global RESULTS_DIR, PLOTS_DIR, PLOTS_GLOBAL_DIR, METRICS_DIR, METRICS_GLOBAL_DIR, LOGS_DIR, MISCLASSIFIED_DIR

    RESULTS_DIR = results_dir
    PLOTS_DIR = RESULTS_DIR / "plots"
    PLOTS_GLOBAL_DIR = PLOTS_DIR / "global"
    METRICS_DIR = RESULTS_DIR / "metrics"
    METRICS_GLOBAL_DIR = METRICS_DIR / "global"
    LOGS_DIR = RESULTS_DIR / "logs"
    MISCLASSIFIED_DIR = RESULTS_DIR / "misclassified"


def apply_overrides(overrides: dict) -> dict:
    # Experiments should be configurable, but defaults must stay stable when keys are missing.
    global BACKBONE, AUGMENTATION, LOSS, CLASS_WEIGHT_CAP, FINETUNE_UNFREEZE_LAYERS
    global FINETUNE_LR, FINETUNE_EPOCHS, LABEL_SMOOTHING, FINE_TUNE_LR, EPOCHS_FINE

    merged: Dict[str, Any] = dict(DEFAULT_OVERRIDES)
    merged.update(overrides or {})

    backbone = str(merged.get("backbone", BACKBONE)).lower()
    if backbone not in {"mobilenetv2", "efficientnetb0", "resnet50", "densenet121"}:
        raise ValueError("backbone must be one of: mobilenetv2, efficientnetb0, resnet50, densenet121")

    augmentation = str(merged.get("augmentation", AUGMENTATION)).lower()
    if augmentation not in {"default", "none"}:
        raise ValueError("augmentation must be 'default' or 'none'")

    loss_name = str(merged.get("loss", LOSS)).lower()
    if loss_name not in {"cross_entropy", "focal"}:
        raise ValueError("loss must be 'cross_entropy' or 'focal'")

    class_weight_cap = merged.get("class_weight_cap", CLASS_WEIGHT_CAP)
    if class_weight_cap is not None:
        class_weight_cap = float(class_weight_cap)
        if class_weight_cap <= 0:
            raise ValueError("class_weight_cap must be positive when provided")

    finetune_unfreeze_layers = int(merged.get("finetune_unfreeze_layers", FINETUNE_UNFREEZE_LAYERS))
    if finetune_unfreeze_layers < 1:
        raise ValueError("finetune_unfreeze_layers must be >= 1")

    finetune_lr = float(merged.get("finetune_lr", FINETUNE_LR))
    if finetune_lr <= 0:
        raise ValueError("finetune_lr must be > 0")

    finetune_epochs = int(merged.get("finetune_epochs", FINETUNE_EPOCHS))
    if finetune_epochs < 1:
        raise ValueError("finetune_epochs must be >= 1")

    label_smoothing = float(merged.get("label_smoothing", LABEL_SMOOTHING))
    if label_smoothing < 0 or label_smoothing > 1:
        raise ValueError("label_smoothing must be between 0 and 1")

    results_override = merged.get("results_dir", DEFAULT_OVERRIDES["results_dir"])
    results_root = Path(results_override)
    if not results_root.is_absolute():
        results_root = (PROJECT_ROOT / results_root).resolve()

    BACKBONE = backbone
    AUGMENTATION = augmentation
    LOSS = loss_name
    CLASS_WEIGHT_CAP = class_weight_cap
    FINETUNE_UNFREEZE_LAYERS = finetune_unfreeze_layers
    FINETUNE_LR = finetune_lr
    FINETUNE_EPOCHS = finetune_epochs
    LABEL_SMOOTHING = label_smoothing

    # Keep legacy constants aligned so existing imports keep working.
    FINE_TUNE_LR = FINETUNE_LR
    EPOCHS_FINE = FINETUNE_EPOCHS

    _set_results_root(results_root)
    merged["results_dir"] = str(RESULTS_DIR)
    return merged


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
