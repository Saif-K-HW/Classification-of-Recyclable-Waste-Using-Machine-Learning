"""
Data Loading and tf.data Pipeline
Builds train/val/test datasets and class metadata from split CSV files
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import tensorflow as tf

from config import BATCH_SIZE, IMAGE_SIZE, MODELS_DIR, SEED, SPLITS_DIR
from experiment_utils import OutputPaths, get_default_output_paths

# Define lightweight augmentation used only for training batches
TRAIN_AUGMENTATION = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ],
    name="train_augmentation",
)


def load_split_dataframe(split_name: str) -> pd.DataFrame:
    # Load one split CSV and validate that it exists
    split_path = SPLITS_DIR / f"{split_name}.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    return pd.read_csv(split_path)


def _save_class_names(class_names: List[str], class_names_path: Path) -> None:
    # Persist class ordering so evaluation/prediction decode labels consistently
    class_names_path.parent.mkdir(parents=True, exist_ok=True)
    class_names_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")


def load_class_names(class_names_path: Path | None = None) -> List[str]:
    # Read class names saved during dataset preparation/training
    path = class_names_path or (MODELS_DIR / "class_names.json")
    if not path.exists():
        raise FileNotFoundError(f"Missing class names file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _decode_resize_normalize(image_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    # Decode image files and normalize pixels to [0, 1]
    image_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def _apply_train_augmentation(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    # Augmenting only train keeps validation and test metrics honest.
    image = TRAIN_AUGMENTATION(image, training=True)
    return image, label


def _build_dataset(df: pd.DataFrame, batch_size: int, is_training: bool, use_augmentation: bool) -> tf.data.Dataset:
    # Build base dataset from file paths and numeric labels
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            df["filepath"].astype(str).to_numpy(),
            df["label_idx"].astype("int32").to_numpy(),
        )
    )

    if is_training:
        dataset = dataset.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)

    dataset = dataset.map(_decode_resize_normalize, num_parallel_calls=tf.data.AUTOTUNE)

    # We cache decoded tensors first so random augmentation can still vary every epoch.
    dataset = dataset.cache()

    if is_training and use_augmentation:
        dataset = dataset.map(_apply_train_augmentation, num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch keeps the GPU/CPU fed while the next batch is prepared in the background.
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def load_datasets(
    batch_size: int = BATCH_SIZE,
    use_augmentation: bool = True,
    output_paths: OutputPaths | None = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    # Default behavior stays unchanged, but experiments can inject isolated output roots.
    output_paths = output_paths or get_default_output_paths()
    output_paths.ensure_directories()

    # Load split CSV files prepared by make_splits.py
    train_df = load_split_dataframe("train")
    val_df = load_split_dataframe("val")
    test_df = load_split_dataframe("test")

    # Build stable class-to-index mapping from training labels
    class_names = sorted(train_df["label"].unique().tolist())
    class_to_index: Dict[str, int] = {label: idx for idx, label in enumerate(class_names)}

    for split_df in (train_df, val_df, test_df):
        split_df["label_idx"] = split_df["label"].map(class_to_index)
        if split_df["label_idx"].isna().any():
            raise ValueError("Found labels in val/test that are missing from train split.")

    # Save class names and build tf.data pipelines for each split
    _save_class_names(class_names, output_paths.class_names_path)

    train_ds = _build_dataset(train_df, batch_size=batch_size, is_training=True, use_augmentation=use_augmentation)
    val_ds = _build_dataset(val_df, batch_size=batch_size, is_training=False, use_augmentation=use_augmentation)
    test_ds = _build_dataset(test_df, batch_size=batch_size, is_training=False, use_augmentation=use_augmentation)

    return train_ds, val_ds, test_ds, class_names
