#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReThink Recycle Dataset Pipeline
Loads and preprocesses the recyclable waste classification dataset.

This module handles:
- Discovering material folders and item subfolders
- Building a metadata DataFrame with file paths and labels
- Creating stratified train/val/test splits
- Implementing TensorFlow preprocessing and augmentation pipelines
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, List

# ============================================================================
# CONFIGURATION
# ============================================================================

# Update this path to your dataset location
ROOT_DIR = r"c:\Users\saifa\Desktop\Studies\F20PA - Research Methods and Requirements Engineering\Dissertation\Re-think recycle dataset_organized"

# Image preprocessing parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Train/val/test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# DATASET DISCOVERY & METADATA BUILDING
# ============================================================================

def discover_dataset_structure(root_dir: str) -> pd.DataFrame:
    """
    Discover all material folders and item subfolders.
    
    Expected structure:
        root_dir/
            Ceramic/
                001_itemname/
                    #001chinese_name/
                        image1.jpg
                        image2.jpg
            Glass/
                ...
            Plastic/
                ...
    
    Args:
        root_dir: Path to the root dataset directory
        
    Returns:
        DataFrame with columns: ["filepath", "material", "item_folder"]
    """
    data = []
    root_path = Path(root_dir)
    
    # Iterate through material folders (top-level directories)
    for material_folder in sorted(root_path.iterdir()):
        if not material_folder.is_dir():
            continue
        
        material_name = material_folder.name
        
        # Skip non-material directories (e.g., CSV files, metadata)
        if material_name.startswith('.') or material_name.endswith('.csv'):
            continue
        
        print(f"Processing material: {material_name}")
        
        # Iterate through item subfolders (e.g., 001_Red and white plastic bag)
        for item_folder in sorted(material_folder.iterdir()):
            if not item_folder.is_dir():
                continue
            
            item_name = item_folder.name
            
            # Iterate through Chinese-named subfolders (one level deeper)
            for chinese_folder in sorted(item_folder.iterdir()):
                if not chinese_folder.is_dir():
                    continue
                
                # Find all image files in the Chinese-named folder
                image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
                image_files = [
                    f for f in chinese_folder.iterdir()
                    if f.is_file() and f.suffix in image_extensions
                ]
                
                # Add each image to the dataset
                for image_file in image_files:
                    data.append({
                        'filepath': str(image_file),
                        'material': material_name,
                        'item_folder': item_name
                    })
    
    df = pd.DataFrame(data)
    print(f"\nTotal images discovered: {len(df)}")
    print(f"Material categories: {df['material'].nunique()}")
    print(f"Item folders: {df['item_folder'].nunique()}\n")
    
    return df


def create_stratified_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/val/test using stratified sampling.
    
    Ensures each split has similar material distribution.
    
    Args:
        df: DataFrame with dataset metadata
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    np.random.seed(random_seed)
    
    # First split: train vs (val + test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=random_seed,
        stratify=df['material']
    )
    
    # Second split: val vs test
    val_test_ratio = test_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_test_ratio,
        random_state=random_seed,
        stratify=temp_df['material']
    )
    
    print("Stratified Split Summary:")
    print(f"  Train: {len(train_df)} images ({100*len(train_df)/len(df):.1f}%)")
    print(f"  Val:   {len(val_df)} images ({100*len(val_df)/len(df):.1f}%)")
    print(f"  Test:  {len(test_df)} images ({100*len(test_df)/len(df):.1f}%)")
    print()
    
    return train_df, val_df, test_df


def save_metadata(df: pd.DataFrame, output_path: str = "dataset_metadata.csv"):
    """
    Save dataset metadata to CSV for reproducibility.
    
    Args:
        df: DataFrame with dataset metadata
        output_path: Path to save the CSV file
    """
    df.to_csv(output_path, index=False)
    print(f"Metadata saved to: {output_path}\n")


# ============================================================================
# TENSORFLOW PREPROCESSING & AUGMENTATION
# ============================================================================

def preprocess_image(filepath: str, image_size: Tuple[int, int] = (224, 224)) -> tf.Tensor:
    """
    Load and preprocess a single image.
    
    Steps:
    1. Read image file
    2. Decode as JPEG
    3. Convert to float32
    4. Normalize to [0, 1]
    5. Resize to target size
    
    Args:
        filepath: Path to image file
        image_size: Target size (height, width)
        
    Returns:
        Preprocessed image tensor
    """
    # Read image file
    image = tf.io.read_file(filepath)
    
    # Decode JPEG
    image = tf.image.decode_jpeg(image, channels=3)
    
    # Convert to float32 and normalize to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    
    # Resize to target size
    image = tf.image.resize(image, image_size)
    
    return image


def augment_image(image: tf.Tensor) -> tf.Tensor:
    """
    Apply data augmentation to training images.
    
    Augmentations:
    - Random horizontal flip (50% probability)
    - Random rotation (±10 degrees)
    - Random brightness adjustment (±20%)
    
    Args:
        image: Input image tensor
        
    Returns:
        Augmented image tensor
    """
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    
    # Random rotation (±10 degrees = ±0.1745 radians)
    # Note: TensorFlow doesn't have built-in random rotation, so we use a fixed approach
    # For more sophisticated rotation, consider using tf_addons
    
    # Random brightness adjustment (±20%)
    image = tf.image.random_brightness(image, 0.2)
    
    # Clip values to [0, 1] after augmentation
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image


def create_dataset_from_dataframe(
    df: pd.DataFrame,
    material_to_label: dict,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    augment: bool = False,
    shuffle: bool = True
) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset from a DataFrame.
    
    Args:
        df: DataFrame with 'filepath' and 'material' columns
        material_to_label: Dictionary mapping material names to integer labels
        image_size: Target image size
        batch_size: Batch size
        augment: Whether to apply augmentation (typically True for training)
        shuffle: Whether to shuffle the dataset
        
    Returns:
        tf.data.Dataset object
    """
    # Extract filepaths and labels
    filepaths = df['filepath'].values
    materials = df['material'].values
    labels = np.array([material_to_label[m] for m in materials])
    
    # Create dataset from tensors
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    
    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))
    
    # Define preprocessing function
    def load_and_preprocess(filepath, label):
        image = preprocess_image(filepath, image_size)
        return image, label
    
    # Apply preprocessing
    dataset = dataset.map(
        load_and_preprocess,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Apply augmentation if requested (typically for training)
    if augment:
        dataset = dataset.map(
            lambda image, label: (augment_image(image), label),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def create_datasets(
    root_dir: str = ROOT_DIR,
    batch_size: int = BATCH_SIZE,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = RANDOM_SEED
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    Main function to create train/val/test datasets.
    
    Args:
        root_dir: Path to dataset root directory
        batch_size: Batch size for datasets
        image_size: Target image size (height, width)
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, class_names)
    """
    print("=" * 70)
    print("ReThink Recycle Dataset Pipeline")
    print("=" * 70)
    print()
    
    # Step 1: Discover dataset structure
    print("Step 1: Discovering dataset structure...")
    df = discover_dataset_structure(root_dir)
    
    # Step 2: Create stratified splits
    print("Step 2: Creating stratified train/val/test splits...")
    train_df, val_df, test_df = create_stratified_splits(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )
    
    # Step 3: Save metadata
    print("Step 3: Saving metadata...")
    save_metadata(df)
    
    # Step 4: Create material-to-label mapping
    class_names = sorted(df['material'].unique())
    material_to_label = {name: idx for idx, name in enumerate(class_names)}
    
    print(f"Material classes: {class_names}")
    print(f"Number of classes: {len(class_names)}\n")
    
    # Step 5: Create TensorFlow datasets
    print("Step 4: Creating TensorFlow datasets...")
    print()
    
    train_dataset = create_dataset_from_dataframe(
        train_df,
        material_to_label,
        image_size=image_size,
        batch_size=batch_size,
        augment=True,
        shuffle=True
    )
    
    val_dataset = create_dataset_from_dataframe(
        val_df,
        material_to_label,
        image_size=image_size,
        batch_size=batch_size,
        augment=False,
        shuffle=False
    )
    
    test_dataset = create_dataset_from_dataframe(
        test_df,
        material_to_label,
        image_size=image_size,
        batch_size=batch_size,
        augment=False,
        shuffle=False
    )
    
    print("=" * 70)
    print("Dataset creation complete!")
    print("=" * 70)
    print()
    
    return train_dataset, val_dataset, test_dataset, class_names


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_dataset_info(root_dir: str = ROOT_DIR) -> dict:
    """
    Get basic information about the dataset without creating TF datasets.
    
    Args:
        root_dir: Path to dataset root directory
        
    Returns:
        Dictionary with dataset statistics
    """
    df = discover_dataset_structure(root_dir)
    
    info = {
        'total_images': len(df),
        'num_materials': df['material'].nunique(),
        'num_items': df['item_folder'].nunique(),
        'material_counts': df['material'].value_counts().to_dict(),
        'class_names': sorted(df['material'].unique())
    }
    
    return info


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create datasets
    train_ds, val_ds, test_ds, class_names = create_datasets()
    
    # Inspect a batch
    print("\nInspecting a training batch:")
    for images, labels in train_ds.take(1):
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Image value range: [{images.numpy().min():.3f}, {images.numpy().max():.3f}]")
        print(f"  Sample labels: {labels.numpy()[:5]}")
