#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis (EDA) for ReThink Recycle Dataset

This script generates:
1. Bar chart of image counts per material
2. CSV of image counts per item folder
3. Grid of sample images per material
4. Histograms of image widths and heights
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
import random
from typing import Tuple, List

# ============================================================================
# CONFIGURATION
# ============================================================================

# Update this path to your dataset location
ROOT_DIR = r"c:\Users\saifa\Desktop\Studies\F20PA - Research Methods and Requirements Engineering\Dissertation\Re-think recycle dataset_organized"

# Metadata file (should be created by dataset_pipeline.py)
METADATA_FILE = "dataset_metadata.csv"

# Output directory for EDA plots
OUTPUT_DIR = "eda_outputs"

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# SETUP
# ============================================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# LOAD METADATA
# ============================================================================

def load_metadata(metadata_file: str = METADATA_FILE) -> pd.DataFrame:
    """
    Load dataset metadata from CSV.
    
    Args:
        metadata_file: Path to metadata CSV
        
    Returns:
        DataFrame with dataset metadata
    """
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_file}\n"
            "Please run dataset_pipeline.py first to generate metadata."
        )
    
    df = pd.read_csv(metadata_file)
    print(f"Loaded metadata: {len(df)} images")
    print(f"Columns: {list(df.columns)}\n")
    
    return df


# ============================================================================
# ANALYSIS 1: IMAGE COUNT PER MATERIAL
# ============================================================================

def plot_material_counts(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """
    Generate bar chart of image counts per material category.
    
    Args:
        df: DataFrame with dataset metadata
        output_dir: Directory to save the plot
    """
    print("Generating material counts plot...")
    
    # Count images per material
    material_counts = df['material'].value_counts().sort_values(ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bar chart
    bars = ax.bar(range(len(material_counts)), material_counts.values, color='steelblue')
    
    # Customize plot
    ax.set_xlabel('Material Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax.set_title('Image Count per Material Category', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(material_counts)))
    ax.set_xticklabels(material_counts.index, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, material_counts.values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(value), ha='center', va='bottom', fontweight='bold')
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, "eda_material_counts.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}\n")
    
    plt.close()
    
    # Print summary
    print("Material Distribution:")
    for material, count in material_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {material:15} : {count:3d} images ({percentage:5.1f}%)")
    print()


# ============================================================================
# ANALYSIS 2: IMAGE COUNT PER ITEM FOLDER
# ============================================================================

def save_item_counts(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """
    Save image counts per item folder to CSV.
    
    Args:
        df: DataFrame with dataset metadata
        output_dir: Directory to save the CSV
    """
    print("Generating item folder counts...")
    
    # Count images per item folder
    item_counts = df.groupby('item_folder').size().reset_index(name='image_count')
    item_counts = item_counts.sort_values('image_count', ascending=False)
    
    # Save to CSV
    output_path = os.path.join(output_dir, "item_counts.csv")
    item_counts.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}\n")
    
    # Print summary
    print(f"Item Folder Statistics:")
    print(f"  Total item folders: {len(item_counts)}")
    print(f"  Min images per item: {item_counts['image_count'].min()}")
    print(f"  Max images per item: {item_counts['image_count'].max()}")
    print(f"  Mean images per item: {item_counts['image_count'].mean():.1f}")
    print(f"  Median images per item: {item_counts['image_count'].median():.1f}\n")


# ============================================================================
# ANALYSIS 3: SAMPLE IMAGE GRID
# ============================================================================

def plot_sample_grid(
    df: pd.DataFrame,
    num_materials: int = 4,
    images_per_material: int = 4,
    output_dir: str = OUTPUT_DIR
):
    """
    Generate grid of sample images from each material category.
    
    Args:
        df: DataFrame with dataset metadata
        num_materials: Number of materials to sample
        images_per_material: Number of images per material
        output_dir: Directory to save the plot
    """
    print(f"Generating sample image grid ({num_materials} materials, {images_per_material} images each)...")
    
    # Get unique materials
    all_materials = df['material'].unique()
    
    # Randomly sample materials
    sampled_materials = np.random.choice(
        all_materials,
        size=min(num_materials, len(all_materials)),
        replace=False
    )
    
    # Create figure with subplots
    fig, axes = plt.subplots(
        len(sampled_materials),
        images_per_material,
        figsize=(12, 3 * len(sampled_materials))
    )
    
    # Ensure axes is 2D
    if len(sampled_materials) == 1:
        axes = axes.reshape(1, -1)
    
    # Plot sample images
    for row, material in enumerate(sampled_materials):
        # Get all images for this material
        material_images = df[df['material'] == material]['filepath'].values
        
        # Randomly sample images
        sampled_images = np.random.choice(
            material_images,
            size=min(images_per_material, len(material_images)),
            replace=False
        )
        
        for col, image_path in enumerate(sampled_images):
            ax = axes[row, col]
            
            try:
                # Load and display image
                img = Image.open(image_path)
                ax.imshow(img)
            except Exception as e:
                print(f"  Warning: Could not load {image_path}: {e}")
                ax.text(0.5, 0.5, 'Failed to load', ha='center', va='center')
            
            # Customize subplot
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add title only to first column
            if col == 0:
                ax.set_ylabel(material, fontsize=11, fontweight='bold')
    
    plt.suptitle('Sample Images per Material Category', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, "eda_sample_grid.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}\n")
    
    plt.close()


# ============================================================================
# ANALYSIS 4: IMAGE DIMENSIONS HISTOGRAMS
# ============================================================================

def plot_image_dimensions(
    df: pd.DataFrame,
    sample_size: int = 500,
    output_dir: str = OUTPUT_DIR
):
    """
    Generate histograms of image widths and heights.
    
    Args:
        df: DataFrame with dataset metadata
        sample_size: Number of images to sample for analysis
        output_dir: Directory to save the plots
    """
    print(f"Analyzing image dimensions (sampling {sample_size} images)...")
    
    # Sample images
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=RANDOM_SEED)
    
    widths = []
    heights = []
    
    # Read image dimensions
    for filepath in sample_df['filepath'].values:
        try:
            img = Image.open(filepath)
            width, height = img.size
            widths.append(width)
            heights.append(height)
        except Exception as e:
            print(f"  Warning: Could not read {filepath}: {e}")
    
    print(f"  Successfully read {len(widths)} images\n")
    
    # Print statistics
    print("Image Dimension Statistics:")
    print(f"  Widths:  min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.1f}, median={np.median(widths):.1f}")
    print(f"  Heights: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.1f}, median={np.median(heights):.1f}\n")
    
    # Plot width histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(widths, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Image Width (pixels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Image Widths', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "eda_width_hist.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    # Plot height histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(heights, bins=30, color='coral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Image Height (pixels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Image Heights', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "eda_height_hist.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}\n")
    plt.close()


# ============================================================================
# MAIN EDA FUNCTION
# ============================================================================

def run_eda(
    metadata_file: str = METADATA_FILE,
    output_dir: str = OUTPUT_DIR,
    num_sample_materials: int = 4,
    images_per_material: int = 4,
    dimension_sample_size: int = 500
):
    """
    Run complete EDA pipeline.
    
    Args:
        metadata_file: Path to metadata CSV
        output_dir: Directory to save outputs
        num_sample_materials: Number of materials to show in sample grid
        images_per_material: Number of images per material in grid
        dimension_sample_size: Number of images to sample for dimension analysis
    """
    print("=" * 70)
    print("ReThink Recycle Dataset - Exploratory Data Analysis")
    print("=" * 70)
    print()
    
    # Load metadata
    df = load_metadata(metadata_file)
    
    # Analysis 1: Material counts
    plot_material_counts(df, output_dir)
    
    # Analysis 2: Item counts
    save_item_counts(df, output_dir)
    
    # Analysis 3: Sample grid
    plot_sample_grid(df, num_sample_materials, images_per_material, output_dir)
    
    # Analysis 4: Image dimensions
    plot_image_dimensions(df, dimension_sample_size, output_dir)
    
    print("=" * 70)
    print("EDA Complete!")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 70)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    run_eda()
