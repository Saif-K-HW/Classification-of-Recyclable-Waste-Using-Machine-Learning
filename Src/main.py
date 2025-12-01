#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script to run the complete ReThink Recycle dataset pipeline.

This script:
1. Creates the dataset pipeline (train/val/test splits)
2. Runs exploratory data analysis (EDA)
3. Generates visualizations for presentation slides
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_pipeline import create_datasets, get_dataset_info
from eda_analysis import run_eda

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset paths
DATASET_ROOT = r"c:\Users\saifa\Desktop\Studies\F20PA - Research Methods and Requirements Engineering\Dissertation\Re-think recycle dataset_organized"
METADATA_FILE = "dataset_metadata.csv"
EDA_OUTPUT_DIR = "eda_outputs"

# Dataset parameters
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function.
    """
    print("\n" + "=" * 80)
    print("RETHINK RECYCLE DATASET PIPELINE - COMPLETE WORKFLOW")
    print("=" * 80 + "\n")
    
    # ========================================================================
    # STEP 1: CREATE DATASET PIPELINE
    # ========================================================================
    print("STEP 1: Creating Dataset Pipeline")
    print("-" * 80)
    
    try:
        train_ds, val_ds, test_ds, class_names = create_datasets(
            root_dir=DATASET_ROOT,
            batch_size=BATCH_SIZE,
            image_size=IMAGE_SIZE,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO,
            random_seed=RANDOM_SEED
        )
        
        print("[OK] Dataset pipeline created successfully!")
        print(f"  Classes: {class_names}")
        print(f"  Batch size: {BATCH_SIZE}")
        print(f"  Image size: {IMAGE_SIZE}\n")
        
    except Exception as e:
        print(f"[ERROR] Error creating dataset pipeline: {e}")
        sys.exit(1)
    
    # ========================================================================
    # STEP 2: INSPECT DATASET
    # ========================================================================
    print("STEP 2: Dataset Information")
    print("-" * 80)
    
    try:
        info = get_dataset_info(DATASET_ROOT)
        
        print(f"Total images: {info['total_images']}")
        print(f"Number of materials: {info['num_materials']}")
        print(f"Number of item folders: {info['num_items']}")
        print(f"\nMaterial distribution:")
        for material, count in sorted(info['material_counts'].items()):
            percentage = (count / info['total_images']) * 100
            print(f"  {material:15} : {count:3d} images ({percentage:5.1f}%)")
        print()
        
    except Exception as e:
        print(f"[ERROR] Error getting dataset info: {e}")
        sys.exit(1)
    
    # ========================================================================
    # STEP 3: RUN EDA
    # ========================================================================
    print("STEP 3: Running Exploratory Data Analysis (EDA)")
    print("-" * 80)
    
    try:
        run_eda(
            metadata_file=METADATA_FILE,
            output_dir=EDA_OUTPUT_DIR,
            num_sample_materials=4,
            images_per_material=4,
            dimension_sample_size=500
        )
        
        print("[OK] EDA completed successfully!\n")
        
    except Exception as e:
        print(f"[ERROR] Error running EDA: {e}")
        sys.exit(1)
    
    # ========================================================================
    # STEP 4: SUMMARY
    # ========================================================================
    print("=" * 80)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 80)
    print()
    print("Generated outputs:")
    print(f"  - Metadata: {METADATA_FILE}")
    print(f"  - EDA plots: {EDA_OUTPUT_DIR}/")
    print()
    print("Next steps:")
    print("  1. Review EDA plots in the 'eda_outputs' directory")
    print("  2. Use train_ds, val_ds, test_ds for model training")
    print("  3. Refer to dataset_pipeline.py for dataset loading in your training script")
    print()


if __name__ == "__main__":
    main()
