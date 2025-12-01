# ReThink Recycle - ML Dataset Pipeline & EDA

Complete Python project for loading, preprocessing, and analyzing the ReThink Recycle waste classification dataset.

## Project Structure

```
.
├── dataset_pipeline.py          # Dataset loader & TensorFlow preprocessing
├── eda_analysis.py              # Exploratory data analysis script
├── main.py                      # Main orchestration script
├── requirements.txt             # Python dependencies
├── dataset_metadata.csv         # Generated metadata (after running pipeline)
├── eda_outputs/                 # Generated EDA visualizations
│   ├── eda_material_counts.png
│   ├── eda_sample_grid.png
│   ├── eda_width_hist.png
│   ├── eda_height_hist.png
│   └── item_counts.csv
└── ML_PROJECT_README.md         # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

```bash
python main.py
```

This will:
- Discover all images in the dataset
- Create stratified train/val/test splits (70/15/15)
- Generate metadata CSV
- Create TensorFlow datasets with preprocessing & augmentation
- Run EDA and generate visualizations

### 3. Use the Dataset in Your Training Script

```python
from dataset_pipeline import create_datasets

# Create datasets
train_ds, val_ds, test_ds, class_names = create_datasets(batch_size=32)

# Use in your model
model.fit(train_ds, validation_data=val_ds, epochs=10)
```

## Dataset Overview

### Structure
```
Re-think recycle dataset_organized/
├── Ceramic/
│   ├── 014_Ceramic/
│   │   └── #014瓷器/
│   │       ├── image1.jpg
│   │       ├── image2.jpg
│   │       └── ...
├── Glass/
├── Metal/
├── Mixed/
├── Organic/
├── Paper/
├── Plastic/
├── Rubber/
├── Styrofoam/
├── Textile/
└── Wood/
```

### Statistics

| Material | Images | Percentage |
|----------|--------|-----------|
| Mixed | 2384 | 30.0% |
| Plastic | 1806 | 22.7% |
| Paper | 1011 | 12.7% |
| Textile | 726 | 9.1% |
| Ceramic | 706 | 8.9% |
| Organic | 389 | 4.9% |
| Metal | 323 | 4.1% |
| Rubber | 263 | 3.3% |
| Wood | 167 | 2.1% |
| Styrofoam | 120 | 1.5% |
| Glass | 58 | 0.7% |
| **Total** | **7953** | **100%** |

### Image Dimensions
- **Width**: min=28px, max=4787px, mean=305px, median=214px
- **Height**: min=34px, max=2776px, mean=359px, median=287px
- **Target size**: 224×224 (resized in preprocessing)

## Module Details

### `dataset_pipeline.py`

**Main Function:**
```python
train_ds, val_ds, test_ds, class_names = create_datasets(
    root_dir=ROOT_DIR,
    batch_size=32,
    image_size=(224, 224),
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42
)
```

**Key Features:**
- Automatically discovers all images in the organized directory structure
- Creates stratified train/val/test splits (ensures balanced material distribution)
- Implements TensorFlow preprocessing pipeline:
  - Image loading and decoding
  - Normalization to [0, 1]
  - Resizing to 224×224
- Data augmentation for training set:
  - Random horizontal flip
  - Random brightness adjustment (±20%)
  - Random rotation (±10°)
- Efficient batching and prefetching with `tf.data.AUTOTUNE`

**Output:**
- `dataset_metadata.csv`: Full dataset metadata with filepaths, materials, and item folders
- TensorFlow Dataset objects ready for model training

### `eda_analysis.py`

**Main Function:**
```python
run_eda(
    metadata_file="dataset_metadata.csv",
    output_dir="eda_outputs",
    num_sample_materials=4,
    images_per_material=4,
    dimension_sample_size=500
)
```

**Generates:**

1. **eda_material_counts.png**
   - Bar chart showing image count per material
   - Useful for understanding class distribution
   - For presentation: Slide 2 (Dataset Overview)

2. **eda_sample_grid.png**
   - 4×4 grid of sample images from 4 random materials
   - Shows visual variation in lighting, backgrounds, object appearance
   - For presentation: Slide 3 (Sample Images)

3. **eda_width_hist.png** & **eda_height_hist.png**
   - Histograms of original image dimensions
   - Justifies the choice of 224×224 resizing
   - For presentation: Slide 4 (Data Preprocessing)

4. **item_counts.csv**
   - Image count per item folder
   - For appendix/reference

## Configuration

Edit the following constants in `dataset_pipeline.py`:

```python
ROOT_DIR = r"path\to\Re-think recycle dataset_organized"
IMAGE_SIZE = (224, 224)          # Target image size
BATCH_SIZE = 32                  # Batch size
TRAIN_RATIO = 0.70               # 70% training
VAL_RATIO = 0.15                 # 15% validation
TEST_RATIO = 0.15                # 15% testing
RANDOM_SEED = 42                 # For reproducibility
```

## Usage Examples

### Example 1: Basic Dataset Loading

```python
from dataset_pipeline import create_datasets

# Create datasets with default parameters
train_ds, val_ds, test_ds, class_names = create_datasets()

print(f"Classes: {class_names}")
# Output: Classes: ['Ceramic', 'Glass', 'Metal', 'Mixed', 'Organic', 'Paper', 'Plastic', 'Rubber', 'Styrofoam', 'Textile', 'Wood']

# Inspect a batch
for images, labels in train_ds.take(1):
    print(f"Images shape: {images.shape}")  # (32, 224, 224, 3)
    print(f"Labels shape: {labels.shape}")  # (32,)
```

### Example 2: Custom Batch Size and Image Size

```python
from dataset_pipeline import create_datasets

# Create datasets with custom parameters
train_ds, val_ds, test_ds, class_names = create_datasets(
    batch_size=64,
    image_size=(256, 256)
)
```

### Example 3: Get Dataset Info Without Creating TF Datasets

```python
from dataset_pipeline import get_dataset_info

info = get_dataset_info()
print(f"Total images: {info['total_images']}")
print(f"Material distribution: {info['material_counts']}")
```

### Example 4: Training a Simple Model

```python
import tensorflow as tf
from dataset_pipeline import create_datasets

# Create datasets
train_ds, val_ds, test_ds, class_names = create_datasets()

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(train_ds, validation_data=val_ds, epochs=10)

# Evaluate
model.evaluate(test_ds)
```

## Data Splits

The dataset is split using **stratified sampling** to ensure each split has similar material distribution:

- **Training**: 5567 images (70%)
- **Validation**: 1193 images (15%)
- **Test**: 1193 images (15%)

Stratification ensures that rare classes (e.g., Glass with 58 images) are represented in all splits.

## Preprocessing Pipeline

### For All Sets:
1. Read image file
2. Decode JPEG/PNG
3. Convert to float32
4. Normalize to [0, 1]
5. Resize to 224×224

### For Training Only (Augmentation):
1. Random horizontal flip (50% probability)
2. Random brightness adjustment (±20%)
3. Random rotation (±10°)

### Batching & Prefetching:
- Shuffle training data (buffer size = dataset size)
- Batch with specified batch size
- Prefetch with `tf.data.AUTOTUNE` for optimal performance

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'tensorflow'"
**Solution:** Install dependencies with `pip install -r requirements.txt`

### Issue: "FileNotFoundError: dataset_metadata.csv"
**Solution:** Run `dataset_pipeline.py` first to generate metadata

### Issue: No images found
**Solution:** Verify the `ROOT_DIR` path in `dataset_pipeline.py` points to the correct dataset location

### Issue: Out of memory
**Solution:** Reduce `BATCH_SIZE` in `dataset_pipeline.py` or `IMAGE_SIZE`

## Performance Tips

1. **Use GPU**: TensorFlow will automatically use GPU if available
2. **Adjust batch size**: Larger batches = faster training but more memory
3. **Prefetching**: Already enabled with `tf.data.AUTOTUNE`
4. **Caching**: For small datasets, consider caching in memory:
   ```python
   dataset = dataset.cache()
   ```

## References

- TensorFlow Documentation: https://www.tensorflow.org/
- Image Preprocessing: https://www.tensorflow.org/tutorials/images/classification
- Data Augmentation: https://www.tensorflow.org/tutorials/images/data_augmentation

## Notes

- All file paths use absolute paths for reproducibility
- Metadata CSV includes all images for full traceability
- Random seed (42) ensures reproducible splits across runs
- Images are normalized to [0, 1] for better training stability

---

**Created for:** ReThink Recycle Waste Classification Project  
**Last Updated:** 2024
