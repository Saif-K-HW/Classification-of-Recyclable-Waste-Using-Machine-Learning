# Classification of Recyclable Waste Using Transfer Learning

This repository implements a full dissertation-ready image classification pipeline with:

- Stratified 70/15/15 data splits
- `tf.data` loading with on-the-fly preprocessing
- Baseline CNN + ResNet50 frozen + ResNet50 fine-tuned stable pipeline
- Reproducible training and evaluation artifacts
- Error analysis outputs for Chapter 4 discussion
- Clean global vs per-model artifact separation

## Project structure

```text
.
├── data/
│   ├── raw/
│   └── splits/
├── models/
├── results/
│   ├── plots/
│   │   ├── global/
│   │   ├── baseline_cnn/
│   │   ├── resnet50_frozen/
│   │   └── resnet50_finetuned/
│   ├── metrics/
│   │   ├── global/
│   │   ├── baseline_cnn/
│   │   ├── resnet50_frozen/
│   │   └── resnet50_finetuned/
│   ├── logs/
│   └── misclassified/
│       └── resnet50_finetuned/
├── experiments/
│   ├── exp01_backbones/
│   ├── exp02_augmentation_ResNet50/
│   ├── exp03_class_imbalance/
│   ├── exp04_finetune_schedule/
│   └── exp05_calibration/
├── dashboard/
│   ├── app.py
│   ├── utils.py
│   ├── launch_dashboard.bat
│   └── launch_dashboard.ps1
├── src/
│   ├── config.py
│   ├── make_splits.py
│   ├── eda.py
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── error_analysis.py
│   ├── calibration.py
│   ├── experiment_utils.py
│   ├── predict.py
│   └── main.py
├── requirements.txt
└── README.md
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run order (recommended)

```bash
# Run once (creates split CSV indexes)
python src/main.py --mode make_splits

# Dataset analysis outputs
python src/main.py --mode eda

# Trains baseline, frozen ResNet50, then fine-tuned ResNet50 (stable default uses no augmentation)
python src/main.py --mode train

# Test-set metrics and model comparison table
python src/main.py --mode evaluate

# Uses best model from evaluation_summary by default
python src/main.py --mode error_analysis
```

## Final pipeline (stable)

This is the stable baseline pipeline. It still writes to `results/` and is not overwritten by experiment runs.

- Default stable backbone: `resnet50`
- Default stable augmentation: `none`
- Backbone comparison (`exp01_backbones`) showed ResNet50 as the best overall model on test accuracy and Macro F1.
- Augmentation ablation showed no augmentation is the best stable default.

Cross-platform (recommended):

```bash
python src/main.py --mode run_all
```

Bash runner (Git Bash/WSL/macOS/Linux):

```bash
bash scripts/run_all.sh
```

PowerShell runner (Windows):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1
```

If splits already exist, `run_all` will skip split creation.

## Experiments

Each experiment has its own folder and override config under `experiments/<exp_name>/`.

Run any experiment with:

```bash
python src/main.py --mode experiment --exp_name <exp_name>
```

Available experiment names:

- `exp01_backbones`
- `exp02_augmentation_ResNet50`
- `exp03_class_imbalance`
- `exp04_finetune_schedule`
- `exp05_calibration`

Examples:

```bash
python src/main.py --mode experiment --exp_name exp01_backbones
python src/main.py --mode experiment --exp_name exp02_augmentation_ResNet50
python src/main.py --mode experiment --exp_name exp03_class_imbalance
python src/main.py --mode experiment --exp_name exp04_finetune_schedule
python src/main.py --mode experiment --exp_name exp05_calibration
```

Experiment outputs are isolated and saved only under:

- `experiments/<exp_name>/results/metrics/...`
- `experiments/<exp_name>/results/plots/...`
- `experiments/<exp_name>/results/logs/...`
- `experiments/<exp_name>/results/misclassified/...`

Default mode behavior is unchanged:

- `python src/main.py --mode run_all` -> writes to `results/`
- `python src/main.py --mode experiment ...` -> writes to `experiments/<exp_name>/results/`

### Calibration mode

You can also run calibration directly:

```bash
python src/main.py --mode calibration --model_path models/resnet50_finetuned_best.keras
```

Calibration outputs include:

- `reliability_curve.png`
- `calibration_metrics.csv`
- `confidence_examples.csv`

### How to compare baseline vs experiments

1. Use baseline summary from `results/metrics/global/model_comparison.csv`.
2. Use experiment summary from `experiments/<exp_name>/results/metrics/global/model_comparison.csv`.
3. Compare macro F1, accuracy, and confusion/error-analysis outputs.

## Single-image prediction

```bash
python src/main.py --mode predict --image_path "path/to/image.jpg"
```

Optional overrides:

```bash
python src/main.py --mode predict \
  --image_path "path/to/image.jpg" \
  --model_path "models/resnet50_finetuned_best.keras" \
  --save_csv "results/metrics/global/prediction_examples.csv"
```

## Dashboard (Streamlit)

Use the dashboard for interactive prediction and artifact review.

Quick launch (Windows):

```bash
dashboard/launch_dashboard.bat
```

PowerShell launch:

```powershell
powershell -ExecutionPolicy Bypass -File dashboard/launch_dashboard.ps1
```

Or run manually:

```bash
python -m streamlit run dashboard/app.py
```

Dashboard features include:

- Sidebar navigation pages (Overview, Prediction, Metrics, Error Analysis, Calibration, Artifacts)
- Live single-image prediction with top-3 confidence output
- Scope switch between stable outputs and experiment outputs
- Calibration artifact generation button for missing files
- Error-analysis visuals with confusion plots and misclassified image gallery
- Live artifact reading from disk with Refresh support

Dashboard implementation files:

- `dashboard/app.py` -> Streamlit UI and page rendering
- `dashboard/utils.py` -> scope discovery, artifact loading, prediction helpers
- `dashboard/launch_dashboard.bat` -> one-click launcher (Windows)
- `dashboard/launch_dashboard.ps1` -> PowerShell launcher

## Notes

- Seeds are set in `config.py` and reused across scripts for reproducibility.
- Split CSVs store file indexes only; source images are never moved.
- The stable final model pointer is saved in `models/final_model.txt` and targets `resnet50_finetuned_best.keras` after default training.
- Global metrics live in `results/metrics/global/`, while per-model reports live in `results/metrics/<model_name>/`.
- Global plots live in `results/plots/global/`, while per-model curves and confusion plots live in `results/plots/<model_name>/`.
- For experiment runs, the same artifact structure is mirrored under `experiments/<exp_name>/results/`.
