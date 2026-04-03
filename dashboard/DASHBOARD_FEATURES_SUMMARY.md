# Dashboard Features Summary

## Purpose

This dashboard provides a single visual interface for:

- Running model prediction on uploaded images
- Reviewing training and evaluation performance
- Inspecting error-analysis outputs
- Checking calibration quality
- Monitoring artifact freshness after new pipeline runs

It is designed to work with both stable pipeline outputs and experiment outputs.

---

## Core Design Goals

1. **Modern UI/UX**
   - Custom visual theme with gradient background and card-based layout
   - Clear sidebar navigation menu
   - Sidebar controls for scope switching and fast refresh

2. **Live artifact sync**
   - Reads metrics/plots directly from disk each run
   - No hardcoded static dashboards
   - Reflects latest generated CSV and image artifacts

3. **Reusability with existing code**
   - Uses your existing prediction pipeline logic from `src/predict.py`
   - Uses project output conventions (`results/...`, `experiments/...`)

4. **Submission-ready reporting support**
   - Combines quantitative tables and qualitative visual outputs
   - Provides one place to capture screenshots for report chapters

---

## Scope Model (Data Sources)

The dashboard supports multiple scopes:

- **Stable Pipeline (default)**
  - `results/`
  - `models/`

- **Experiment Scope(s)**
  - `experiments/<exp_name>/results/`
  - `experiments/<exp_name>/results/models/`

This lets you switch quickly between baseline/stable and experiment outputs.

---

## Feature Breakdown by Tab

## 1) Prediction Tab

### Inputs
- Uploaded image file (`jpg`, `jpeg`, `png`, `bmp`, `webp`)
- Selected model checkpoint (`.keras`) from current scope

### Behavior
- Runs inference via existing prediction code path
- Returns top-3 class predictions with confidence
- Visualizes top-3 confidence as a horizontal bar chart

### Outputs
- On-screen top prediction and confidence KPI cards
- Optional CSV append to:
  - `<scope>/metrics/global/prediction_examples_dashboard.csv`

---

## 2) Metrics Tab

### Artifacts loaded
- `metrics/global/model_comparison.csv`
- `metrics/global/evaluation_summary.csv`
- `metrics/<model_name>/classification_report.csv`

### Visuals
- Sortable model comparison and evaluation tables
- Accuracy vs Macro-F1 scatter chart
- Per-model classification report viewer

---

## 3) Error Analysis Tab

### Artifacts loaded
- `metrics/global/top_confusions.csv`
- `metrics/global/worst_classes.csv`
- `plots/<model_name>/confusion_matrix.png`
- `plots/<model_name>/common_confusions.png` (if available)
- `misclassified/<model_name>/` image samples

### Visuals
- Confusion summary tables
- Confusion matrix + common confusion image panels
- Misclassified image gallery (latest files)

---

## 4) Calibration Tab

### Artifacts loaded
- `metrics/global/calibration_metrics.csv`
- `metrics/global/reliability_bins.csv`
- `plots/global/reliability_curve.png`

### Visuals
- Calibration metrics table
- Reliability curve image
- Interactive confidence vs accuracy trend plot from reliability bins

---

## 5) Artifacts Tab

### Purpose
- Quick health/status check of major dashboard dependencies

### Displays
- Artifact name
- Relative path
- Availability status (Available/Missing)
- Last updated timestamp

Useful for confirming that latest runs produced all expected files.

---

## Refresh and Update Behavior

- Dashboard reads files from disk on each rerun.
- Click **Refresh now** in sidebar after any pipeline run.
- Launch scripts use:
  - `--server.runOnSave true`
- This means saving dashboard code also triggers automatic rerun.

---

## Launch and Access

## One-click launch
- `dashboard/launch_dashboard.bat` (double-click)

## PowerShell launch
- `dashboard/launch_dashboard.ps1`

## Manual launch
```bash
python -m streamlit run dashboard/app.py
```

---

## Implementation Files

- `dashboard/app.py` -> Streamlit UI, tabs, charts, artifact viewer
- `dashboard/utils.py` -> scope discovery, file loading, prediction runner helpers
- `dashboard/launch_dashboard.bat` -> Windows shortcut-style launcher
- `dashboard/launch_dashboard.ps1` -> PowerShell launcher
- `README.md` -> consolidated usage instructions

---

## Integration Notes

- The dashboard is non-invasive: it does not replace training/evaluation scripts.
- It consumes existing outputs and can be used throughout experimentation.
- Prediction uses your current model pointer conventions, class-name mapping, and saved checkpoints.
