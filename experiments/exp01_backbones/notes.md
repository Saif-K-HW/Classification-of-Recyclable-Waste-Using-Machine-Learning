# Exp01 — Backbone Comparison

## Aim
Compare transfer-learning backbones under the same training setup.

## What changed
- Backbones: `mobilenetv2`, `efficientnetb0`, `resnet50`, `densenet121`
- Same head and same default frozen + fine-tune schedule for all runs

## Expected effect
Shows whether architecture choice alone improves macro F1 and overall accuracy.

## How to run
```bash
python src/main.py --mode experiment --exp_name exp01_backbones
```

## Outputs
Saved only under:
- `experiments/exp01_backbones/results/metrics/...`
- `experiments/exp01_backbones/results/plots/...`
- `experiments/exp01_backbones/results/logs/...`
- `experiments/exp01_backbones/results/misclassified/...`
