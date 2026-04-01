# Exp02 — Augmentation Ablation (ResNet50)

## Aim
Measure the effect of removing train-time augmentation while keeping the rest of the pipeline unchanged.

## What changed
- Backbone: `resnet50`
- Two runs are executed for a strict control:
  - `aug_on` with `augmentation = default`
  - `aug_off` with `augmentation = none`
- Training still uses shuffle, cache, and prefetch.

## Expected effect
No augmentation can help when data is clean and class visuals are stable, while augmentation may help robustness on noisier samples.

## How to run
```bash
python src/main.py --mode experiment --exp_name exp02_augmentation_ResNet50
```

The default command already runs the strict control comparison.

## Outputs
Saved only under:
- `experiments/exp02_augmentation_ResNet50/results/metrics/...`
- `experiments/exp02_augmentation_ResNet50/results/plots/...`
- `experiments/exp02_augmentation_ResNet50/results/logs/...`
- `experiments/exp02_augmentation_ResNet50/results/misclassified/...`
