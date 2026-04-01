# Exp03 — Class Imbalance Strategies

## Aim
Test whether imbalance handling improves macro F1 and minority-class behavior.

## What changed
Three runs on `mobilenetv2`:
1. Uncapped class weights + cross-entropy
2. Capped class weights (`cap = 10`) + cross-entropy
3. Capped class weights (`cap = 10`) + focal loss (`gamma = 2.0`)

## Expected effect
- Weight capping should reduce over-penalization from extreme minority weights.
- Focal loss should push learning toward harder mistakes.

## How to run
```bash
python src/main.py --mode experiment --exp_name exp03_class_imbalance
```

## Outputs
Saved only under:
- `experiments/exp03_class_imbalance/results/metrics/...`
- `experiments/exp03_class_imbalance/results/plots/...`
- `experiments/exp03_class_imbalance/results/logs/...`
- `experiments/exp03_class_imbalance/results/misclassified/...`
