# Exp05 — Calibration & Confidence Analysis

## Aim
Assess how well model confidence reflects real correctness on the test set.

## What changed
- Runs calibration analysis only (`steps = ["calibration"]`)
- Computes reliability bins, ECE, and Brier score
- Exports top high-confidence correct and wrong samples

## Expected effect
Highlights confidence mismatch (confidence is not always accuracy), which is useful for deployment risk analysis.

## How to run
```bash
python src/main.py --mode experiment --exp_name exp05_calibration
```

## Outputs
Saved only under:
- `experiments/exp05_calibration/results/metrics/global/...`
- `experiments/exp05_calibration/results/plots/global/...`
