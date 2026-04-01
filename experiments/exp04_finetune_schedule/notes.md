# Exp04 — Fine-Tuning Schedule

## Aim
Check whether a deeper/longer fine-tune schedule improves final transfer performance.

## What changed
- Backbone: `mobilenetv2`
- Fine-tune overrides:
  - `finetune_unfreeze_layers = 40`
  - `finetune_epochs = 15`
  - `finetune_lr = 5e-6`
- Early stopping remains enabled.

## Expected effect
Potentially better adaptation with slower updates, at the cost of more training time.

## How to run
```bash
python src/main.py --mode experiment --exp_name exp04_finetune_schedule
```

## Outputs
Saved only under:
- `experiments/exp04_finetune_schedule/results/metrics/...`
- `experiments/exp04_finetune_schedule/results/plots/...`
- `experiments/exp04_finetune_schedule/results/logs/...`
- `experiments/exp04_finetune_schedule/results/misclassified/...`
