# Kaggle Setup

1. Place the repository in `/kaggle/working`.
2. Ensure `data/datasets/unified_dataset_splits.csv` is available, or pass `--split-csv`.
3. Run:

```bash
python kaggle_train.py --output-dir /kaggle/working/results/cnn_baseline
```

Optional overrides:

```bash
python kaggle_train.py --split-csv /kaggle/input/<dataset>/unified_dataset_splits.csv --epochs 6 --batch-size 32 --image-size 160
```

Outputs:

- `logs/training.log`
- `results/cnn_baseline/checkpoints/latest_checkpoint.pt`
- `results/cnn_baseline/checkpoints/best_checkpoint.pt`
- `results/cnn_baseline/experiment.json`
- `results/cnn_baseline/cnn_metrics.json`
- `results/cnn_baseline/figures/*`
