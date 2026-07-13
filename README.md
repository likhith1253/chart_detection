# Chart Classification Research

Production-cleaned repository for chart understanding research. The cleanup preserves the existing research methodology, models, datasets, and evaluation logic while consolidating the project into a single reproducible repository.

## Installation

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

## Dataset

The repository uses one dataset layout only:

```text
data/
  raw_images/
  datasets/
```

Important dataset artifacts include:

- `data/datasets/unified_dataset_splits.csv`
- `data/datasets/unified_dataset_manifest.csv`
- `data/datasets/dataset_metadata.csv`
- `data/datasets/dataset_fingerprint.json`

Synthetic dataset configs:

- `configs/default.yaml`
- `configs/smoke_test.yaml`

Generate the synthetic dataset:

```bash
python scripts/generate_dataset.py --config configs/default.yaml
```

Smoke test generation:

```bash
python scripts/generate_dataset.py --config configs/smoke_test.yaml
```

## Training

Primary training entrypoint:

```bash
python kaggle_train.py --split-csv data/datasets/unified_dataset_splits.csv
```

Training outputs are written under `results/`, and long-running task logs are written under `logs/`.

## Evaluation

Evaluation artifacts remain under `results/`. Standardized logs live in:

- `logs/training.log`
- `logs/evaluation.log`
- `logs/runtime.log`
- `logs/progress.log`

## Kaggle

`kaggle_train.py` expects the same repository-relative dataset layout on Kaggle:

- `data/raw_images`
- `data/datasets`

It does not perform recursive filesystem scans. If your split manifest lives elsewhere, pass `--split-csv` explicitly.

## Project Structure

```text
project_root/
  src/
  data/
  results/
  configs/
  scripts/
  models/
  tests/
  paper/
  docs/
  README.md
  requirements.txt
  kaggle_train.py
```

## Paper Reproduction

Research notes, methodology, and migration/background documents were consolidated into:

- `paper/`
- `docs/`

This cleanup changes repository structure, paths, and documentation only. It does not introduce new research ideas or alter the experimental methodology.
