# Repository Audit

## Scope

Audit performed on `D:\chart detection` during repository cleanup and production hardening. The goal is structural cleanup only: preserve research methodology, models, datasets, and evaluation logic.

## Summary

- Primary repository root: `D:\chart detection`
- Nested duplicate repository: `D:\chart detection\chart_research_project`
- Nested package-like duplicate folder: `D:\chart detection\chart_research_project\chart_research_project`
- Duplicate source trees detected between:
  - `src/` and `chart_research_project/src/`
  - `tests/` and `chart_research_project/tests/`
  - `scripts/` and `chart_research_project/scripts/`
- Duplicate configuration trees detected between:
  - `config/` and `chart_research_project/config/`
  - root `config.yaml` / `config.py` and nested `chart_research_project/config.yaml` / `chart_research_project/config.py`
- Duplicate documentation detected between root markdown files and mirrored files under `chart_research_project/`

## Duplicate Repositories

Detected `.git` directories:

1. `D:\chart detection\.git`
2. `D:\chart detection\chart_research_project\.git`

Required final state: keep only the outer repository.

Instruction before deleting nested `.git`:

1. Confirm the outer repository is the canonical repository.
2. Move any required tracked assets from `chart_research_project/` into the outer root.
3. Remove `D:\chart detection\chart_research_project\.git`.
4. Remove the remaining duplicate repository tree once all required assets are migrated.

## Duplicate / Nested Project Folders

- Outer project folder: `D:\chart detection`
- Full nested duplicate project: `D:\chart detection\chart_research_project`
- Additional nested folder: `D:\chart detection\chart_research_project\chart_research_project`

The nested repository contains the only populated production assets for:

- `data/`
- `results/`
- `logs/`
- `yolov8n.pt`
- `src/utils/dataset_paths.py` (missing from the outer `src/utils/`)

## Dataset Layout Findings

Current active dataset layout exists only under:

- `D:\chart detection\chart_research_project\data\raw_images`
- `D:\chart detection\chart_research_project\data\datasets`

Important manifest files found:

- `data/datasets/unified_dataset_splits.csv`
- `data/datasets/unified_dataset_manifest.csv`
- `data/datasets/unified_dataset_distribution.csv`
- `data/datasets/dataset_metadata.csv`
- `data/datasets/dataset_fingerprint.json`

Cleanup constraint:

- Do not remove anything required by `unified_dataset_splits.csv`.

Additional cleanup candidates found inside dataset storage:

- dataset-local Hugging Face cache folders under:
  - `data/datasets/figureqa/.cache`
  - `data/datasets/novachart/.cache`

## Environment / Generated Artifact Findings

Non-production directories found inside the nested repository:

- `.pytest_cache/`
- `__pycache__/`
- `ocr_env/` (embedded virtual environment)
- `chart_research.egg-info/`
- `cache/`

These should not remain as part of the cleaned production repository.

## Path Handling Findings

Good news:

- Most code already uses `pathlib`.

Issues found:

- `kaggle_train.py` performs fallback filesystem scanning via `rglob`.
- `src/generate_synthetic.py` hardcodes `chart_research_project/data/raw_images`.
- `src/download_datasets.py` hardcodes `data/temp_downloads`.
- `src/utils/runtime.py` and nested `dataset_paths.py` reference Kaggle paths directly.
- `src/config/constants.py` still points to `config/` instead of the desired `configs/`.
- `src/pipeline/config.py` defaults logs to `results/logs` instead of top-level `logs/`.

## Logging Findings

Current logging is split across incompatible conventions:

- root `logs/training.log`
- pipeline config writes to `results/logs/pipeline.log`
- CNN training writes to `logs/training.log`

Required final standard:

- `logs/training.log`
- `logs/evaluation.log`
- `logs/runtime.log`
- `logs/progress.log`

## Import / Test Consistency Findings

The outer repository has code drift:

- tests import `src.data.dataset`, but the current outer `src/` tree does not contain `src/data/`
- source code references both configuration systems:
  - `src/config/settings.py`
  - `src/pipeline/config.py`
  - top-level `config.py`

This means cleanup requires both structural consolidation and selective import/path fixes to restore a coherent production tree.

## Planned Consolidation Target

Final root structure:

- `src/`
- `data/`
- `results/`
- `configs/`
- `scripts/`
- `models/`
- `tests/`
- `paper/`
- `docs/`
- `README.md`
- `requirements.txt`
- `kaggle_train.py`

## Decision

Use the outer repository at `D:\chart detection` as canonical. Migrate required assets out of `chart_research_project/`, then remove the nested repository and duplicate source tree.
