# Chart Classification and Parsing Research Framework

A research codebase for understanding chart images: classify the chart type (bar, histogram, line, pie, scatter), detect/crop the chart, run OCR, and extract data with type-specific parsers. The classification part has trained baselines and reported metrics; the parser, detection and orchestration layers are implemented and unit-tested but have not been scored on a labelled benchmark in this repository.

## What is implemented

| Area | Where | Status |
|---|---|---|
| Synthetic chart generation | `src/generate_synthetic.py`, `scripts/generate_dataset.py` | working, configurable via `configs/*.yaml` |
| Unified dataset manifest, splits, fingerprinting | `src/dataset/`, `src/pipeline/reproducibility.py` | working |
| Hand-crafted features + classical ML classifiers | `src/features/`, `src/feature_extraction/`, `src/classifier/` | trained and evaluated (below) |
| CNN baseline (EfficientNet-B0, transfer learning) | `src/research/cnn_classifier.py` | trained and evaluated (below) |
| Chart detection (YOLO backend, manifest loaders, adapters for COCO/YOLO/VOC) | `src/detection/` | implemented, unit-tested with stub detectors; no trained detector results in this repo |
| OCR layer (backend interface, caching, post-processing) | `src/ocr/` | implemented |
| Chart-type router and parser engine; bar, boxplot, histogram, line, pie, scatter parsers | `src/router/`, `src/parser_engine/`, `src/parsers/` | implemented, unit-tested on synthetic inputs |
| End-to-end pipeline orchestrator and evaluation framework | `src/pipeline/`, `src/evaluation/` | implemented, unit-tested |

## Results (from `results/`, one run each)

Test split of the unified dataset: 989 images, 5 classes.

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic regression | 0.9596 | 0.9575 |
| SVM (RBF) | 0.9788 | 0.9765 |
| Random forest | 0.9848 | 0.9830 |
| XGBoost | 0.9889 | 0.9871 |
| EfficientNet-B0 (fine-tuned, CPU) | 0.9990 | 0.9990 |

XGBoost's weakest class is `scatter_plot` (per-class accuracy 0.9565; 7 of 161 scatter plots predicted as line charts). The CNN misclassified 1 of 989 test images (a bar chart predicted as pie). Sources: `results/classification/metrics.json`, `results/cnn_baseline/cnn_metrics.json`. These result folders are git-ignored; regenerate them with the commands below.

## Limitations (please read before quoting the numbers)

* **The dataset labels are heuristic and the data is largely easy.** All 6,590 images in `data/datasets/unified_dataset_splits.csv` carry `label_source = filename_heuristic`; 2,979 of them are synthetic charts (`synthetic_*.png`) and 6,584 come from the `local_raw` collection (only 6 from PlotQA). The train/validation/test split is 4,612 / 989 / 989. Accuracy near 99% on this split says little about real-world charts.
* **No leakage audit across splits** (for example near-duplicate renderings of the same synthetic template) was run for the classification split, so the CNN's 99.9% may be optimistic.
* **One run, no confidence intervals.** Results are single training runs without repeated seeds or cross-validation.
* The detection, OCR and parser components are not evaluated on labelled real charts here, so no extraction-accuracy claim is made.
* Several tests need local, git-ignored datasets and are skipped when the data is absent.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
pip install -e .
python -m pytest tests -q       # 95 passed in about 30 s on a CPU machine (no data required)
```

## Usage

```bash
# synthetic dataset (config controls count, styles and splits)
python scripts/generate_dataset.py --config configs/default.yaml
python scripts/generate_dataset.py --config configs/smoke_test.yaml    # tiny, for a quick check

# training / evaluation (writes to results/ and logs/)
python kaggle_train.py --split-csv data/datasets/unified_dataset_splits.csv
```

`kaggle_train.py` expects the repository-relative layout `data/raw_images` and `data/datasets` (see `docs/kaggle_setup.md`). Design notes and the research plan are in `paper/` and `docs/`.

## License

MIT, see [LICENSE](LICENSE). Datasets keep their original licences (ChartQA, PlotQA, DVQA and FigureQA are used under their own terms).
