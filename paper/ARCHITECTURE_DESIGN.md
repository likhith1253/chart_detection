# New Architecture Design

## Phase 2: Brand New Modular Architecture

This document describes the redesigned architecture for the chart understanding research repository, designed from scratch to be modular, reproducible, and publication-ready.

---

## Design Principles

1. **Single Responsibility**: Each module has one clear purpose
2. **Modularity**: Components are loosely coupled and independently testable
3. **Reproducibility**: All experiments are fully reproducible with snapshots
4. **CPU-First**: Optimized for CPU with optional GPU acceleration
5. **Minimal Dependencies**: Only essential libraries, no technical debt
6. **Publication-Ready**: Built-in artifact generation for papers
7. **Understandable**: New researcher can understand structure in <1 hour

---

## Folder Hierarchy

```
chart_research_project/
├── README.md                          # Project documentation
├── requirements.txt                   # Minimal dependencies
├── pyproject.toml                     # Modern Python project config
├── config/
│   ├── default.yaml                   # Default configuration
│   ├── smoke_test.yaml                # Quick test configuration
│   └── ablation_variants.yaml         # Ablation study configurations
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py                # Configuration loader (dataclasses)
│   │   └── constants.py               # Global constants
│   ├── core/
│   │   ├── __init__.py
│   │   ├── pipeline.py                # Main pipeline orchestrator
│   │   ├── experiment.py              # Experiment runner
│   │   └── reproducibility.py        # Snapshot and seeding utilities
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                 # Dataset management
│   │   ├── versioning.py              # Dataset fingerprinting
│   │   └── synthetic.py                # Synthetic chart generation
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── image.py                   # Image preprocessing
│   ├── ocr/
│   │   ├── __init__.py
│   │   ├── engine.py                  # OCR engine wrapper
│   │   └── ensemble.py                # OCR ensemble fusion
│   ├── detection/
│   │   ├── __init__.py
│   │   └── segmenter.py               # Traditional segmentation
│   ├── features/
│   │   ├── __init__.py
│   │   ├── geometric.py               # Geometric feature extraction
│   │   └── structural.py              # Structural feature extraction
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── heuristic.py               # Rule-based classifier
│   │   ├── ml.py                      # ML classifier (RF/SVM/GB/XGB)
│   │   ├── cnn.py                     # CNN transfer learning
│   │   └── hybrid.py                  # Hybrid ensemble classifier
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Evaluation metrics
│   │   ├── statistical.py             # Statistical analysis
│   │   └── analysis.py                # Error analysis
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plots.py                   # Research plots
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging.py                 # Logging setup
│   │   ├── caching.py                 # Generic caching utilities
│   │   └── profiling.py               # runtime profiling
│   └── models/
│       ├── __init__.py
│       └── loader.py                  # Model loading utilities
├── models/
│   └── pretrained/
├── data/
│   ├── raw_images/                    # Input images
│   └── datasets/                      # Processed datasets
├── cache/
│   └── ocr/                           # OCR cache
├── results/
│   ├── experiments/                   # Experiment outputs
│   ├── models/                        # Trained models
│   ├── plots/                         # Generated plots
│   └── reports/                       # Research reports
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_ocr.py
│   ├── test_classification.py
│   └── test_pipeline.py
├── examples/
│   ├── quick_start.py                 # Quick start example
│   └── custom_experiment.py           # Custom experiment example
└── scripts/
    ├── train_classifier.py            # Train classifier script
    ├── run_ablation.py                # Run ablation study
    └── generate_report.py             # Generate paper report
```

---

## Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     Main Entry Point                         │
│                    (scripts/run_experiment.py)               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Configuration Loader                          │
│              (src/config/settings.py)                         │
│  - Load YAML config                                          │
│  - Validate with dataclasses                                 │
│  - Set global seed                                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Experiment Runner                           │
│                  (src/core/experiment.py)                     │
│  - Build reproducibility snapshot                             │
│  - Initialize dataset                                         │
│  - Initialize pipeline components                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     Pipeline Orchestrator                      │
│                     (src/core/pipeline.py)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ Preprocessing│  │     OCR      │  │  Detection   │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│         │                 │                 │                │
│         ▼                 ▼                 ▼                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  Features    │  │ Classification│  │              │        │
│  └──────────────┘  └──────────────┘  │              │        │
│         │                 │              │              │        │
│         └─────────────────┴──────────────┘              │        │
│                           │                          │        │
│                           ▼                          │        │
│                  ┌──────────────┐                    │        │
│                  │  Evaluation  │                    │        │
│                  └──────────────┘                    │        │
│                           │                          │        │
│                           ▼                          │        │
│                  ┌──────────────┐                    │        │
│                  │  Artifacts   │                    │        │
│                  └──────────────┘                    │        │
└─────────────────────────────────────────────────────────────┘
```

---

## Pipeline Stages

### Stage 1: Preprocessing
- **Module**: `src/preprocessing/image.py`
- **Input**: Raw image path
- **Output**: Preprocessed image (numpy array)
- **Operations**: Grayscale, CLAHE, adaptive threshold, denoise, morphology, deskew
- **Configurable**: Yes, via YAML

### Stage 2: OCR
- **Module**: `src/ocr/engine.py`
- **Input**: Preprocessed image
- **Output**: OCR results (text, confidence, bboxes)
- **Engines**: PaddleOCR, EasyOCR (ensemble)
- **Caching**: Yes, disk-backed with hash keys

### Stage 3: Detection
- **Module**: `src/detection/segmenter.py`
- **Input**: Preprocessed image
- **Output**: Element counts (bars, lines, points, pie slices, bins, axes)
- **Methods**: Traditional segmentation (morphological filtering, Hough transforms, DBSCAN)

### Stage 4: Feature Extraction
- **Module**: `src/features/geometric.py` and `src/features/structural.py`
- **Input**: Preprocessed image, detection results
- **Output**: Feature vector (30 features)
- **Features**: Geometric (15), structural (8), OCR-based (4), histogram-specific (3)

### Stage 5: Classification
- **Module**: `src/classification/` (heuristic.py, ml.py, cnn.py, hybrid.py)
- **Input**: Feature vector, detection results
- **Output**: Chart type prediction with confidence
- **Methods**: Heuristic rules, ML (RF/SVM/GB/XGB), CNN (EfficientNet/ResNet), Hybrid ensemble

### Stage 6: Evaluation
- **Module**: `src/evaluation/metrics.py`
- **Input**: Predictions, ground truth (classification)
- **Output**: Metrics (accuracy, precision, recall, F1, confusion matrix, histogram vs. bar disambiguation)

### Stage 7: Artifact Generation
- **Module**: `src/visualization/plots.py`
- **Input**: Metrics, results
- **Output**: Plots, reports, CSVs, JSONs

---

## Configuration System

### Structure

```yaml
# config/default.yaml
experiment:
  name: "baseline_experiment"
  seed: 42
  description: "Baseline chart classification experiment"

dataset:
  raw_images_dir: "data/raw_images"
  min_size: 1000
  split_ratio:
    train: 0.7
    val: 0.15
    test: 0.15
  split_seed: 42

preprocessing:
  enabled: true
  grayscale: true
  clahe: true
  adaptive_threshold: true
  denoise: true
  morphology_cleanup: true
  deskew: true

ocr:
  enabled_engines: ["paddleocr", "easyocr"]
  ensemble_enabled: true
  paddle_retry_count: 1
  disable_mkldnn_on_retry: true
  force_engine: null

detection:
  method: "yolo"  # "yolo" or "traditional"
  yolo:
    imgsz: 512
    device: "cpu"
    confidence_threshold: 0.25
  traditional:
    enabled: true  # fallback if YOLO unavailable

classification:
  methods: ["heuristic", "ml", "cnn", "hybrid"]
  heuristic:
    enabled: true
  ml:
    enabled: true
    models: ["randomforest", "svm", "gradientboosting", "xgboost"]
    cv_folds: 5
  cnn:
    enabled: true
    backbone: "efficientnet_b0"  # or "resnet50"
    image_size: 224
    batch_size: 16
    epochs: 20
    early_stopping_patience: 4
  hybrid:
    enabled: true
    weights:
      ml: 0.5
      cnn: 0.3
      yolo: 0.15
      heuristic: 0.05

qa:
  enabled: true
  question_types: ["value_retrieval", "comparison", "trend", "aggregation", "histogram"]
  templates_file: "config/qa_templates.yaml"
  histogram_specific: true

cache:
  enabled: true
  root_dir: "cache"
  ocr_subdir: "ocr"
  invalidate: false

parallel:
  enabled: true
  worker_count: 0  # 0 = auto
  max_worker_cap: 32
  executor_type: "thread"

outputs:
  results_dir: "results"
  experiments_dir: "results/experiments"
  models_dir: "results/models"
  plots_dir: "results/plots"
  reports_dir: "results/reports"

logging:
  level: "INFO"
  structured_json: true
  file_name: "pipeline.log"

ablation:
  enabled: true
  variants_file: "config/ablation_variants.yaml"
```

### Implementation

```python
# src/config/settings.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml

@dataclass
class ExperimentConfig:
    name: str = "baseline_experiment"
    seed: int = 42
    description: str = ""

@dataclass
class DatasetConfig:
    raw_images_dir: str = "data/raw_images"
    min_size: int = 1000
    split_ratio: Dict[str, float] = field(default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15})
    split_seed: int = 42

@dataclass
class PreprocessingConfig:
    enabled: bool = True
    grayscale: bool = True
    clahe: bool = True
    adaptive_threshold: bool = True
    denoise: bool = True
    morphology_cleanup: bool = True
    deskew: bool = True

@dataclass
class OCRConfig:
    enabled_engines: List[str] = field(default_factory=lambda: ["paddleocr", "easyocr"])
    ensemble_enabled: bool = True
    paddle_retry_count: int = 1
    disable_mkldnn_on_retry: bool = True
    force_engine: Optional[str] = None

@dataclass
class DetectionConfig:
    method: str = "yolo"
    yolo: Dict = field(default_factory=dict)
    traditional: Dict = field(default_factory=lambda: {"enabled": True})

@dataclass
class ClassificationConfig:
    methods: List[str] = field(default_factory=lambda: ["heuristic", "ml", "cnn", "hybrid"])
    heuristic: Dict = field(default_factory=lambda: {"enabled": True})
    ml: Dict = field(default_factory=lambda: {"enabled": True, "models": ["randomforest", "svm", "gradientboosting", "xgboost"], "cv_folds": 5})
    cnn: Dict = field(default_factory=lambda: {"enabled": True, "backbone": "efficientnet_b0", "image_size": 224, "batch_size": 16, "epochs": 20, "early_stopping_patience": 4})
    hybrid: Dict = field(default_factory=lambda: {"enabled": True, "weights": {"ml": 0.5, "cnn": 0.3, "yolo": 0.15, "heuristic": 0.05}})

@dataclass
class CacheConfig:
    enabled: bool = True
    root_dir: str = "cache"
    ocr_subdir: str = "ocr"
    invalidate: bool = False

@dataclass
class ParallelConfig:
    enabled: bool = True
    worker_count: int = 0
    max_worker_cap: int = 32
    executor_type: str = "thread"

@dataclass
class OutputsConfig:
    results_dir: str = "results"
    experiments_dir: str = "results/experiments"
    models_dir: str = "results/models"
    plots_dir: str = "results/plots"
    reports_dir: str = "results/reports"

@dataclass
class LoggingConfig:
    level: str = "INFO"
    structured_json: bool = True
    file_name: str = "pipeline.log"

@dataclass
class AblationConfig:
    enabled: bool = True
    variants_file: str = "config/ablation_variants.yaml"

@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
```

---

## Logging System

### Implementation

```python
# src/utils/logging.py
import logging
import sys
from pathlib import Path
import json
from typing import Any

def setup_logging(config: LoggingConfig, log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("chart_research")
    logger.setLevel(getattr(logging, config.level))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.level))
    
    if config.structured_json:
        console_formatter = JsonFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_dir / config.file_name)
    file_handler.setLevel(getattr(logging, config.level))
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "file": record.pathname,
            "line": record.lineno,
        }
        if hasattr(record, "event"):
            log_obj["event"] = record.event
        return json.dumps(log_obj)
```

---

## Checkpoint and Result Management

### Checkpoint System

```python
# src/core/experiment.py
import json
from pathlib import Path
from datetime import datetime, timezone
from src.core.reproducibility import build_experiment_snapshot

class Experiment:
    def __init__(self, config: Config, experiment_dir: Path):
        self.config = config
        self.experiment_dir = experiment_dir
        self.checkpoint_dir = experiment_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize experiment
        self._initialize_experiment()
    
    def _initialize_experiment(self):
        # Create experiment metadata
        metadata = {
            "experiment_name": self.config.experiment.name,
            "description": self.config.experiment.description,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "config": self.config.to_dict(),
        }
        
        # Save metadata
        metadata_path = self.experiment_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
        
        # Build reproducibility snapshot
        snapshot = build_experiment_snapshot(self.config)
        snapshot_path = self.experiment_dir / "reproducibility_snapshot.json"
        snapshot_path.write_text(json.dumps(snapshot, indent=2))
    
    def save_checkpoint(self, stage: str, data: dict):
        checkpoint_path = self.checkpoint_dir / f"{stage}.json"
        checkpoint = {
            "stage": stage,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        checkpoint_path.write_text(json.dumps(checkpoint, index=2))
    
    def load_checkpoint(self, stage: str) -> dict:
        checkpoint_path = self.checkpoint_dir / f"{stage}.json"
        if checkpoint_path.exists():
            return json.loads(checkpoint_path.read_text())
        return None
```

### Result Management

```python
# src/core/pipeline.py
class Pipeline:
    def __init__(self, config: Config, experiment: Experiment):
        self.config = config
        self.experiment = experiment
        self.results = []
    
    def run(self, images: List[Path]) -> Dict:
        for image in images:
            result = self.process_image(image)
            self.results.append(result)
        
        # Save results
        results_df = pd.DataFrame(self.results)
        results_path = self.experiment.experiment_dir / "results.csv"
        results_df.to_csv(results_path, index=False)
        
        # Generate summary
        summary = self._generate_summary(results_df)
        summary_path = self.experiment.experiment_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        
        return summary
```

---

## Paper Artifact Generation

### Artifacts Generated

1. **Confusion Matrix** - Heatmap of classification performance
2. **Performance Metrics** - Accuracy, precision, recall, F1 per class
3. **Histogram vs. Bar Disambiguation** - Specific analysis of histogram/bar distinction
4. **Ablation Study Results** - Table comparing variants
5. **Runtime Analysis** - Per-stage timing breakdown
6. **Feature Importance** - Top features from ML models
7. **Error Analysis** - Common error patterns (classification)
8. **Dataset Statistics** - Distribution of chart types, sources
9. **Reproducibility Report** - Full experiment snapshot
10. **Histogram Analysis** - Histogram-specific performance and insights

### Implementation

```python
# src/visualization/plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

class ResearchPlotGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")
    
    def generate_all(self, results_df: pd.DataFrame, metrics: dict) -> int:
        count = 0
        count += self.plot_confusion_matrix(metrics)
        count += self.plot_performance_metrics(metrics)
        count += self.plot_runtime_breakdown(metrics)
        count += self.plot_feature_importance(metrics)
        count += self.plot_dataset_distribution(results_df)
        return count
    
    def plot_confusion_matrix(self, metrics: dict) -> bool:
        cm = metrics.get("classification", {}).get("confusion_matrix")
        labels = metrics.get("classification", {}).get("labels")
        
        if not cm or not labels:
            return False
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        
        path = self.output_dir / "confusion_matrix.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return True
```

---

## Core Pipeline

```python
# src/core/pipeline.py
from pathlib import Path
from typing import List, Dict
from src.config.settings import Config
from src.preprocessing.image import ImagePreprocessor
from src.ocr.engine import OCREngine
from src.detection.yolo import YOLODetector
from src.features.geometric import GeometricFeatureExtractor
from src.classification.hybrid import HybridClassifier
from src.evaluation.metrics import Evaluator

class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize components
        self.preprocessor = ImagePreprocessor(config.preprocessing)
        self.ocr_engine = OCREngine(config.ocr, self.config.cache)
        self.detector = YOLODetector(config.detection.yolo)
        self.feature_extractor = GeometricFeatureExtractor()
        self.classifier = HybridClassifier(config.classification)
        self.evaluator = Evaluator()
    
    def process_image(self, image_path: Path) -> Dict:
        # Stage 1: Preprocessing
        preprocessed = self.preprocessor.process(image_path)
        
        # Stage 2: OCR
        ocr_results = self.ocr_engine.run_ocr(image_path)
        
        # Stage 3: Detection
        detection_results = self.detector.detect(image_path)
        
        # Stage 4: Feature Extraction
        features = self.feature_extractor.extract(image_path)
        
        # Stage 5: Classification
        classification = self.classifier.classify(features, detection_results)
        
        return {
            "image_path": str(image_path),
            "ocr_results": ocr_results,
            "detection_results": detection_results,
            "features": features,
            "classification": classification,
        }
```

---

## CPU-First Design

### GPU Usage Strategy

1. **Optional GPU**: All components work on CPU by default
2. **GPU Detection**: Auto-detect GPU availability
3. **Graceful Fallback**: If GPU unavailable, use CPU
4. **Configuration**: User can force CPU via config

### Implementation

```python
# src/utils/device.py
import torch

def get_device(preferred: str = "auto") -> str:
    if preferred == "cpu":
        return "cpu"
    
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    
    if preferred == "auto" and torch.cuda.is_available():
        return "cuda"
    
    return "cpu"
```

---

## Minimal Dependencies

### Core Dependencies

```
# requirements.txt
numpy>=1.24.0
pandas>=2.0.0
opencv-python>=4.8.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
pillow>=10.0.0
scipy>=1.11.0
```

### Optional Dependencies

```
# requirements-optional.txt
# OCR
easyocr>=1.7.0
paddleocr>=2.7.0

# Deep Learning
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0

# ML
xgboost>=2.0.0
```

---

## Reproducibility Guarantees

1. **Deterministic Seeding**: All random operations seeded
2. **Dataset Fingerprinting**: Hash-based dataset identification
3. **Configuration Snapshot**: Full config saved with results
4. **Library Versioning**: All library versions recorded
5. **Git Commit**: Git hash recorded in snapshot
6. **Code Version**: Experiment tied to specific commit

---

## Testing Strategy

### Unit Tests

```python
# tests/test_ocr.py
import pytest
from src.ocr.engine import OCREngine
from src.config.settings import OCRConfig

def test_ocr_engine_initialization():
    config = OCRConfig(enabled_engines=["easyocr"])
    engine = OCREngine(config)
    assert engine is not None

def test_ocr_engine_run():
    config = OCRConfig(enabled_engines=["easyocr"])
    engine = OCREngine(config)
    # Test with sample image
    result = engine.run_ocr("tests/fixtures/sample_chart.png")
    assert "text" in result
```

### Integration Tests

```python
# tests/test_pipeline.py
import pytest
from src.core.pipeline import Pipeline
from src.config.settings import Config

def test_pipeline_end_to_end():
    config = Config.from_yaml("config/smoke_test.yaml")
    pipeline = Pipeline(config)
    # Test with sample dataset
    results = pipeline.run([Path("tests/fixtures/sample_chart.png")])
    assert len(results) == 1
```

---

## Summary

The new architecture provides:

- **Modularity**: 15+ independent modules with single responsibilities
- **Reproducibility**: Full snapshot and seeding support
- **CPU-First**: All components work on CPU with optional GPU
- **Minimal Dependencies**: Core deps only, optional for advanced features
- **Publication-Ready**: Built-in artifact generation
- **Understandable**: Clear structure, documented interfaces
- **Testable**: Unit and integration test framework
