# Migration Plan

## Phase 5: Detailed Migration Strategy

This document provides a step-by-step migration plan from the current repository to the new modular architecture.

---

## Migration Strategy Overview

### Approach

**Strategy**: Incremental migration with legacy support
- Create new folder structure alongside existing code
- Migrate components one module at a time
- Maintain backward compatibility during transition
- Delete old code only after validation
- Use feature flags to switch between old/new implementations

### Migration Phases

1. **Phase 1**: Infrastructure setup (new folder structure, config system)
2. **Phase 2**: Core utilities migration (logging, caching, reproducibility)
3. **Phase 3**: Data layer migration (dataset, versioning, synthetic generation)
4. **Phase 4**: Preprocessing and OCR migration
5. **Phase 5**: Detection and feature extraction migration
6. **Phase 6**: Classification migration
7. **Phase 7**: Evaluation and visualization migration
8. **Phase 8**: Pipeline orchestration migration
9. **Phase 9**: Testing and validation
10. **Phase 10**: Legacy cleanup

### Risk Mitigation

- **Git branches**: Use feature branches for each phase
- **Rollback**: Keep old code until validation complete
- **Testing**: Unit tests for each migrated component
- **Documentation**: Update docs as code is migrated
- **Communication**: Clear commit messages and PR descriptions

---

## Phase 1: Infrastructure Setup (Days 1-2)

### Goals

- Create new folder structure
- Set up new configuration system
- Initialize new module structure
- Set up testing framework

### Tasks

#### Task 1.1: Create New Folder Structure

```bash
# Create new directories
mkdir -p src/config
mkdir -p src/core
mkdir -p src/data
mkdir -p src/preprocessing
mkdir -p src/ocr
mkdir -p src/detection
mkdir -p src/features
mkdir -p src/classification
mkdir -p src/extraction
mkdir -p src/evaluation
mkdir -p src/visualization
mkdir -p src/utils
mkdir -p src/models
mkdir -p tests
mkdir -p examples
mkdir -p scripts
mkdir -p config
mkdir -p models/pretrained
mkdir -p data/datasets
mkdir -p results/experiments
mkdir -p results/models
mkdir -p results/plots
mkdir -p results/reports
```

**Validation**: All directories created, no conflicts with existing structure

#### Task 1.2: Set Up Configuration System

**Files to Create**:
- `config/default.yaml` - Default configuration
- `config/smoke_test.yaml` - Quick test configuration
- `config/ablation_variants.yaml` - Ablation study configurations
- `src/config/__init__.py` - Package init
- `src/config/settings.py` - Configuration dataclasses
- `src/config/constants.py` - Global constants

**Implementation**:
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

# ... (other config classes as defined in ARCHITECTURE_DESIGN.md)

@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    # ... (other config sections)
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
```

**Validation**: Config loads from YAML, all fields accessible

#### Task 1.3: Initialize Module Structure

**Files to Create**:
- `src/__init__.py`
- `src/core/__init__.py`
- `src/data/__init__.py`
- `src/preprocessing/__init__.py`
- `src/ocr/__init__.py`
- `src/detection/__init__.py`
- `src/features/__init__.py`
- `src/classification/__init__.py`
- `src/extraction/__init__.py`
- `src/evaluation/__init__.py`
- `src/visualization/__init__.py`
- `src/utils/__init__.py`
- `src/models/__init__.py`
- `tests/__init__.py`

**Validation**: All modules importable without errors

#### Task 1.4: Set Up Testing Framework

**Files to Create**:
- `tests/conftest.py` - Pytest fixtures
- `tests/test_config.py` - Config tests
- `tests/__init__.py`

**Implementation**:
```python
# tests/conftest.py
import pytest
from pathlib import Path

@pytest.fixture
def sample_config():
    return {
        "experiment": {"name": "test", "seed": 42},
        "dataset": {"raw_images_dir": "data/raw_images"},
    }

@pytest.fixture
def sample_image_path():
    return Path("tests/fixtures/sample_chart.png")
```

**Validation**: Pytest runs, fixtures work

**Estimated Time**: 2 days

---

## Phase 2: Core Utilities Migration (Days 3-4)

### Goals

- Migrate logging system
- Migrate caching utilities
- Migrate reproducibility utilities
- Migrate runtime profiler

### Tasks

#### Task 2.1: Migrate Logging System

**Source**: Create new implementation based on ARCHITECTURE_DESIGN.md
**Target**: `src/utils/logging.py`

**Implementation**:
```python
# src/utils/logging.py
import logging
import sys
from pathlib import Path
import json
from typing import Any

def setup_logging(config: LoggingConfig, log_dir: Path):
    # Implementation from ARCHITECTURE_DESIGN.md
    pass

class JsonFormatter(logging.Formatter):
    # Implementation from ARCHITECTURE_DESIGN.md
    pass
```

**Validation**: Logging works, JSON formatter works, file output works

#### Task 2.2: Migrate Caching Utilities

**Source**: `src/caching/ocr_cache.py`
**Target**: `src/utils/caching.py`

**Changes**:
- Generalize to generic `DiskCache` class
- Keep `OCRCache` as subclass
- Update imports to new config

**Implementation**:
```python
# src/utils/caching.py
from pathlib import Path
from typing import Any, Dict, Optional
import hashlib
import json
import threading

class DiskCache:
    """Generic disk-backed cache"""
    def __init__(self, root_dir: Path, subdir: str):
        self.root_dir = Path(root_dir)
        self.cache_dir = self.root_dir / subdir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Dict]:
        # Generic implementation
        pass
    
    def set(self, key: str, payload: Dict) -> None:
        # Generic implementation
        pass

class OCRCache(DiskCache):
    """OCR-specific cache with salt for preprocessing config"""
    @staticmethod
    def image_hash(image_path: Path, salt: str = "") -> str:
        # OCR-specific implementation
        pass
```

**Validation**: Cache works, thread-safe, metadata tracking works

#### Task 2.3: Migrate Reproducibility Utilities

**Source**: `src/pipeline/reproducibility.py`
**Target**: `src/core/reproducibility.py`

**Changes**: Update imports to new config system

**Validation**: Seeding works, git commit detection works, snapshot generation works

#### Task 2.4: Migrate Runtime Profiler

**Source**: `src/profiling/runtime_profiler.py`
**Target**: `src/utils/profiling.py`

**Changes**: No changes, just move file

**Validation**: Profiler works, context manager works, JSON serialization works

**Estimated Time**: 2 days

---

## Phase 3: Data Layer Migration (Days 5-7)

### Goals

- Migrate dataset manager
- Migrate dataset versioning
- Refactor synthetic generation
- Migrate dataset verifier

### Tasks

#### Task 3.1: Migrate Dataset Versioning

**Source**: `src/dataset/versioning.py`
**Target**: `src/data/versioning.py`

**Changes**: No changes, just move file

**Validation**: Fingerprinting works, deterministic splits work

#### Task 3.2: Refactor Dataset Manager

**Source**: `src/dataset/dataset_manager.py`
**Target**: Split into `src/data/dataset.py` and `src/data/synthetic.py`

**Changes**:
- Split dataset management and synthetic generation
- Update imports to new config
- Simplify interface

**Implementation**:
```python
# src/data/dataset.py
from pathlib import Path
from src.config.settings import DatasetConfig
from src.data.synthetic import SyntheticChartGenerator

class DatasetManager:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.raw_dir = Path(config.raw_images_dir)
        self.generator = SyntheticChartGenerator(config)
    
    def ensure_dataset(self) -> int:
        # Check existing, call generator if needed
        pass

# src/data/synthetic.py
from pathlib import Path
from src.config.settings import SyntheticConfig

class SyntheticChartGenerator:
    def __init__(self, config: SyntheticConfig):
        self.config = config
    
    def generate_bar_chart(self, path: Path, idx: int) -> None:
        # Existing generation logic
        pass
    
    # ... other generation methods
```

**Validation**: Dataset generation works, metadata generation works

#### Task 3.3: Migrate Dataset Verifier

**Source**: `src/research/dataset_verifier.py`
**Target**: `src/data/dataset_verifier.py`

**Changes**: Update imports to new config

**Validation**: Verification works, balancing works, report generation works

**Estimated Time**: 3 days

---

## Phase 4: Preprocessing and OCR Migration (Days 8-10)

### Goals

- Migrate image preprocessor
- Migrate OCR engine
- Migrate OCR ensemble

### Tasks

#### Task 4.1: Migrate Image Preprocessor

**Source**: `src/preprocessing/image_preprocessor.py`
**Target**: `src/preprocessing/image.py`

**Changes**: Update imports to new config

**Validation**: All preprocessing steps work, config integration works

#### Task 4.2: Migrate OCR Engine

**Source**: `src/ocr/ocr_engine.py`
**Target**: `src/ocr/engine.py`

**Changes**: Update imports to new config, integrate new caching

**Validation**: OCR works, ensemble works, fallback works, caching works

#### Task 4.3: Migrate OCR Ensemble

**Source**: `src/ocr/ensemble.py`
**Target**: `src/ocr/ensemble.py`

**Changes**: No changes, just move file

**Validation**: Ensemble fusion works, token matching works

**Estimated Time**: 3 days

---

## Phase 5: Detection and Feature Extraction Migration (Days 11-14)

### Goals

- Migrate chart element segmenter
- Merge feature extractors
- Migrate YOLO detector

### Tasks

#### Task 5.1: Migrate Chart Element Segmenter

**Source**: `src/segmentation/chart_element_segmenter.py`
**Target**: `src/detection/segmenter.py`

**Changes**: Update imports to new config, extract thresholds to config

**Validation**: Segmentation works, all element types detected

#### Task 5.2: Merge Feature Extractors

**Source**: `src/feature_extraction/geometric_features.py` + `src/features/chart_feature_extractor.py`
**Target**: `src/features/geometric.py`

**Changes**: Merge features from both files, remove duplicates

**Validation**: All 35+ features extracted, no duplicates

#### Task 5.3: Merge YOLO Detectors

**Source**: `src/detection/yolo_detector.py` + `src/research/yolo_chart_detector.py`
**Target**: `src/detection/yolo.py`

**Changes**: Merge inference and training logic, keep thread-safe wrapper

**Validation**: YOLO detection works, training works (optional), fallback works

**Estimated Time**: 4 days

---

## Phase 6: Classification Migration (Days 15-20)

### Goals

- Migrate heuristic classifier
- Merge ML classifiers
- Migrate CNN classifier
- Migrate hybrid classifier

### Tasks

#### Task 6.1: Migrate Heuristic Classifier

**Source**: `src/classifier/chart_classifier.py`
**Target**: `src/classification/heuristic.py`

**Changes**: Simplify interface, update imports to new config

**Validation**: Classification works, confidence computation works

#### Task 6.2: Merge ML Classifiers

**Source**: `src/classifier/ml_classifier.py` + `src/research/advanced_ml.py`
**Target**: `src/classification/ml.py`

**Changes**: Merge GridSearchCV from advanced_ml.py, keep clean interface from ml_classifier.py

**Validation**: All models train, cross-validation works, feature importance works

#### Task 6.3: Migrate CNN Classifier

**Source**: `src/research/cnn_classifier.py`
**Target**: `src/classification/cnn.py`

**Changes**: Update imports to new config, simplify interface

**Validation**: CNN trains, inference works, early stopping works

#### Task 6.4: Migrate Hybrid Classifier

**Source**: `src/research/hybrid_classifier.py`
**Target**: `src/classification/hybrid.py`

**Changes**: Update to use new classifier interfaces, simplify weight optimization

**Validation**: Ensemble works, weight optimization works

**Estimated Time**: 6 days

---

## Phase 7: Evaluation and Visualization Migration (Days 21-23)

### Goals

- Migrate research evaluator
- Migrate research plot generator
- Delete duplicate evaluators/plots

### Tasks

#### Task 7.1: Migrate Research Evaluator

**Source**: `src/evaluation/research_evaluator.py`
**Target**: `src/evaluation/metrics.py`

**Changes**: Rename file, no other changes

**Validation**: All metrics compute correctly

#### Task 7.2: Migrate Research Plot Generator

**Source**: `src/visualization/research_plots.py`
**Target**: `src/visualization/plots.py`

**Changes**: No changes, just move file

**Validation**: All plots generate correctly

#### Task 7.3: Delete Duplicate Files

**Files to Delete**:
- `src/evaluation/evaluator.py`
- `src/visualization/plots.py` (old version)
- `src/visualization/debug_visualizer.py`

**Validation**: No import errors, tests still pass

**Estimated Time**: 3 days

---

## Phase 8: Pipeline Orchestration Migration (Days 24-27)

### Goals

- Create new pipeline orchestrator
- Create experiment runner
- Create entry point scripts
- Update main.py

### Tasks

#### Task 8.1: Create Pipeline Orchestrator

**Target**: `src/core/pipeline.py`

**Implementation**: Based on ARCHITECTURE_DESIGN.md

**Validation**: Pipeline runs end-to-end, all stages execute

#### Task 8.2: Create Experiment Runner

**Target**: `src/core/experiment.py`

**Implementation**: Based on ARCHITECTURE_DESIGN.md

**Validation**: Experiment runs, checkpoints saved, results saved

#### Task 8.3: Create Entry Point Scripts

**Files to Create**:
- `scripts/run_experiment.py` - Main experiment runner
- `scripts/train_classifier.py` - Train classifier
- `scripts/run_ablation.py` - Run ablation study
- `examples/quick_start.py` - Quick start example

**Validation**: Scripts run without errors

#### Task 8.4: Update main.py

**Changes**: Rewrite to use new pipeline

**Validation**: main.py runs, produces results

**Estimated Time**: 4 days

---

## Phase 9: Testing and Validation (Days 28-30)

### Goals

- Write unit tests for all modules
- Write integration tests
- Run smoke tests
- Validate against old results

### Tasks

#### Task 9.1: Write Unit Tests

**Files to Create**:
- `tests/test_config.py`
- `tests/test_logging.py`
- `tests/test_caching.py`
- `tests/test_dataset.py`
- `tests/test_preprocessing.py`
- `tests/test_ocr.py`
- `tests/test_detection.py`
- `tests/test_features.py`
- `tests/test_classification.py`
- `tests/test_evaluation.py`

**Validation**: All tests pass, coverage >80%

#### Task 9.2: Write Integration Tests

**Files to Create**:
- `tests/test_pipeline.py` - End-to-end pipeline test
- `tests/test_experiment.py` - Experiment runner test

**Validation**: Integration tests pass

#### Task 9.3: Run Smoke Tests

**Test**: Run quick experiment on small dataset

**Validation**: Experiment completes, results valid

#### Task 9.4: Validate Against Old Results

**Test**: Run same experiment with old and new pipeline, compare results

**Validation**: Results match within acceptable tolerance

**Estimated Time**: 3 days

---

## Phase 10: Legacy Cleanup (Days 31-32)

### Goals

- Delete old code files
- Delete old folder structure
- Update documentation
- Final validation

### Tasks

#### Task 10.1: Delete Old Code Files

**Files to Delete**:
- `config.py` (root level)
- `debug_pipeline.py`
- `demo_experiment.py`
- `research_pipeline_final.py`
- `research_pipeline_v3.py`
- `download_test_images.py`
- `src/pipeline/research_pipeline.py` (old monolithic pipeline)
- `src/research/` (all files migrated)
- `src/segmentation/` (migrated to detection)
- `src/feature_extraction/` (migrated to features)
- `src/caching/` (migrated to utils)
- `src/profiling/` (migrated to utils)
- `ocr_env/`

**Validation**: No import errors, all tests pass

#### Task 10.2: Delete Old Folder Structure

**Folders to Delete**:
- `src/pipeline/` (except config.py and reproducibility.py which are migrated)
- `src/research/`
- `src/segmentation/`
- `src/feature_extraction/`
- `src/caching/`
- `src/profiling/`

**Validation**: Clean folder structure, no orphaned files

#### Task 10.3: Update Documentation

**Files to Update**:
- `README.md` - Update to reflect new structure
- Create `MIGRATION_SUMMARY.md` - Document what was changed
- Update inline docstrings

**Validation**: Documentation accurate, examples work

#### Task 10.4: Final Validation

**Test**: Run full experiment on full dataset

**Validation**: Experiment completes successfully, results valid

**Estimated Time**: 2 days

---

## Rollback Plan

### If Migration Fails

1. **Git Branch Strategy**: Each phase on separate branch
2. **Rollback Command**: `git checkout previous_phase_branch`
3. **Data Safety**: No data deleted until final validation
4. **Checkpoint**: Save working state after each phase

### Rollback Triggers

- Critical test failures
- Performance degradation >20%
- Data corruption
- Unrecoverable errors

---

## Success Criteria

### Per Phase

- **Phase 1**: New structure created, config system works
- **Phase 2**: All utilities migrated and tested
- **Phase 3**: Data layer migrated, dataset generation works
- **Phase 4**: Preprocessing and OCR migrated
- **Phase 5**: Detection and features migrated
- **Phase 6**: All classifiers migrated
- **Phase 7**: Evaluation and visualization migrated
- **Phase 8**: Pipeline orchestration works
- **Phase 9**: All tests pass, coverage >80%
- **Phase 10**: Legacy code removed, final validation passes

### Overall

- All old code migrated or deleted
- All tests pass
- Coverage >80%
- Performance matches or exceeds old system
- Documentation updated
- Examples work

---

## Timeline Summary

| Phase | Days | Tasks |
|-------|------|-------|
| Phase 1: Infrastructure | 2 | Folder structure, config, modules, testing |
| Phase 2: Core Utilities | 2 | Logging, caching, reproducibility, profiling |
| Phase 3: Data Layer | 3 | Dataset, versioning, synthetic, verifier |
| Phase 4: Preprocessing/OCR | 3 | Preprocessor, OCR engine, ensemble |
| Phase 5: Detection/Features | 4 | Segmenter, features, YOLO |
| Phase 6: Classification | 6 | Heuristic, ML, CNN, hybrid |
| Phase 7: Evaluation/Viz | 3 | Evaluator, plots, cleanup |
| Phase 8: Pipeline | 4 | Orchestrator, experiment, scripts |
| Phase 9: Testing | 3 | Unit tests, integration tests, validation |
| Phase 10: Cleanup | 2 | Delete old code, update docs, final validation |
| **Total** | **32** | **All migration tasks** |

---

## Next Steps

1. **Phase 6**: Create implementation milestones
2. **Phase 7**: Produce comprehensive migration document
