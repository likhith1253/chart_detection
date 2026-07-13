# Repository Migration Audit

## Phase 1: Full Repository Audit

This document provides a comprehensive categorization of all files in the chart understanding research repository.

---

## Root Level Files

### KEEP

**README.md**
- **Reason**: High-quality project documentation describing capabilities, modules, and output structure
- **Action**: Keep and update to reflect new architecture

**config.yaml**
- **Reason**: Well-structured YAML configuration with comprehensive pipeline parameters
- **Action**: Keep as template, restructure for new architecture

**config.smoke.yaml**
- **Reason**: Useful for quick smoke testing
- **Action**: Keep, rename to `config.smoke_test.yaml`

**yolov8n.pt**
- **Reason**: Pretrained YOLOv8 weights for chart element detection
- **Action**: Keep, move to `models/pretrained/`

### REWRITE

**config.py**
- **Reason**: Contains hardcoded paths and global constants mixed with configuration
- **Action**: Rewrite as proper configuration module using dataclasses/pydantic

**main.py**
- **Reason**: Entry point needs to be simplified for new modular architecture
- **Action**: Rewrite to use new pipeline orchestration

### DELETE

**debug_pipeline.py**
- **Reason**: Independent testing script - functionality should be in unit tests
- **Action**: Delete, replace with proper pytest unit tests

**demo_experiment.py**
- **Reason**: Demo script with presentation artifacts - not research-grade
- **Action**: Delete, functionality should be in examples/ directory

**research_pipeline_final.py**
- **Reason**: 829-line monolithic pipeline - violates single responsibility principle
- **Action**: Delete, functionality moved to modular pipeline

**research_pipeline_v3.py**
- **Reason**: Duplicate/earlier version of research pipeline - dead code
- **Action**: Delete, superseded by modular pipeline

**download_test_images.py**
- **Reason**: Test data generation - should be in dataset management module
- **Action**: Delete, functionality moved to dataset manager

---

## src/pipeline/

### KEEP

**src/pipeline/config.py**
- **Reason**: Well-designed typed configuration loader with dataclasses
- **Action**: Keep, extend for new architecture

**src/pipeline/reproducibility.py**
- **Reason**: Clean reproducibility utilities (seeding, git commit, library versions)
- **Action**: Keep, no changes needed

### REWRITE

**src/pipeline/research_pipeline.py**
- **Reason**: 442-line orchestrator with mixed responsibilities (processing, evaluation, ML training)
- **Action**: Rewrite as pure orchestrator, delegate to specialized modules

---

## src/ocr/

### KEEP

**src/ocr/ocr_engine.py**
- **Reason**: Robust OCR wrapper with PaddleOCR/EasyOCR ensemble, fallback, retries, caching
- **Action**: Keep, excellent design

**src/ocr/ensemble.py**
- **Reason**: Clean token-level fusion with confidence arbitration
- **Action**: Keep, no changes needed

---

## src/classifier/

### KEEP

**src/classifier/chart_classifier.py**
- **Reason**: Well-structured heuristic classifier with combined features
- **Action**: Keep, minor refactoring for modularity

**src/classifier/ml_classifier.py**
- **Reason**: Research-grade ML classifier with cross-validation, multiple models
- **Action**: Keep, consolidate with advanced_ml.py

---

## src/extraction/

### KEEP

**src/extraction/value_extractor.py**
- **Reason**: Clean value extraction for all chart types using OpenCV
- **Action**: Keep, no changes needed

---

## src/research/

### MERGE

**src/research/advanced_ml.py** → **src/classifier/ml_classifier.py**
- **Reason**: Duplicate ML training functionality - advanced_ml.py has GridSearchCV but ml_classifier.py is simpler
- **Action**: Merge GridSearchCV capability into ml_classifier.py, delete advanced_ml.py

**src/research/cnn_classifier.py** → **src/classifier/cnn_classifier.py**
- **Reason**: CNN classifier should be in classifier module, not research module
- **Action**: Move to src/classifier/, keep functionality

**src/research/yolo_chart_detector.py** → **src/detection/yolo_detector.py**
- **Reason**: YOLO detection should be in detection module
- **Action**: Merge with existing yolo_detector.py, delete research version

**src/research/hybrid_classifier.py** → **src/classifier/hybrid_classifier.py**
- **Reason**: Hybrid classifier belongs in classifier module
- **Action**: Move to src/classifier/

**src/research/dataset_verifier.py** → **src/dataset/dataset_verifier.py**
- **Reason**: Dataset verification belongs in dataset module
- **Action**: Move to src/dataset/

---

## src/evaluation/

### KEEP

**src/evaluation/research_evaluator.py**
- **Reason**: Clean research-grade evaluation metrics
- **Action**: Keep, no changes needed

### DELETE

**src/evaluation/evaluator.py**
- **Reason**: Duplicate/older evaluator - research_evaluator.py is more comprehensive
- **Action**: Delete, use research_evaluator.py

---

## src/preprocessing/

### KEEP

**src/preprocessing/image_preprocessor.py**
- **Reason**: Clean configurable preprocessing with all standard operations
- **Action**: Keep, no changes needed

---

## src/segmentation/

### KEEP

**src/segmentation/chart_element_segmenter.py**
- **Reason**: Research-grade segmentation with noise filtering, Hough transforms, DBSCAN
- **Action**: Keep, excellent implementation

---

## src/feature_extraction/

### MERGE

**src/feature_extraction/geometric_features.py** + **src/features/chart_feature_extractor.py**
- **Reason**: Two feature extraction modules with overlapping functionality
- **Action**: Merge into single module, keep best features from both
- **Decision**: Keep geometric_features.py (more comprehensive), integrate chart_feature_extractor.py's metrics

---

## src/caching/

### KEEP

**src/caching/ocr_cache.py**
- **Reason**: Clean disk-backed cache with metadata tracking
- **Action**: Keep, no changes needed

---

## src/dataset/

### KEEP

**src/dataset/dataset_manager.py**
- **Reason**: Excellent dataset management with synthetic chart generation
- **Action**: Keep, no changes needed

**src/dataset/versioning.py**
- **Reason**: Clean dataset fingerprinting and deterministic splits
- **Action**: Keep, no changes needed

---

## src/visualization/

### KEEP

**src/visualization/research_plots.py**
- **Reason**: Clean research-grade plot generator
- **Action**: Keep, no changes needed

### DELETE

**src/visualization/plots.py**
- **Reason**: Duplicate plotting functionality - research_plots.py is sufficient
- **Action**: Delete

**src/visualization/debug_visualizer.py**
- **Reason**: Debug visualization should be in examples or debug tools, not main module
- **Action**: Move to tools/debug_visualizer.py or delete

---

## src/detection/

### KEEP

**src/detection/yolo_detector.py**
- **Reason**: Clean thread-safe adapter for YOLO detection
- **Action**: Keep, merge with yolo_chart_detector.py

---

## src/profiling/

### KEEP

**src/profiling/runtime_profiler.py**
- **Reason**: Clean runtime profiling with context manager
- **Action**: Keep, no changes needed

---

## src/__init__.py files

### KEEP

All `__init__.py` files are empty or minimal - keep for package structure.

---

## External Dependencies

### DELETE

**ocr_env/**
- **Reason**: Virtual environment should not be in repository
- **Action**: Delete, document dependencies in requirements.txt

---

## Summary Statistics

| Category | Count |
|----------|-------|
| KEEP | 20 |
| REWRITE | 3 |
| MERGE | 5 |
| DELETE | 9 |
| MOVE | 4 |

---

## High-Value Components to Preserve

1. **OCR Engine** (`src/ocr/ocr_engine.py`) - Robust ensemble with fallback
2. **Segmentation** (`src/segmentation/chart_element_segmenter.py`) - Research-grade element detection
3. **Feature Extraction** (`src/feature_extraction/geometric_features.py`) - Comprehensive geometric features
4. **Dataset Management** (`src/dataset/dataset_manager.py`) - Synthetic generation with rich variety
5. **Configuration** (`src/pipeline/config.py`) - Typed configuration system
6. **Reproducibility** (`src/pipeline/reproducibility.py`) - Clean snapshot utilities
7. **Caching** (`src/caching/ocr_cache.py`) - Efficient disk-backed cache
8. **Evaluation** (`src/evaluation/research_evaluator.py`) - Research metrics
9. **Visualization** (`src/visualization/research_plots.py`) - Publication plots
10. **Runtime Profiling** (`src/profiling/runtime_profiler.py`) - Clean profiling

---

## Technical Debt Identified

1. **Monolithic pipelines**: research_pipeline_final.py (829 lines), research_pipeline_v3.py (551 lines)
2. **Duplicate functionality**: Multiple evaluators, multiple feature extractors, multiple ML trainers
3. **Dead code**: Debug scripts, demo scripts, older pipeline versions
4. **Hardcoded configuration**: config.py mixes paths with constants
5. **Module organization**: Research module contains classifiers, detectors, dataset tools (wrong locations)
6. **Missing unit tests**: Only debug_pipeline.py for testing
7. **Virtual environment in repo**: ocr_env/ should not be committed

---

## Next Steps

1. **Phase 2**: Design new modular architecture
2. **Phase 3**: Detailed reuse analysis of KEEP components
3. **Phase 4**: Research methodology redesign
4. **Phase 5**: Migration plan with freeze/legacy strategy
5. **Phase 6**: Implementation milestones
