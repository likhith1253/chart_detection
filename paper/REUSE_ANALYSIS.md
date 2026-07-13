# Reuse Analysis

## Phase 3: High-Quality Reusable Components

This document analyzes the high-value components identified in the audit and maps them to the new architecture.

---

## Component Mapping

### 1. OCR Engine

**Existing**: `src/ocr/ocr_engine.py` (417 lines)
**New Location**: `src/ocr/engine.py`

**Quality Assessment**: ⭐⭐⭐⭐⭐ Excellent
- Thread-safe with proper locking
- PaddleOCR/EasyOCR ensemble with fallback
- Retry logic with MKLDNN disable
- Configurable preprocessing integration
- Cache integration with hash keys
- Success rate tracking
- Clean text cleaning with deduplication

**Reuse Strategy**: 
- **Direct reuse** with minimal changes
- Update imports to new config system
- Remove dependency on old config.py
- Keep all core logic intact

**Required Changes**:
```python
# Change from:
import config
from src.pipeline.config import OCRConfig

# To:
from src.config.settings import OCRConfig
```

**Migration Effort**: Low (1-2 hours)

---

### 2. OCR Ensemble

**Existing**: `src/ocr/ensemble.py` (123 lines)
**New Location**: `src/ocr/ensemble.py`

**Quality Assessment**: ⭐⭐⭐⭐⭐ Excellent
- Token-level alignment with IoU matching
- Confidence arbitration
- Numeric token correction (O→0, I→1, etc.)
- Duplicate suppression with bbox+text matching
- Clean, well-documented functions

**Reuse Strategy**:
- **Direct reuse** with no changes
- Already independent of other modules
- Pure functions, no state

**Migration Effort**: None (0 hours)

---

### 3. Chart Element Segmenter

**Existing**: `src/segmentation/chart_element_segmenter.py` (348 lines)
**New Location**: `src/detection/segmenter.py`

**Quality Assessment**: ⭐⭐⭐⭐⭐ Excellent
- Morphological filtering
- Hough line detection for axes
- Connected component filtering for bars
- Circular Hough transform for pie charts
- DBSCAN clustering for scatter points
- Noise filtering with thresholds
- Axis exclusion zones

**Reuse Strategy**:
- **Direct reuse** with minimal changes
- Update imports to new config system
- Consider extracting constants to config

**Required Changes**:
```python
# Extract hardcoded thresholds to config
_MIN_CONTOUR_AREA = 200  # → config.detection.min_contour_area
_MIN_BAR_AREA = 400      # → config.detection.min_bar_area
```

**Migration Effort**: Low (2-3 hours)

---

### 4. Geometric Feature Extractor

**Existing**: `src/feature_extraction/geometric_features.py` (376 lines)
**New Location**: `src/features/geometric.py`

**Quality Assessment**: ⭐⭐⭐⭐⭐ Excellent
- 35+ geometric features
- Edge orientation entropy
- Axis numeric density
- Lightweight clustering
- Hough circle detection
- Radial density analysis
- Comprehensive feature set

**Reuse Strategy**:
- **Direct reuse** with minimal changes
- Merge with chart_feature_extractor.py metrics
- Update imports

**Merge with Chart Feature Extractor**:
```python
# Add these features from chart_feature_extractor.py:
- rectangle_count (already in geometric)
- bar_spacing_ratio (already in geometric)
- circle_count (already in geometric)
- avg_circularity (already in geometric)
- scatter_density (already in geometric)
- line_continuity_score (already in geometric)
- contour_area_entropy (already in geometric)
- contour_area_variance (already in geometric)
```

**Migration Effort**: Low (2-3 hours)

---

### 5. Dataset Manager

**Existing**: `src/dataset/dataset_manager.py` (348 lines)
**New Location**: `src/data/dataset.py`

**Quality Assessment**: ⭐⭐⭐⭐⭐ Excellent
- Rich synthetic chart generation
- 5 chart types with variety
- Randomized palettes, fonts, sizes
- Metadata generation
- Target distribution support
- Incremental generation

**Reuse Strategy**:
- **Direct reuse** with minimal changes
- Split into two modules:
  - `src/data/dataset.py` - Dataset management
  - `src/data/synthetic.py` - Synthetic generation
- Update imports to new config system

**Refactoring Plan**:
```python
# src/data/dataset.py
class DatasetManager:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.raw_dir = Path(config.raw_images_dir)
    
    def ensure_dataset(self) -> int:
        # Check existing, call synthetic generator if needed
        pass

# src/data/synthetic.py
class SyntheticChartGenerator:
    def __init__(self, config: SyntheticConfig):
        self.config = config
    
    def generate_bar_chart(self, path: Path, idx: int) -> None:
        # Existing generation logic
        pass
```

**Migration Effort**: Medium (4-5 hours)

---

### 6. Dataset Versioning

**Existing**: `src/dataset/versioning.py` (138 lines)
**New Location**: `src/data/versioning.py`

**Quality Assessment**: ⭐⭐⭐⭐⭐ Excellent
- Dataset fingerprinting with SHA256
- Source inference from filenames
- Chart type inference
- Deterministic splits with seeding
- Clean dataclass for DatasetImage

**Reuse Strategy**:
- **Direct reuse** with no changes
- Already independent and clean
- Keep dataclass structure

**Migration Effort**: None (0 hours)

---

### 7. Configuration Loader

**Existing**: `src/pipeline/config.py` (167 lines)
**New Location**: `src/config/settings.py`

**Quality Assessment**: ⭐⭐⭐⭐ Good
- Well-designed dataclasses
- YAML loading with deep merge
- Type-safe configuration
- Default values

**Reuse Strategy**:
- **Reuse structure**, expand for new architecture
- Add new config sections (detection, classification methods)
- Keep dataclass pattern
- Add validation

**Required Additions**:
```python
@dataclass
class DetectionConfig:
    method: str = "yolo"
    yolo: Dict = field(default_factory=dict)
    traditional: Dict = field(default_factory=dict)

@dataclass
class ClassificationConfig:
    methods: List[str] = field(default_factory=lambda: ["heuristic", "ml", "cnn", "hybrid"])
    heuristic: Dict = field(default_factory=dict)
    ml: Dict = field(default_factory=dict)
    cnn: Dict = field(default_factory=dict)
    hybrid: Dict = field(default_factory=dict)
```

**Migration Effort**: Low (2-3 hours)

---

### 8. Reproducibility Utilities

**Existing**: `src/pipeline/reproducibility.py` (81 lines)
**New Location**: `src/core/reproducibility.py`

**Quality Assessment**: ⭐⭐⭐⭐⭐ Excellent
- Global seeding (random, numpy, python hash)
- Git commit detection
- Library version detection
- Experiment snapshot generation
- Clean, minimal dependencies

**Reuse Strategy**:
- **Direct reuse** with no changes
- Already perfect for new architecture
- Move to core module

**Migration Effort**: None (0 hours)

---

### 9. OCR Cache

**Existing**: `src/caching/ocr_cache.py` (101 lines)
**New Location**: `src/utils/caching.py`

**Quality Assessment**: ⭐⭐⭐⭐⭐ Excellent
- Disk-backed cache with JSON
- SHA256 hash with salt support
- Thread-safe with locking
- Metadata tracking (hits, misses, entries)
- Invalidation support

**Reuse Strategy**:
- **Direct reuse** with minor generalization
- Make generic cache class
- Keep OCR-specific implementation as subclass

**Refactoring Plan**:
```python
# src/utils/caching.py
class DiskCache:
    """Generic disk-backed cache"""
    def __init__(self, root_dir: Path, subdir: str):
        # Generic implementation
        pass

class OCRCache(DiskCache):
    """OCR-specific cache with salt for preprocessing config"""
    def image_hash(self, image_path: Path, salt: str = "") -> str:
        # OCR-specific implementation
        pass
```

**Migration Effort**: Low (1-2 hours)

---

### 10. Image Preprocessor

**Existing**: `src/preprocessing/image_preprocessor.py` (97 lines)
**New Location**: `src/preprocessing/image.py`

**Quality Assessment**: ⭐⭐⭐⭐⭐ Excellent
- Configurable preprocessing stages
- All standard operations (grayscale, CLAHE, threshold, denoise, morphology, deskew)
- Clean process() and process_path() methods
- Integration with config

**Reuse Strategy**:
- **Direct reuse** with no changes
- Already matches new architecture
- Update imports to new config

**Migration Effort**: None (0 hours)

---

### 11. Heuristic Classifier

**Existing**: `src/classifier/chart_classifier.py` (290 lines)
**New Location**: `src/classification/heuristic.py`

**Quality Assessment**: ⭐⭐⭐⭐ Good
- Combined feature classification
- Histogram/bar disambiguation
- Score-based decision making
- Fallback classification

**Reuse Strategy**:
- **Direct reuse** with minor refactoring
- Extract scoring logic to separate methods
- Update imports to new config
- Simplify interface

**Refactoring Plan**:
```python
# src/classification/heuristic.py
class HeuristicClassifier:
    def __init__(self, config: HeuristicConfig):
        self.config = config
    
    def classify(self, features: Dict, detection: Dict) -> Dict:
        # Simplified interface
        chart_type = self._classify_combined(features, detection)
        return {
            "chart_type": chart_type,
            "confidence": self._compute_confidence(features, detection),
        }
```

**Migration Effort**: Low (2-3 hours)

---

### 12. ML Classifier

**Existing**: `src/classifier/ml_classifier.py` (232 lines)
**New Location**: `src/classification/ml.py`

**Quality Assessment**: ⭐⭐⭐⭐ Good
- Multiple models (RF, SVM, GB)
- Cross-validation
- Feature scaling
- Label encoding
- Model persistence with joblib
- Feature importance extraction

**Reuse Strategy**:
- **Merge with advanced_ml.py** and refactor
- Keep GridSearchCV from advanced_ml.py
- Simplify interface
- Update to new config system

**Merge Plan**:
```python
# From advanced_ml.py, add:
- GridSearchCV hyperparameter tuning
- XGBoost support
- OOF predictions
- Better feature importance handling

# Keep from ml_classifier.py:
- Clean interface
- Model persistence
- Cross-validation
```

**Migration Effort**: Medium (4-5 hours)

---

### 13. CNN Classifier

**Existing**: `src/research/cnn_classifier.py` (320 lines)
**New Location**: `src/classification/cnn.py`

**Quality Assessment**: ⭐⭐⭐⭐ Good
- Transfer learning (EfficientNetB0, ResNet50)
- Early stopping
- Data augmentation
- Model checkpointing
- GPU/CPU detection

**Reuse Strategy**:
- **Direct reuse** with minor changes
- Move from research to classification module
- Update imports to new config
- Simplify interface

**Migration Effort**: Low (1-2 hours)

---

### 14. Hybrid Classifier

**Existing**: `src/research/hybrid_classifier.py` (152 lines)
**New Location**: `src/classification/hybrid.py`

**Quality Assessment**: ⭐⭐⭐⭐ Good
- Confidence-weighted voting
- Histogram/bar disambiguation
- Weight optimization
- Clean voting logic

**Reuse Strategy**:
- **Direct reuse** with minor changes
- Move from research to classification module
- Update to use new classifier interfaces
- Simplify weight optimization

**Migration Effort**: Low (2-3 hours)

---

### 15. Value Extractor

**Existing**: `src/extraction/value_extractor.py` (290 lines)
**New Location**: `src/extraction/values.py`

**Quality Assessment**: ⭐⭐⭐⭐ Good
- Chart-specific extraction (bar, line, scatter, pie)
- OpenCV-based algorithms
- Normalized value output
- Label mapping

**Reuse Strategy**:
- **Direct reuse** with no changes
- Already clean and modular
- Update imports to new config

**Migration Effort**: None (0 hours)

---

### 16. Research Evaluator

**Existing**: `src/evaluation/research_evaluator.py` (136 lines)
**New Location**: `src/evaluation/metrics.py`

**Quality Assessment**: ⭐⭐⭐⭐⭐ Excellent
- Classification metrics (accuracy, precision, recall, F1)
- OCR metrics (confidence, token recall, coverage)
- Value extraction metrics
- Per-chart-type accuracy
- Per-dataset-source accuracy
- Clean, modular methods

**Reuse Strategy**:
- **Direct reuse** with no changes
- Already perfect for new architecture
- Rename to metrics.py for clarity

**Migration Effort**: None (0 hours)

---

### 17. Research Plot Generator

**Existing**: `src/visualization/research_plots.py` (157 lines)
**New Location**: `src/visualization/plots.py`

**Quality Assessment**: ⭐⭐⭐⭐⭐ Excellent
- Confusion matrix heatmap
- OCR confidence distribution
- Runtime breakdown
- Chart type distribution
- Success rate comparison
- Feature importance
- Clean seaborn-based implementation

**Reuse Strategy**:
- **Direct reuse** with no changes
- Already matches new architecture
- Delete duplicate plots.py

**Migration Effort**: None (0 hours)

---

### 18. Runtime Profiler

**Existing**: `src/profiling/runtime_profiler.py` (61 lines)
**New Location**: `src/utils/profiling.py`

**Quality Assessment**: ⭐⭐⭐⭐⭐ Excellent
- Context manager for tracking
- Thread-safe
- Mean/total/count statistics
- JSON serialization

**Reuse Strategy**:
- **Direct reuse** with no changes
- Move to utils module
- Already perfect

**Migration Effort**: None (0 hours)

---

### 19. YOLO Detector

**Existing**: `src/research/yolo_chart_detector.py` (354 lines)
**New Location**: `src/detection/yolo.py`

**Quality Assessment**: ⭐⭐⭐⭐ Good
- Pseudo-label generation
- YOLOv8 training
- Inference with fallback
- Thread-safe wrapper
- 8 class detection

**Reuse Strategy**:
- **Merge with yolo_detector.py**
- Keep pseudo-label generation (useful for training)
- Keep training logic (optional feature)
- Simplify inference interface
- Move to detection module

**Merge Plan**:
```python
# Keep from yolo_detector.py:
- Thread-safe wrapper
- Fallback to segmentation

# Keep from yolo_chart_detector.py:
- Pseudo-label generation (for training)
- Training logic (optional)
- 8-class detection
```

**Migration Effort**: Medium (4-5 hours)

---

### 20. Dataset Verifier

**Existing**: `src/research/dataset_verifier.py` (196 lines)
**New Location**: `src/data/dataset_verifier.py`

**Quality Assessment**: ⭐⭐⭐⭐ Good
- Dataset verification
- Corrupt image detection
- Class balancing
- Synthetic generation for balance
- CSV/JSON report generation

**Reuse Strategy**:
- **Direct reuse** with minor changes
- Move to data module
- Update imports to new config
- Keep as optional verification tool

**Migration Effort**: Low (1-2 hours)

---

## Components to Delete

### Dead Code (Delete)

1. **debug_pipeline.py** - Replace with pytest unit tests
2. **demo_experiment.py** - Move to examples/
3. **research_pipeline_final.py** - Superseded by modular pipeline
4. **research_pipeline_v3.py** - Superseded by modular pipeline
5. **download_test_images.py** - Functionality in dataset manager
6. **src/evaluation/evaluator.py** - Duplicate of research_evaluator.py
7. **src/visualization/plots.py** - Duplicate of research_plots.py
8. **src/visualization/debug_visualizer.py** - Move to tools/ or delete
9. **ocr_env/** - Virtual environment should not be in repo

### Merge Operations

1. **advanced_ml.py** → **ml_classifier.py** (merge GridSearchCV)
2. **chart_feature_extractor.py** → **geometric_features.py** (merge features)
3. **yolo_chart_detector.py** → **yolo_detector.py** (merge training)

---

## Reuse Summary

| Component | Quality | Reuse Strategy | Effort |
|-----------|---------|----------------|--------|
| OCR Engine | ⭐⭐⭐⭐⭐ | Direct reuse | Low |
| OCR Ensemble | ⭐⭐⭐⭐⭐ | Direct reuse | None |
| Segmenter | ⭐⭐⭐⭐⭐ | Direct reuse | Low |
| Geometric Features | ⭐⭐⭐⭐⭐ | Merge + reuse | Low |
| Dataset Manager | ⭐⭐⭐⭐⭐ | Refactor + reuse | Medium |
| Dataset Versioning | ⭐⭐⭐⭐⭐ | Direct reuse | None |
| Config Loader | ⭐⭐⭐⭐ | Expand + reuse | Low |
| Reproducibility | ⭐⭐⭐⭐⭐ | Direct reuse | None |
| OCR Cache | ⭐⭐⭐⭐⭐ | Generalize + reuse | Low |
| Image Preprocessor | ⭐⭐⭐⭐⭐ | Direct reuse | None |
| Heuristic Classifier | ⭐⭐⭐⭐ | Refactor + reuse | Low |
| ML Classifier | ⭐⭐⭐⭐ | Merge + reuse | Medium |
| CNN Classifier | ⭐⭐⭐⭐ | Direct reuse | Low |
| Hybrid Classifier | ⭐⭐⭐⭐ | Direct reuse | Low |
| Value Extractor | ⭐⭐⭐⭐ | Direct reuse | None |
| Research Evaluator | ⭐⭐⭐⭐⭐ | Direct reuse | None |
| Research Plots | ⭐⭐⭐⭐⭐ | Direct reuse | None |
| Runtime Profiler | ⭐⭐⭐⭐⭐ | Direct reuse | None |
| YOLO Detector | ⭐⭐⭐⭐ | Merge + reuse | Medium |
| Dataset Verifier | ⭐⭐⭐⭐ | Direct reuse | Low |

**Total Estimated Migration Effort**: 30-40 hours

---

## High-Value Components Summary

### Tier 1: Perfect Reuse (No Changes)
1. OCR Ensemble
2. Dataset Versioning
3. Reproducibility Utilities
4. Image Preprocessor
5. Value Extractor
6. Research Evaluator
7. Research Plot Generator
8. Runtime Profiler

### Tier 2: Minor Changes (Import Updates)
1. OCR Engine
2. Configuration Loader
3. OCR Cache
4. Heuristic Classifier
5. CNN Classifier
6. Hybrid Classifier
7. Dataset Verifier

### Tier 3: Moderate Refactoring (Structure Changes)
1. Chart Element Segmenter
2. Geometric Feature Extractor (merge)
3. Dataset Manager (split)
4. ML Classifier (merge)
5. YOLO Detector (merge)

---

## Code Quality Metrics

| Metric | Current | Target |
|--------|---------|-------|
| Lines of Code | ~8,000 | ~5,000 |
| Number of Files | 49 | 35 |
| Cyclomatic Complexity | High | Low |
| Test Coverage | 0% | >80% |
| Documentation | Partial | Complete |
| Type Hints | Partial | Complete |

---

## Next Steps

1. **Phase 4**: Research methodology redesign
2. **Phase 5**: Detailed migration plan
3. **Phase 6**: Implementation milestones
