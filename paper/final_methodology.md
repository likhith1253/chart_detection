# Final Methodology

## Phase 4: Research Methodology Design

This document describes the complete methodology for achieving the research objectives defined in the final research question.

---

## Methodology Overview

The research employs a multi-stage pipeline approach with systematic ablation studies to answer the primary and secondary research questions.

```
┌─────────────────────────────────────────────────────────────┐
│                     Input: Chart Image                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Stage 1: Preprocessing                     │
│  - Grayscale conversion                                       │
│  - CLAHE (Contrast Limited Adaptive Histogram Equalization)   │
│  - Adaptive thresholding                                      │
│  - Morphological cleanup                                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Stage 2: OCR Engine                         │
│  - PaddleOCR (primary)                                        │
│  - EasyOCR (secondary)                                        │
│  - Ensemble fusion (confidence-weighted)                      │
│  - Caching for reproducibility                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Stage 3: Detection                         │
│  - Traditional segmentation (CPU-only)                        │
│  - Element counting (bars, lines, points, pie slices, bins)   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 Stage 4: Feature Extraction                   │
│  - Geometric features (35+ features)                          │
│  - Structural features (axis detection, layout)               │
│  - OCR-based features (text statistics)                       │
│  - Histogram-specific features (bin detection, distribution)  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Stage 5: Classification Ensemble                  │
│  - ML classifier (Random Forest, SVM, Gradient Boosting)       │
│  - CNN classifier (EfficientNet-B0, optional GPU)             │
│  - Weighted ensemble fusion                                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Stage 6: Evaluation                          │
│  - Classification metrics (accuracy, F1, confusion matrix)     │
│  - Efficiency metrics (runtime, memory)                        │
│  - Statistical analysis (bootstrap, significance tests)        │
└─────────────────────────────────────────────────────────────┘
```

---

## Detailed Methodology

### Stage 1: Preprocessing

**Objective**: Prepare images for downstream processing while preserving chart structure.

**Operations**:
1. **Grayscale Conversion**: Reduce dimensionality while preserving structural information
2. **CLAHE**: Enhance local contrast for better feature extraction
3. **Adaptive Thresholding**: Binarize for text and element detection
4. **Morphological Cleanup**: Remove noise and small artifacts
5. **Deskewing (optional)**: Correct rotation for better OCR

**Implementation**:
```python
class ImagePreprocessor:
    def process(self, image_path: Path) -> Dict:
        image = cv2.imread(str(image_path))
        
        # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return {
            "original": image,
            "gray": gray,
            "enhanced": enhanced,
            "binary": cleaned
        }
```

**Ablation Variants**:
- No preprocessing
- Grayscale only
- Full preprocessing

---

### Stage 2: OCR Engine

**Objective**: Extract text from chart images for classification and QA support.

**Engines**:
1. **PaddleOCR**: Higher accuracy, heavier computation
2. **EasyOCR**: Faster, lighter weight
3. **Ensemble**: Confidence-weighted fusion of both engines

**Ensemble Strategy**:
```python
class OCREnsemble:
    def fuse_results(self, paddle_results, easyocr_results):
        fused = []
        for p, e in zip(paddle_results, easyocr_results):
            # Weight by confidence
            if p["confidence"] > e["confidence"]:
                fused.append(p)
            else:
                fused.append(e)
        return fused
```

**Caching Strategy**:
- Hash-based key from image content
- Disk-backed cache for reproducibility
- Configurable cache invalidation

**Ablation Variants**:
- PaddleOCR only
- EasyOCR only
- Ensemble fusion
- No OCR (baseline)

---

### Stage 3: Detection

**Objective**: Detect and count chart elements for classification and feature extraction.

```python
class ElementDetector:
    def __init__(self, config=None):
        self.traditional = TraditionalDetector()
    
    def detect(self, image):
        return self.traditional_detect(image)
```

**Output**:
```python
{
    "bars": 5,
    "lines": 2,
    "points": 12,
    "pie_slices": 4,
    "axes": 2,
    "bboxes": [...]  # Bounding boxes for each element
}
```

**Ablation Variants**:
- Traditional only
- No detection (heuristic only)

---

### Stage 4: Feature Extraction

**Objective**: Extract carefully chosen features for classification (30 features total).

**Feature Groups**:

#### 4.1 Geometric Features (15 features)
- Image dimensions (width, height, aspect ratio)
- Contour statistics (area, perimeter, compactness)
- Edge density
- Line segment count and orientation
- Circle detection count
- Blob count and size distribution

#### 4.2 Structural Features (8 features)
- Axis detection (presence, count, orientation)
- Grid detection (presence, spacing)
- Legend detection (presence, position)
- Title detection (presence, position)

#### 4.3 OCR-Based Features (4 features)
- Text count
- Text density
- Numeric text ratio
- Label statistics

#### 4.4 Histogram-Specific Features (3 features)
- Bin count and spacing
- Gap analysis between bins
- Axis gap analysis (histograms typically have no gap between bars and x-axis)

**Implementation**:
```python
class FeatureExtractor:
    def extract(self, image, detection_results, ocr_results):
        features = {}
        
        # Geometric
        features.update(self.extract_geometric(image))
        
        # Structural
        features.update(self.extract_structural(image))
        
        # OCR-based
        features.update(self.extract_ocr_features(ocr_results))
        
        # Histogram-specific
        features.update(self.extract_histogram_features(image, detection_results))
        
        return features
```

**Ablation Variants**:
- Geometric only
- Structural only
- OCR-based only
- Histogram-specific only
- All features

---

### Stage 5: Classification Ensemble

**Objective**: Classify chart type using multiple methods and ensemble fusion.

**Classifiers**:

#### 5.1 Heuristic Classifier
- Rule-based classification using detection results
- Example: If bars > 0 and lines == 0 → bar chart
- Fast but limited accuracy

#### 5.2 ML Classifiers
- **Random Forest**: Primary ML classifier (fast, interpretable)
- **SVM**: For comparison
- **Gradient Boosting**: For comparison
- **XGBoost**: For comparison

**Training**:
- 5-fold stratified cross-validation
- Hyperparameter tuning via grid search
- Feature importance analysis

#### 5.3 CNN Classifier (Kaggle GPU)
- **EfficientNet-B0**: Lightweight CNN
- Transfer learning from ImageNet
- Fine-tuning on chart dataset

#### 5.4 Ensemble Fusion
- Weighted voting with learned weights
- Confidence-based fusion

**Implementation**:
```python
class ClassificationEnsemble:
    def __init__(self, config):
        self.heuristic = HeuristicClassifier()
        self.ml = MLClassifier(config.ml)
        self.cnn = CNNClassifier(config.cnn) if config.cnn.enabled else None
        self.weights = config.classification.hybrid.weights
    
    def classify(self, features, detection_results):
        predictions = {}
        
        # Get predictions from each classifier
        predictions["heuristic"] = self.heuristic.classify(detection_results)
        predictions["ml"] = self.ml.classify(features)
        if self.cnn:
            predictions["cnn"] = self.cnn.classify(features)
        
        # Ensemble fusion
        return self.ensemble(predictions)
    
    def ensemble(self, predictions):
        # Weighted voting
        votes = defaultdict(float)
        for method, pred in predictions.items():
            weight = self.weights.get(method, 0)
            for chart_type, confidence in pred.items():
                votes[chart_type] += confidence * weight
        
        return dict(votes)
```

**Ablation Variants**:
- Heuristic only
- ML only (each model separately)
- CNN only
- Ensemble (different weight configurations)

---

### Stage 6: Evaluation

**Objective**: Comprehensive evaluation of classification performance.

#### 6.1 Classification Metrics
- **Overall Accuracy**: Correct predictions / total predictions
- **Per-Class Accuracy**: Accuracy for each chart type
- **Histogram vs. Bar Disambiguation Accuracy**: Specific metric for histogram/bar distinction
- **Macro F1-Score**: Average F1 across classes (unweighted)
- **Weighted F1-Score**: F1 weighted by class support
- **Confusion Matrix**: Visualize classification errors
- **Precision/Recall**: Per-class precision and recall

#### 6.2 Efficiency Metrics
- **Per-Image Runtime**: Total time per image (broken down by stage)
- **Memory Footprint**: Peak memory usage
- **Model Loading Time**: Time to load models
- **Cache Hit Rate**: OCR cache effectiveness

#### 6.3 Statistical Analysis
- **Bootstrap Confidence Intervals**: 1,000 bootstrap samples for accuracy estimates
- **McNemar's Test**: Pairwise significance testing between methods
- **Paired t-Tests**: Compare methods across folds
- **Bonferroni Correction**: Adjust for multiple comparisons

#### 6.4 Error Analysis
- **Per-Class Error Rates**: Which classes are most confused?
- **Confusion Patterns**: Systematic confusion analysis (especially histogram vs. bar)
- **Feature Importance**: Which features contribute most?
- **Qualitative Failure Cases**: Visual analysis of errors
- **OCR Failure Impact**: How often does OCR cause errors?

**Implementation**:
```python
class Evaluator:
    def evaluate_classification(self, predictions, ground_truth):
        metrics = {}
        metrics["accuracy"] = accuracy_score(ground_truth, predictions)
        metrics["macro_f1"] = f1_score(ground_truth, predictions, average="macro")
        metrics["weighted_f1"] = f1_score(ground_truth, predictions, average="weighted")
        metrics["confusion_matrix"] = confusion_matrix(ground_truth, predictions)
        metrics["per_class_report"] = classification_report(ground_truth, predictions)
        
        # Histogram vs. Bar disambiguation
        hist_bar_mask = (ground_truth == "histogram") | (ground_truth == "bar")
        if np.sum(hist_bar_mask) > 0:
            metrics["hist_bar_accuracy"] = accuracy_score(
                ground_truth[hist_bar_mask], 
                predictions[hist_bar_mask]
            )
        
        return metrics
    
    def bootstrap_ci(self, metric_func, data, n_samples=1000):
        bootstrap_samples = []
        for _ in range(n_samples):
            sample = resample(data)
            bootstrap_samples.append(metric_func(sample))
        return np.percentile(bootstrap_samples, [2.5, 97.5])
```

---

## Experimental Design

### Dataset

**Primary Dataset**: Synthetic Charts
- **Size**: 2,500 images (500 per chart type)
- **Chart Types**: Bar, Line, Pie, Scatter, Histogram
- **Variation**: Fonts, colors, sizes, orientations
- **Ground Truth**: Perfect labels

**Dataset Splits**:
- Train: 70% (1,750 images)
- Validation: 15% (375 images)
- Test: 15% (375 images)
- Stratified by chart type

### Baseline Methods

**Classification Baselines**:
1. **Random Baseline**: Random classification
2. **Heuristic-Only**: Rule-based without ML
3. **Feature-Only ML**: ML on features without OCR/detection
4. **CNN-Only**: Deep learning without handcrafted features
5. **Full Pipeline**: Complete proposed system

### Ablation Studies

**Ablation 1: Feature Groups**
- Geometric only
- Structural only
- OCR-based only
- Histogram-specific only
- All features

**Ablation 2: Classification Methods**
- Heuristic only
- ML only (RF, SVM, GB, XGB)
- CNN only
- Ensemble (different weights)

**Ablation 3: OCR Impact**
- PaddleOCR only
- EasyOCR only
- Ensemble
- No OCR

**Ablation 4: Detection Impact**
- Traditional only
- No detection

**Ablation 5: Preprocessing**
- No preprocessing
- Minimal (grayscale only)
- Full preprocessing

### Evaluation Protocol

**Cross-Validation**:
- 5-fold stratified cross-validation
- Same folds for all methods (fair comparison)
- Report mean and standard deviation

**Statistical Significance**:
- Bootstrap confidence intervals (1,000 samples)
- McNemar's test for pairwise comparison
- Bonferroni correction for multiple comparisons

**Reproducibility**:
- Fixed random seed for all experiments
- Dataset fingerprint recorded
- Library versions recorded
- Git commit hash recorded
- Full configuration saved

---

## Implementation Timeline

### Week 1-2: Baseline Implementation
1. Implement preprocessing pipeline
2. Implement OCR engine (both engines + ensemble)
3. Implement detection (traditional only)
4. Implement feature extraction (30 features)
5. Implement heuristic classifier
6. Implement ML classifier (Random Forest only)
7. Generate synthetic dataset (2,500 images)
8. Run initial classification experiments

### Week 3-4: Full Pipeline
1. Implement additional ML classifiers (SVM, GB, XGB)
2. Implement CNN classifier (EfficientNet-B0) - use Kaggle GPU for training
3. Implement ensemble classifier
4. Implement evaluation pipeline
5. Run full classification experiments

### Week 5-6: Ablation Studies
1. Feature group ablations (including histogram-specific)
2. Classification method ablations
3. OCR impact ablations
4. Detection impact ablations
5. Preprocessing ablations
6. Ensemble weight optimization
7. Efficiency analysis

### Week 7-8: Analysis and Reporting
1. Error analysis (classification)
2. Statistical analysis (bootstrap, significance tests)
3. Feature importance analysis
4. Histogram-specific analysis (disambiguation focus)
5. Figure generation (confusion matrices, performance plots)
6. Results interpretation
7. Paper writing

---

## Success Criteria

### Classification
- Overall accuracy >90% (minimum >85%)
- Per-class accuracy >85% for all types (minimum >80%)
- Histogram vs. Bar disambiguation >95% (minimum >90%)
- Macro F1 >0.88 (minimum >0.82)
- Statistical significance vs. baselines (p < 0.05)

### Efficiency
- Per-image runtime <1s on CPU (minimum <2s)
- Memory footprint <2GB peak (minimum <4GB)
- Model loading time <5s (minimum <10s)

### Analysis
- Clear ablation study insights
- Comprehensive error analysis
- Statistical significance demonstrated
- Histogram disambiguation analysis completed

---

## Risk Mitigation

### Technical Risks
1. **CNN training too slow on CPU**: Use Kaggle GPU for CNN training only
2. **OCR quality too low**: Implement OCR-free baseline, analyze impact
3. **Feature extraction too slow**: Profile and optimize, remove slow features
4. **Histogram disambiguation inconclusive**: Emphasize general framework if no significant differences

### Research Risks
1. **Results not competitive with SOTA**: Focus on CPU-only comparison, emphasize efficiency
2. **Novelty insufficient**: Emphasize CPU-first + histogram disambiguation
3. **Ablation studies inconclusive**: Design clear hypotheses, use statistical testing
4. **Histogram analysis inconclusive**: Emphasize general framework if no significant differences

---

## Conclusion

This methodology provides a comprehensive framework for answering the research questions:

- **Systematic**: Clear stages with defined inputs/outputs
- **Reproducible**: Full snapshot and seeding support
- **Focused**: Classification only with histogram disambiguation
- **Efficient**: CPU-first design with Kaggle GPU for CNN training
- **Rigorous**: Statistical analysis and ablation studies

The methodology is designed to be achievable within the 6-8 week timeline while producing publication-quality results.
