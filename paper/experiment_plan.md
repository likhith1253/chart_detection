# Complete Experiment Plan

## Phase 5: Experiment Design

This document provides a detailed experiment plan including datasets, baselines, evaluation metrics, ablation studies, and statistical analysis protocols.

---

## Dataset Design

### Primary Dataset: Synthetic Charts

#### Dataset Specifications
- **Total Images**: 2,500
- **Chart Types**: 5 types (1,000 each)
  - Bar charts (vertical, horizontal, grouped, stacked)
  - Line charts (single, multiple, with/without markers)
  - Pie charts
  - Scatter plots
  - Histograms
- **Image Resolution**: 1024x768 pixels
- **Images per Type**: 500
- **Color Spaces**: RGB
- **Variation Dimensions**:
  - Fonts: 5 different font families
  - Colors: 10 different color palettes
  - Sizes: 3 different scales (0.8x, 1.0x, 1.2x)
  - Orientations: 2 (normal, rotated 90°)
  - Noise levels: 3 (none, low, high)

#### Ground Truth Annotation
Each image includes:
- Chart type label
- Element bounding boxes (bars, lines, points, pie slices, bins)
- Axis positions and labels
- Legend position and labels
- Title position and text

#### Dataset Splits
- **Train**: 1,750 images (70%)
  - 350 per chart type
- **Validation**: 375 images (15%)
  - 75 per chart type
- **Test**: 375 images (15%)
  - 75 per chart type
- **Stratification**: Balanced by chart type

#### Dataset Generation Pipeline

```python
class SyntheticChartGenerator:
    def __init__(self, config):
        self.config = config
        self.chart_types = ["bar", "line", "pie", "scatter", "histogram"]
        self.fonts = ["Arial", "Times New Roman", "Courier New", "Verdana", "Georgia"]
        self.colors = self.generate_color_palettes(10)
    
    def generate_dataset(self, n_per_type=500):
        dataset = []
        for chart_type in self.chart_types:
            for i in range(n_per_type):
                # Random variation parameters
                font = random.choice(self.fonts)
                colors = random.choice(self.colors)
                scale = random.choice([0.8, 1.0, 1.2])
                rotation = random.choice([0, 90])
                noise = random.choice([0, 0.1, 0.2])
                
                # Generate chart
                chart = self.generate_chart(
                    chart_type, font, colors, scale, rotation, noise
                )
                
                dataset.append({
                    "image": chart["image"],
                    "chart_type": chart_type,
                    "ground_truth": chart["data"],
                    "metadata": {
                        "font": font,
                        "colors": colors,
                        "scale": scale,
                        "rotation": rotation,
                        "noise": noise
                    }
                })
        
        return dataset
```

### Dataset Fingerprinting

For reproducibility, each dataset will have a fingerprint:

```python
def compute_dataset_fingerprint(dataset):
    fingerprint = {
        "n_images": len(dataset),
        "chart_type_distribution": Counter([d["chart_type"] for d in dataset]),
        "image_hashes": [hashlib.md5(d["image"].tobytes()).hexdigest() for d in dataset],
        "config_hash": hashlib.md5(json.dumps(dataset[0]["metadata"]).encode()).hexdigest(),
        "generation_timestamp": datetime.now().isoformat()
    }
    return fingerprint
```

---

## Baseline Methods

### Classification Baselines

#### Baseline 1: Random Classifier
- **Description**: Randomly selects chart type
- **Implementation**: `random.choice(chart_types)`
- **Purpose**: Lower bound on performance

#### Baseline 2: Heuristic-Only Classifier
- **Description**: Rule-based classification using detection results
- **Rules**:
  - If bars > 0 and lines == 0 → bar chart
  - If lines > 0 and bars == 0 → line chart
  - If pie_slices > 0 → pie chart
  - If points > 0 and lines == 0 → scatter plot
  - If bins > 0 → histogram
- **Purpose**: Non-ML baseline

#### Baseline 3: Feature-Only ML (Random Forest)
- **Description**: Random Forest on geometric features only
- **Features**: Geometric features (15)
- **No OCR or detection features**
- **Purpose**: ML baseline without OCR/detection

#### Baseline 4: CNN-Only (EfficientNet-B0)
- **Description**: Transfer learning from ImageNet
- **Architecture**: EfficientNet-B0
- **Input**: Raw image (224x224)
- **No handcrafted features**
- **Purpose**: Deep learning baseline

#### Baseline 5: Full Pipeline (Proposed)
- **Description**: Complete proposed system
- **Components**: All features + ensemble classification
- **Purpose**: Main proposed method

## Evaluation Metrics

### Classification Metrics

#### Primary Metrics
1. **Overall Accuracy**: `correct / total`
2. **Macro F1-Score**: Average F1 across classes (unweighted)
3. **Weighted F1-Score**: F1 weighted by class support

#### Secondary Metrics
1. **Per-Class Accuracy**: Accuracy for each chart type
2. **Per-Class Precision**: Precision for each chart type
3. **Per-Class Recall**: Recall for each chart type
4. **Confusion Matrix**: Visual representation of errors

#### Implementation
```python
def compute_classification_metrics(y_true, y_pred):
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro")
    metrics["weighted_f1"] = f1_score(y_true, y_pred, average="weighted")
    metrics["per_class_report"] = classification_report(y_true, y_pred, output_dict=True)
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    return metrics
```

### Efficiency Metrics

#### Runtime Metrics
1. **Per-Image Runtime**: Total time per image
2. **Per-Stage Runtime**: Time for each pipeline stage
3. **Model Loading Time**: Time to load all models

#### Memory Metrics
1. **Peak Memory Usage**: Maximum memory during execution
2. **Model Memory**: Memory used by loaded models
3. **Per-Image Memory**: Memory per image processing

#### Implementation
```python
class EfficiencyProfiler:
    def __init__(self):
        self.stage_times = defaultdict(list)
        self.peak_memory = 0
    
    def profile_stage(self, stage_name, func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            self.stage_times[stage_name].append(end_time - start_time)
            self.peak_memory = max(self.peak_memory, end_memory - start_memory)
            
            return result
        return wrapper
```

---

## Ablation Studies

### Ablation 1: Feature Groups

**Objective**: Determine contribution of each feature group

**Variants**:
1. Geometric & Structural only (~23 features)
2. OCR-based only (~4 features)
3. Histogram-specific only (~3 features)
4. All features (~30 features)

**Expected Outcome**:
- Geometric features: High contribution for classification
- OCR features: Moderate contribution for classification
- Histogram-specific features: High contribution for histogram vs. bar disambiguation

**Analysis**:
- Feature importance from Random Forest
- Ablation accuracy comparison
- Statistical significance testing

### Ablation 2: Classification Methods

**Objective**: Compare different classification approaches

**Variants**:
1. Heuristic only
2. ML - Random Forest
3. ML - SVM
4. ML - Gradient Boosting
5. ML - XGBoost
6. CNN - EfficientNet-B0
9. Ensemble (equal weights)
10. Ensemble (learned weights)

**Expected Outcome**:
- Heuristic: Fast but low accuracy (~60%)
- ML: Good accuracy (~85-90%), fast
- CNN: High accuracy (~90-95%), slow on CPU
- Ensemble: Best accuracy (~92-95%), moderate speed

**Analysis**:
- Accuracy comparison
- Runtime comparison
- Accuracy vs. runtime trade-off analysis

### Ablation 3: OCR Impact

**Objective**: Measure OCR quality impact on classification

**Variants**:
1. With OCR
2. No OCR (baseline)

**Expected Outcome**:
- Classification: 10-15% accuracy difference with/without OCR

**Expected Outcome**:
- YOLO: Best accuracy, requires GPU
- Traditional: Good accuracy, CPU-only
- No detection: Lower accuracy (~70%)

**Analysis**:
- Detection accuracy on elements
- Detection failure impact on classification
- CPU vs. GPU comparison

### Ablation 5: Preprocessing

**Objective**: Measure preprocessing impact on performance

**Variants**:
1. No preprocessing
2. Grayscale only
3. Full preprocessing (grayscale + CLAHE + threshold + morphology)

**Expected Outcome**:
- No preprocessing: Lower accuracy, faster
- Grayscale only: Moderate improvement
- Full preprocessing: Best accuracy, slower

**Analysis**:
- Accuracy vs. preprocessing complexity
- Runtime vs. preprocessing complexity
- Optimal preprocessing configuration

---

## Statistical Analysis

### Bootstrap Confidence Intervals

**Objective**: Compute confidence intervals for accuracy estimates

**Method**:
- 1,000 bootstrap samples
- Compute metric on each sample
- Report 2.5th and 97.5th percentiles

**Implementation**:
```python
def bootstrap_ci(metric_func, data, n_samples=1000, ci=95):
    bootstrap_samples = []
    for _ in range(n_samples):
        sample = resample(data)
        bootstrap_samples.append(metric_func(sample))
    
    lower = (100 - ci) / 2
    upper = 100 - lower
    return np.percentile(bootstrap_samples, [lower, upper])
```

**Application**:
- Overall accuracy CI
- Per-class accuracy CI
- QA accuracy CI
- Runtime CI

### McNemar's Test

**Objective**: Test statistical significance between two methods

**Method**:
- Contingency table of correct/incorrect for both methods
- McNemar's test statistic
- p-value calculation

**Implementation**:
```python
def mcnemar_test(y_true, pred1, pred2):
    # Build contingency table
    n00 = sum((p1 == y_true[i]) and (p2 == y_true[i]) for i, p1, p2 in zip(pred1, pred2))
    n01 = sum((p1 != y_true[i]) and (p2 == y_true[i]) for i, p1, p2 in zip(pred1, pred2))
    n10 = sum((p1 == y_true[i]) and (p2 != y_true[i]) for i, p1, p2 in zip(pred1, pred2))
    n11 = sum((p1 != y_true[i]) and (p2 != y_true[i]) for i, p1, p2 in zip(pred1, pred2))
    
    # McNemar's test
    statistic = (abs(n01 - n10) - 1)**2 / (n01 + n10)
    p_value = 1 - chi2.cdf(statistic, 1)
    
    return statistic, p_value
```

**Application**:
- Compare proposed method vs. each baseline
- Compare ablation variants
- Report p-values with Bonferroni correction

### Paired t-Tests

**Objective**: Compare methods across cross-validation folds

**Method**:
- Compute accuracy per fold for each method
- Paired t-test between methods
- p-value calculation

**Implementation**:
```python
def paired_t_test(acc1, acc2):
    statistic, p_value = ttest_rel(acc1, acc2)
    return statistic, p_value
```

**Application**:
- Compare ML classifiers across folds
- Compare ensemble configurations
- Report p-values with Bonferroni correction

### Bonferroni Correction

**Objective**: Adjust for multiple comparisons

**Method**:
- Divide alpha by number of comparisons
- Compare p-values to adjusted alpha

**Implementation**:
```python
def bonferroni_correction(p_values, alpha=0.05):
    n_comparisons = len(p_values)
    adjusted_alpha = alpha / n_comparisons
    significant = [p < adjusted_alpha for p in p_values]
    return adjusted_alpha, significant
```

---

## Experiment Execution Protocol

### Reproducibility Protocol

1. **Fixed Random Seed**: All random operations seeded with 42
2. **Dataset Fingerprint**: Record dataset hash and statistics
3. **Library Versions**: Record all library versions
4. **Git Commit**: Record git commit hash
5. **Configuration Snapshot**: Save full configuration
6. **Code Version**: Tie experiment to specific commit

### Cross-Validation Protocol

1. **5-Fold Stratified Cross-Validation**
2. **Same Folds for All Methods**: Ensure fair comparison
3. **Report Mean and Std**: Across folds
4. **Statistical Testing**: On fold-level results

### Experiment Order

#### Phase 1: Baseline Experiments (Week 1-2)
1. Random classifier
2. Heuristic classifier
3. Feature-only ML (Random Forest)
4. CNN-only (EfficientNet-B0)
#### Phase 2: Full Pipeline Experiments (Week 3-4)
1. Full pipeline (classification)
2. All ML classifiers comparison
3. Ensemble configurations

#### Phase 3: Ablation Experiments (Week 5-6)
1. Feature group ablations
2. Classification method ablations
3. OCR impact ablations
5. Preprocessing ablations

#### Phase 4: Analysis Experiments (Week 7-8)
1. Error analysis
2. Statistical analysis
3. Efficiency analysis
4. Histogram-specific analysis

---

## Result Recording

### Experiment Metadata

Each experiment will record:
```json
{
  "experiment_id": "exp_001",
  "timestamp": "2025-01-15T10:00:00Z",
  "git_commit": "abc123",
  "config": {...},
  "dataset_fingerprint": {...},
  "library_versions": {...},
  "random_seed": 42
}
```

### Results Structure

```json
{
  "classification": {
    "accuracy": 0.92,
    "macro_f1": 0.88,
    "weighted_f1": 0.90,
    "per_class_accuracy": {...},
    "confusion_matrix": [[...]],
    "bootstrap_ci": [0.90, 0.94]
  },
  "efficiency": {
    "per_image_runtime_ms": 850,
    "stage_times_ms": {
      "preprocessing": 50,
      "ocr": 300,
      "detection": 200,
      "feature_extraction": 100,
      "classification": 50
    },
    "peak_memory_mb": 1800
  },
  "statistical_tests": {
    "vs_baseline": {
      "mcnemar_statistic": 45.2,
      "p_value": 1.2e-10,
      "significant": true
    }
  }
}
```

### File Structure

```
results/
├── experiments/
│   ├── exp_001_baseline_random/
│   │   ├── metadata.json
│   │   ├── results.json
│   │   ├── confusion_matrix.png
│   │   └── logs/
│   ├── exp_002_baseline_heuristic/
│   │   └── ...
│   └── ...
├── ablation/
│   ├── ablation_001_features/
│   │   └── ...
│   └── ...
├── plots/
│   ├── accuracy_comparison.png
│   ├── runtime_breakdown.png
│   └── feature_importance.png
└── reports/
    ├── classification_report.md
    ├── qa_report.md
    ├── ablation_report.md
    └── final_report.md
```

---

## Success Criteria

### Classification Success

**Minimum Viable**:
- Overall accuracy >85%
- Per-class accuracy >80%
- Macro F1 >0.82
- Statistical significance vs. baselines (p < 0.05)

**Target**:
- Overall accuracy >90%
- Per-class accuracy >85%
- Macro F1 >0.88
- Statistical significance vs. all baselines (p < 0.01)

**Stretch**:
- Overall accuracy >92%
- Per-class accuracy >88%
- Macro F1 >0.90
- Competitive with GPU methods (within 5%)

### Efficiency Success

**Minimum Viable**:
- Per-image runtime <2s (CPU)
- Memory footprint <4GB
- Model loading <10s

**Target**:
- Per-image runtime <1s (CPU)
- Memory footprint <2GB
- Model loading <5s

**Stretch**:
- Per-image runtime <500ms (CPU)
- Memory footprint <1GB
- Model loading <3s

### Analysis Success

**Minimum Viable**:
- Clear ablation insights
- Basic error analysis
- Statistical significance demonstrated

**Target**:
- Comprehensive ablation analysis
- Detailed error analysis with patterns
- Statistical significance with multiple tests
- Efficiency vs. accuracy trade-off analysis

**Stretch**:
- Histogram-specific insights
- Cross-dataset generalization (if possible)
- Publication-ready figures and tables

---

## Risk Mitigation

### Dataset Generation Risks

**Risk**: Synthetic dataset too simple, doesn't reflect real-world complexity
**Mitigation**: 
- Add noise and variation
- Validate with real-world samples if possible
- Report dataset limitations

### Experiment Risks

**Risk**: Experiments take longer than expected
**Mitigation**:
- Parallel processing where possible
- Prioritize key experiments
- Use smaller dataset for initial testing

**Risk**: Statistical tests not significant
**Mitigation**:
- Increase dataset size if possible
- Use more sensitive tests
- Report non-significant results honestly

### Technical Risks

**Risk**: CNN training too slow on CPU
**Mitigation**:
- Use pre-trained features only
- Limit CNN to GPU comparison
- Accept longer runtime for CNN experiments

**Risk**: OCR failures too frequent
**Mitigation**:
- Implement OCR-free baseline
- Analyze failure modes
- Report OCR accuracy

---

## Timeline Summary

### Week 1-2: Baseline Experiments
- Dataset generation
- Baseline implementations
- Initial experiments

### Week 3-4: Full Pipeline Experiments
- Full pipeline implementation
- Classification experiments

### Week 5-6: Ablation Experiments
- Feature ablations
- Method ablations
- OCR/detection ablations

### Week 7-8: Analysis Experiments
- Error analysis
- Statistical analysis
- Efficiency analysis
- Histogram analysis

### Week 9-10: Reporting
- Figure generation
- Report writing
- Revision
- Submission

---

## Conclusion

This experiment plan provides:

- **Clear dataset design**: 5K synthetic images with 10K QA pairs
- **Comprehensive baselines**: 5 classification baselines, 4 QA baselines
- **Detailed metrics**: Classification, QA, efficiency metrics
- **Systematic ablations**: 6 ablation studies covering all components
- **Rigorous analysis**: Statistical testing with multiple methods
- **Reproducibility**: Full snapshot and fingerprinting protocol

The plan is designed to produce publication-quality results within the 10-12 week timeline.
