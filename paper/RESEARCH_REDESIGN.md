# Research Redesign

## Phase 4: Research Question and Methodology Optimization

This document analyzes the current research approach and proposes an optimized methodology for publication-quality research.

---

## Current Research Analysis

### Existing Research Question

**Current**: "Build a chart understanding system that classifies chart types and extracts values using OCR, computer vision, and machine learning."

**Issues**:
- Too broad and unfocused
- Lacks specific hypothesis or contribution
- Combines multiple research problems (classification + extraction)
- No clear novelty statement
- Difficult to position in literature

### Current Contributions

**Existing**:
1. OCR ensemble with PaddleOCR and EasyOCR
2. YOLO-based chart element detection
3. Hybrid classification (heuristic + ML + CNN)
4. Value extraction for multiple chart types

**Issues**:
- Contributions are incremental rather than novel
- No clear theoretical contribution
- System engineering rather than research innovation
- Limited comparison with state-of-the-art
- No analysis of failure modes

### Current Methodology

**Existing**:
- Monolithic pipeline with multiple stages
- Ablation studies on OCR engines
- ML model comparison (RF, SVM, GB, XGB)
- CNN transfer learning
- Hybrid ensemble

**Issues**:
- Experiments not hypothesis-driven
- Ablation studies too narrow (OCR only)
- No systematic hyperparameter tuning
- Limited statistical analysis
- No cross-dataset evaluation
- No error analysis

---

## Proposed Research Redesign

### Optimized Research Question

**Primary Research Question**:
> "How can we achieve robust chart type classification under resource-constrained environments (CPU-only, minimal dependencies) while maintaining competitive accuracy with GPU-heavy deep learning approaches?"

**Secondary Questions**:
1. What is the relative contribution of heuristic features vs. learned features in chart classification?
2. Can ensemble methods bridge the gap between CPU-only and GPU-accelerated approaches?
3. How does OCR quality impact downstream classification performance?
4. What are the failure modes of lightweight chart classification systems?

### Research Contributions

**Contribution 1: Lightweight Architecture**
- Design of a CPU-first chart classification pipeline
- Systematic analysis of feature extraction methods
- Comparison of single-model vs. ensemble approaches
- Demonstration that competitive accuracy is achievable without GPU

**Contribution 2: Hybrid Feature Engineering**
- Novel combination of geometric, structural, and texture features
- Systematic ablation of feature groups
- Analysis of feature importance across chart types
- Generalizable feature extraction framework

**Contribution 3: Robust OCR Integration**
- OCR ensemble with confidence-weighted fusion
- Analysis of OCR quality impact on classification
- Fallback strategies for OCR failure
- Cache-aware optimization for reproducibility

**Contribution 4: Comprehensive Evaluation**
- Multi-dataset evaluation (synthetic + real)
- Per-chart-type error analysis
- Cross-dataset generalization study
- Statistical significance testing

### Novelty Statement

**Positioning**:
- Unlike GPU-heavy approaches (ChartQA, DVQA), we focus on CPU-only deployment
- Unlike single-method approaches, we systematically compare multiple paradigms
- Unlike black-box deep learning, we provide interpretable feature-based methods
- Unlike engineering-focused systems, we provide research-driven analysis

---

## Optimized Architecture

### Lightweight Design Principles

1. **CPU-First**: All components optimized for CPU, GPU optional
2. **Minimal Dependencies**: Core pipeline uses only numpy, opencv, scikit-learn
3. **Modular Components**: Each component independently testable
4. **Interpretable Features**: Feature-based methods with clear semantics
5. **Reproducible**: Full snapshot and deterministic execution

### Component Optimization

**Preprocessing**:
- Keep: Grayscale, CLAHE, adaptive threshold
- Remove: Heavy denoising (too slow)
- Add: Minimal preprocessing for OCR only

**Feature Extraction**:
- Keep: Geometric features (35+ features)
- Add: Structural features (axis detection, layout analysis)
- Remove: Redundant texture features (high compute, low discriminative power)

**Classification**:
- Primary: Random Forest (fast, interpretable, good accuracy)
- Secondary: SVM (for comparison)
- Tertiary: CNN (optional, for GPU comparison)
- Ensemble: Weighted voting with learned weights

**OCR**:
- Primary: EasyOCR (faster, lighter)
- Secondary: PaddleOCR (better accuracy, heavier)
- Ensemble: Confidence-weighted fusion
- Fallback: No OCR (classification without text)

---

## Experimental Design

### Baseline Methods

**Baselines to Implement**:

1. **Random Baseline**: Random classification (lower bound)
2. **Heuristic-Only**: Rule-based classification without ML
3. **Feature-Only ML**: ML on geometric features only
4. **CNN-Only**: Deep learning without handcrafted features
5. **Full Pipeline**: Complete proposed system

**State-of-the-Art Comparison** (if feasible):
- ChartQA classifier (if code available)
- DVQA classifier (if code available)
- Published chart classification papers

### Datasets

**Primary Dataset**: Synthetic charts (generated in-house)
- 5,000 images (1,000 per chart type)
- Balanced distribution
- Controlled variation (fonts, colors, sizes)
- Perfect ground truth

**Secondary Dataset**: Real charts (if available)
- ChartQA subset (if accessible)
- DVQA subset (if accessible)
- Public chart datasets

**Dataset Splits**:
- Train: 70%
- Validation: 15%
- Test: 15%
- Stratified by chart type

### Evaluation Metrics

**Primary Metrics**:
- Overall accuracy
- Per-class accuracy
- Macro F1-score
- Weighted F1-score

**Secondary Metrics**:
- Precision, recall per class
- Confusion matrix analysis
- Runtime per image
- Memory footprint

**Statistical Analysis**:
- 95% confidence intervals (bootstrap)
- McNemar's test for significance
- Paired t-tests across methods

### Ablation Studies

**Ablation 1: Feature Groups**
- Geometric features only
- Structural features only
- Texture features only
- All features

**Ablation 2: Classification Methods**
- Heuristic only
- ML only (RF, SVM, GB)
- CNN only
- Hybrid ensemble

**Ablation 3: OCR Impact**
- With OCR (EasyOCR)
- With OCR (PaddleOCR)
- With OCR ensemble
- Without OCR

**Ablation 4: Preprocessing**
- No preprocessing
- Minimal preprocessing (grayscale only)
- Full preprocessing

**Ablation 5: Ensemble Weights**
- Equal weights
- Learned weights (grid search)
- Oracle weights (best possible)

---

## Compute Requirements

### Target Hardware

**Primary**: CPU-only environment
- 4-8 CPU cores
- 16GB RAM
- No GPU required

**Optional**: GPU acceleration (for comparison)
- Single consumer GPU (RTX 3060 or similar)
- 8GB VRAM

### Runtime Targets

**Per-Image Runtime**:
- Heuristic classifier: <50ms
- ML classifier: <100ms
- CNN classifier: <500ms (CPU), <50ms (GPU)
- Full pipeline: <1s (CPU), <200ms (GPU)

**Dataset Runtime**:
- 1,000 images: <15 minutes (CPU)
- 5,000 images: <1 hour (CPU)

### Memory Targets

**Memory Footprint**:
- Model loading: <500MB
- Per-image processing: <200MB
- Peak memory: <2GB

---

## Implementation Plan

### Phase 1: Baseline Implementation (Week 1-2)

1. Implement random baseline
2. Implement heuristic-only classifier
3. Implement feature extraction pipeline
4. Implement ML classifier (RF only)
5. Run initial experiments on synthetic dataset

### Phase 2: Full Pipeline (Week 3-4)

1. Implement CNN classifier (optional)
2. Implement ensemble classifier
3. Implement OCR integration
4. Implement evaluation pipeline
5. Run full experiments

### Phase 3: Ablation Studies (Week 5-6)

1. Feature group ablations
2. Classification method ablations
3. OCR impact ablations
4. Preprocessing ablations
5. Ensemble weight optimization

### Phase 4: Analysis and Reporting (Week 7-8)

1. Error analysis
2. Statistical analysis
3. Paper figure generation
4. Results interpretation
5. Paper writing

---

## Evaluation Protocol

### Cross-Validation

**Method**: 5-fold stratified cross-validation
- Ensure each fold has balanced class distribution
- Report mean and std across folds
- Use same folds for all methods (fair comparison)

### Statistical Significance

**Tests**:
- Bootstrap confidence intervals (1,000 samples)
- McNemar's test for pairwise comparison
- Bonferroni correction for multiple comparisons

### Error Analysis

**Analysis Types**:
1. Per-class error rates
2. Confusion patterns (which classes confused?)
3. Feature importance analysis
4. Qualitative failure case analysis
5. OCR failure impact analysis

### Reproducibility

**Requirements**:
1. Random seed fixed for all experiments
2. Dataset fingerprint recorded
3. Library versions recorded
4. Git commit hash recorded
5. Full configuration saved
6. Code and data available

---

## Publication Strategy

### Target Venue

**Primary**: Computer Vision or ML conference/workshop
- CVPR (Chart Understanding Workshop)
- ECCV (Vision for Document Analysis)
- ICCV (Document Analysis Workshop)
- NeurIPS (Dataset and Benchmark Track)

**Secondary**: Application-oriented venues
- IEEE Transactions on Pattern Analysis and Machine Intelligence
- Pattern Recognition
- Expert Systems with Applications

### Paper Structure

**Title**: "Lightweight Chart Classification: Achieving Competitive Accuracy Without GPU Acceleration"

**Abstract**:
- Problem statement (chart classification, resource constraints)
- Proposed method (hybrid features, ensemble, CPU-optimized)
- Results (competitive accuracy, fast runtime)
- Contribution (lightweight architecture, comprehensive analysis)

**Introduction**:
- Motivation (resource-constrained deployment)
- Problem definition
- Challenges (OCR quality, feature engineering, ensemble design)
- Contributions (4 contributions)

**Related Work**:
- Chart classification literature
- OCR for document analysis
- Lightweight computer vision
- Ensemble methods

**Method**:
- System overview
- Feature extraction (geometric, structural)
- Classification methods (heuristic, ML, CNN, ensemble)
- OCR integration
- Implementation details

**Experiments**:
- Datasets
- Baselines
- Implementation details
- Results (accuracy, runtime, ablation)
- Statistical analysis
- Error analysis

**Discussion**:
- Findings interpretation
- Limitations
- Future work

**Conclusion**:
- Summary of contributions
- Impact statement

### Required Figures

1. System architecture diagram
2. Feature extraction pipeline
3. Classification accuracy comparison (bar chart)
4. Per-class accuracy (grouped bar chart)
5. Confusion matrix (heatmap)
6. Ablation results (table)
7. Runtime comparison (bar chart)
8. Feature importance (bar chart)
9. Error case examples (qualitative)

### Required Tables

1. Dataset statistics
2. Baseline comparison (accuracy, F1, runtime)
3. Ablation study results
4. Statistical significance tests
5. Per-class performance

---

## Risk Assessment

### Technical Risks

**Risk 1**: CNN training too slow on CPU
- **Mitigation**: Use pre-trained features, limit CNN to GPU comparison only

**Risk 2**: OCR quality too low for real datasets
- **Mitigation**: Implement OCR-free baseline, analyze OCR failure impact

**Risk 3**: Feature extraction too slow
- **Mitigation**: Profile and optimize, remove slow features

**Risk 4**: Real datasets not accessible
- **Mitigation**: Focus on synthetic dataset with rigorous validation

### Research Risks

**Risk 1**: Results not competitive with SOTA
- **Mitigation**: Focus on CPU-only comparison, emphasize efficiency

**Risk 2**: Novelty insufficient for publication
- **Mitigation**: Emphasize systematic analysis and lightweight contribution

**Risk 3**: Ablation studies inconclusive
- **Mitigation**: Design clear hypotheses, use statistical testing

---

## Success Criteria

### Minimum Viable Results

- Overall accuracy >85% on synthetic dataset
- Per-class accuracy >80% for all chart types
- Runtime <1s per image on CPU
- Clear ablation study insights
- Statistical significance vs. baselines

### Target Results

- Overall accuracy >90% on synthetic dataset
- Per-class accuracy >85% for all chart types
- Runtime <500ms per image on CPU
- Competitive with GPU-heavy methods (within 5% accuracy)
- Clear error analysis with actionable insights

### Stretch Goals

- Overall accuracy >92% on synthetic dataset
- Evaluation on real dataset (if accessible)
- Cross-dataset generalization study
- Publication in top-tier venue

---

## Timeline

### Week 1-2: Baseline Implementation
- Implement random and heuristic baselines
- Implement feature extraction
- Implement ML classifier (RF)
- Initial experiments

### Week 3-4: Full Pipeline
- Implement CNN (optional)
- Implement ensemble
- Implement OCR integration
- Full experiments

### Week 5-6: Ablation Studies
- Feature ablations
- Method ablations
- OCR ablations
- Ensemble optimization

### Week 7-8: Analysis and Writing
- Error analysis
- Statistical analysis
- Figure generation
- Paper writing

### Week 9-10: Revision and Submission
- Internal review
- Revisions
- Final submission

---

## Next Steps

1. **Phase 5**: Detailed migration plan
2. **Phase 6**: Implementation milestones
3. **Phase 7**: Comprehensive migration document
