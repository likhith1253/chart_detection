# Final Research Question

## Phase 3: Research Question Definition

Based on the literature review and idea evaluation, this document defines the final research question and scope for the project.

---

## Primary Research Question

**Main Research Question**:
> "How can we achieve robust chart type classification under resource-constrained environments (CPU-only, minimal dependencies) while maintaining competitive accuracy with GPU-heavy deep learning approaches, with special focus on histogram detection and disambiguation?"

---

## Secondary Research Questions

1. **Feature Engineering**: What is the relative contribution of heuristic features vs. learned features in chart classification?

2. **Ensemble Methods**: Can ensemble methods bridge the gap between CPU-only and GPU-accelerated approaches for classification?

3. **OCR Impact**: How does OCR quality impact downstream classification performance?

4. **Histogram Disambiguation**: How can we effectively distinguish histograms from bar charts, which share similar visual characteristics?

5. **Efficiency Trade-offs**: What are the optimal accuracy vs. compute trade-offs for different application scenarios?

---

## Research Scope

### In Scope

**Tasks**:
1. Chart type classification (5 types: bar, line, pie, scatter, histogram)
2. Histogram detection and disambiguation (distinguishing histograms from bar charts)
3. Lightweight feature extraction for classification support

**Chart Types**:
- Bar charts (vertical, horizontal, grouped, stacked)
- Line charts (single, multiple, with/without markers)
- Pie charts
- Scatter plots
- Histograms (special focus for disambiguation)

**Compute Constraints**:
- CPU-only primary deployment
- Kaggle GPU only for CNN training when required
- Minimal dependencies (numpy, opencv, scikit-learn)
- Target runtime: <1s per image (CPU)

### Out of Scope

**Tasks**:
- Chart question answering
- Chart summarization
- Value extraction
- LLM integration
- Multi-modal language models
- Large foundation models

**Chart Types**:
- Complex multi-panel figures
- Infographics
- Dashboard screenshots
- 3D charts
- Geographic charts

---

## Research Contributions

### Contribution 1: Lightweight Classification Framework
- Design of a CPU-first architecture for chart classification
- Systematic analysis of feature extraction methods
- Comparison of single-model vs. ensemble approaches
- Demonstration that competitive accuracy is achievable without GPU

### Contribution 2: Hybrid Feature Engineering
- Novel combination of geometric, structural, and OCR-based features
- Systematic ablation of feature groups for classification
- Analysis of feature importance across chart types
- Generalizable feature extraction framework

### Contribution 3: Robust OCR Integration
- OCR ensemble with confidence-weighted fusion
- Analysis of OCR quality impact on classification
- Fallback strategies for OCR failure
- Cache-aware optimization for reproducibility

### Contribution 4: Histogram Disambiguation
- Specialized analysis of histogram vs. bar chart distinction
- Histogram-specific features for disambiguation
- Analysis of bin detection and distribution characteristics
- First systematic study on histogram classification challenges

### Contribution 5: Comprehensive Evaluation
- Multi-dataset evaluation (synthetic + real if available)
- Per-chart-type error analysis
- Cross-dataset generalization study
- Statistical significance testing
- Efficiency analysis (accuracy vs. compute)

---

## Novelty Statement

**Positioning**:
- Unlike GPU-heavy approaches (ChartQA, UniChart, MatCha), we focus on CPU-only deployment with systematic efficiency analysis
- Unlike black-box deep learning, we provide interpretable feature-based methods with clear ablation studies
- Unlike engineering-focused systems, we provide research-driven analysis with statistical validation
- Unlike general chart classification, we include specialized histogram disambiguation analysis

**Unique Aspects**:
1. CPU-first design with comprehensive efficiency analysis
2. Histogram-specialized disambiguation (under-explored in literature)
3. Systematic ablation across multiple dimensions (features, methods, OCR)
4. Focus on histogram vs. bar chart distinction (common confusion case)
5. Reproducible experimental protocol with statistical validation

---

## Hypotheses

### Hypothesis 1: Feature Engineering
**H1**: Hybrid features (geometric + structural + OCR) will significantly outperform single feature groups for classification.

### Hypothesis 2: Ensemble Methods
**H2**: Ensemble methods will bridge 50% of the accuracy gap between CPU-only and GPU-accelerated approaches.

### Hypothesis 3: OCR Impact
**H3**: OCR quality will have a moderate impact on classification (10-15% accuracy difference).

### Hypothesis 4: Histogram Disambiguation
**H4**: Histogram-specific features (bin detection, distribution analysis) will significantly improve histogram vs. bar chart disambiguation.

### Hypothesis 5: Efficiency Trade-offs
**H5**: There exists an optimal accuracy vs. compute trade-off point where marginal accuracy gains require disproportionate compute increases.

---

## Success Metrics

### Classification Metrics
- Overall accuracy: Target >90%, Minimum >85%
- Per-class accuracy: Target >85% for all types, Minimum >80%
- Histogram vs. Bar disambiguation accuracy: Target >95%, Minimum >90%
- Macro F1-score: Target >0.88, Minimum >0.82
- Weighted F1-score: Target >0.90, Minimum >0.85

### Efficiency Metrics
- Per-image runtime (CPU): Target <1s, Minimum <2s
- Memory footprint: Target <2GB peak, Minimum <4GB
- Model loading time: Target <5s, Minimum <10s

### Statistical Metrics
- Statistical significance vs. baselines: p < 0.05
- Confidence intervals: 95% via bootstrap
- Reproducibility: 100% (same results on same seed)

---

## Target Venues

### Primary Venues
- CVPR Workshop on Chart Understanding
- ECCV Workshop on Document Analysis
- ICDAR (International Conference on Document Analysis and Recognition)
- WACV (Winter Conference on Applications of Computer Vision)

### Secondary Venues
- Pattern Recognition
- Expert Systems with Applications
- IEEE Transactions on Pattern Analysis and Machine Intelligence
- Computer Vision and Image Understanding

---

## Risk Mitigation

### Risk 1: Histogram Disambiguation Inconclusive
**Mitigation**: Treat histogram as one of five chart types; if no significant differences found, emphasize general framework instead.

### Risk 2: CPU Performance Too Slow
**Mitigation**: Profile and optimize bottlenecks, reduce feature set, use caching, accept lower accuracy for faster runtime variant.

### Risk 3: Dataset Limitations
**Mitigation**: Use synthetic dataset with rigorous validation, if real dataset accessible use for generalization test only.

### Risk 4: Novelty Insufficient
**Mitigation**: Emphasize CPU-first contribution, efficiency analysis, histogram disambiguation, comprehensive ablation studies.

---

## Timeline Alignment

The research question is designed to be achievable within a 6-8 week timeline:

- **Week 1-2**: Dataset generation + baseline classification
- **Week 3-4**: Feature extraction + ensemble classification
- **Week 5-6**: Ablation studies
- **Week 7-8**: Analysis + paper writing

This timeline allows for:
- Focused implementation of classification only
- Comprehensive ablation studies
- Thorough analysis and evaluation
- Paper preparation and submission

---

## Conclusion

The final research question balances novelty with feasibility:

- **Focused scope**: Classification only, with histogram disambiguation as special focus
- **Maintains unique positioning**: CPU-first focus with efficiency analysis
- **Achievable timeline**: 6-8 weeks with clear milestones
- **Student-friendly**: Implementable by one student with CPU development and Kaggle GPU for training

This research question provides a strong foundation for publication-quality work while remaining feasible to implement within the constrained timeline.
