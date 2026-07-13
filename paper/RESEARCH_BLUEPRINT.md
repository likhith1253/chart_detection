# Research Blueprint

## Comprehensive Research Plan for Chart Classification Under Resource Constraints

---

## Executive Summary

This document provides a complete blueprint for conducting research on lightweight chart classification under resource-constrained environments. The research aims to achieve competitive accuracy with GPU-heavy deep learning approaches while operating on CPU-only systems with minimal dependencies, with special focus on histogram detection and disambiguation.

**Key Highlights**:
- **Research Focus**: Chart classification with histogram disambiguation on CPU
- **Dataset**: 2,500 synthetic charts
- **Timeline**: 6-8 weeks
- **Target Accuracy**: Classification >90%, Histogram vs. Bar disambiguation >95%
- **Target Runtime**: <1s per image (CPU)
- **Unique Contribution**: CPU-first design with histogram disambiguation specialization

---

## Table of Contents

1. [Research Question and Objectives](#research-question-and-objectives)
2. [Literature Review Summary](#literature-review-summary)
3. [Research Methodology](#research-methodology)
4. [Experiment Plan](#experiment-plan)
5. [Implementation Milestones](#implementation-milestones)
6. [Success Criteria](#success-criteria)
7. [Timeline](#timeline)
8. [Resources Required](#resources-required)
9. [Risk Management](#risk-management)
10. [Deliverables](#deliverables)

---

## Research Question and Objectives

### Primary Research Question

> "How can we achieve robust chart type classification under resource-constrained environments (CPU-only, minimal dependencies) while maintaining competitive accuracy with GPU-heavy deep learning approaches, with special focus on histogram detection and disambiguation?"

### Secondary Research Questions

1. What is the relative contribution of heuristic features vs. learned features in chart classification?
2. Can ensemble methods bridge the gap between CPU-only and GPU-accelerated approaches?
3. How does OCR quality impact downstream classification performance?
4. How can we effectively distinguish histograms from bar charts, which share similar visual characteristics?
5. What are the optimal accuracy vs. compute trade-offs for different application scenarios?

### Research Objectives

1. **Design a lightweight classification framework** for CPU-only deployment
2. **Develop hybrid feature engineering** combining geometric, structural, OCR-based, and histogram-specific features
3. **Implement robust OCR integration** with ensemble fusion and fallback strategies
4. **Conduct comprehensive evaluation** with statistical analysis and ablation studies
5. **Perform histogram-specialized analysis** to address histogram vs. bar disambiguation

### Research Contributions

1. **Lightweight Classification Framework**: CPU-first architecture for chart classification
2. **Hybrid Feature Engineering**: Novel combination of features with systematic ablation
3. **Robust OCR Integration**: Ensemble fusion with confidence-weighted voting
4. **Histogram Disambiguation**: Specialized analysis of histogram vs. bar chart distinction
5. **Comprehensive Evaluation**: Multi-dataset evaluation with statistical analysis
6. **Histogram-Specialized Analysis**: First systematic study on histogram classification challenges

---

## Literature Review Summary

### Key Findings from 18+ Papers

#### State-of-the-Art Trends
- **Unified Frameworks**: ChartReader, UniChart integrate multiple tasks (classification, QA, summarization)
- **Large-Scale Pretraining**: 611K+ charts for vision-language models
- **Focus Areas**: Chart QA, summarization, chain-of-thought reasoning
- **Architecture**: Vision-language models (T5, BART, Donut), Transformer-based detection

#### Dataset Scale
- ChartQA: 20K+ charts
- PlotQA: 224K plots, 28.9M QA pairs
- UniChart pretraining: 611K charts
- ChartSumm: 84K charts with summaries

#### Research Gaps Identified
- **Limited histogram-specific research**: Most work focuses on bar, line, pie charts
- **Few CPU-only approaches**: Recent literature is GPU-heavy
- **Gap in lightweight models**: Efficiency not systematically studied
- **Histogram vs. bar disambiguation**: Under-explored in literature

#### Positioning
- Unlike GPU-heavy approaches (ChartQA, UniChart), we focus on CPU-only deployment
- Unlike black-box deep learning, we provide interpretable feature-based methods
- Unlike general chart classification, we include histogram disambiguation specialization
- Unlike broad chart understanding frameworks, we focus on classification only

---

## Research Methodology

### Pipeline Architecture

The research employs a 7-stage pipeline:

```
Input Image → Preprocessing → OCR → Detection → Feature Extraction → 
Classification → Evaluation → Artifacts
```

### Stage 1: Preprocessing
- Grayscale conversion
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Adaptive thresholding
- Morphological cleanup
- Deskewing (optional)

### Stage 2: OCR Engine
- PaddleOCR (primary, higher accuracy)
- EasyOCR (secondary, faster)
- Ensemble fusion (confidence-weighted)
- Disk-backed caching for reproducibility

### Stage 3: Detection
- Traditional segmentation (OpenCV-based, CPU-only)
- Element counting (bars, lines, points, pie slices, bins, axes)

### Stage 4: Feature Extraction
- **Geometric & Structural features** (23): Image dimensions, contours, edges, lines, circles, axis/grid/legend detection
- **OCR-based features** (4): Text count, density, numeric ratio
- **Histogram-specific features** (3): Bin detection, gap analysis, axis gap analysis

### Stage 5: Classification Ensemble
- **Heuristic classifier**: Rule-based classification
- **ML classifiers**: Random Forest, SVM, Gradient Boosting, XGBoost
- **CNN classifier** (Kaggle GPU): EfficientNet-B0

### Stage 6: Evaluation
- **Classification metrics**: Accuracy, F1, confusion matrix, per-class accuracy
- **Histogram vs. Bar disambiguation**: Specific metric for histogram/bar distinction
- **Efficiency metrics**: Runtime per stage, memory footprint, model loading time
- **Statistical analysis**: Bootstrap CI, McNemar's test, paired t-tests

### Stage 7: Artifact Generation
- Confusion matrices
- Performance plots
- Runtime breakdown
- Feature importance
- Error analysis
- Histogram-specific analysis

---

## Experiment Plan

### Dataset

**Primary Dataset**: Synthetic Charts
- **Size**: 2,500 images (500 per chart type)
- **Chart Types**: Bar, Line, Pie, Scatter, Histogram
- **Variation**: 5 fonts, 10 color palettes, 3 scales, 2 orientations, 3 noise levels
- **Splits**: Train 70%, Validation 15%, Test 15% (stratified)

### Baseline Methods

**Classification Baselines**:
1. Random classifier (lower bound)
2. Heuristic-only classifier
3. Feature-only ML (Random Forest)
4. CNN-only (EfficientNet-B0)
5. Full pipeline (proposed)

### Ablation Studies

**Ablation 1: Feature Groups**
- Geometric only
- Structural only
- OCR-based only
- Color only
- Detection-based only
- Histogram-specific only
- All features

**Ablation 2: Classification Methods**
- Heuristic only
- ML only (RF, SVM, GB, XGB)
- CNN only
- Ensemble variants

**Ablation 3: OCR Impact**
- With OCR
- No OCR

**Ablation 5: Preprocessing**
- No preprocessing
- Grayscale only
- Full preprocessing

### Evaluation Protocol

**Cross-Validation**: 5-fold stratified cross-validation
- Same folds for all methods
- Report mean and standard deviation

**Statistical Analysis**:
- Bootstrap confidence intervals (1,000 samples)
- McNemar's test for pairwise comparison
- Paired t-tests across folds
- Bonferroni correction for multiple comparisons

**Reproducibility**:
- Fixed random seed (42)
- Dataset fingerprinting
- Library version recording
- Git commit hash
- Full configuration snapshot

---

## Implementation Milestones

### Milestone 1: Infrastructure and Dataset Generation (Week 1)
- Set up folder structure (no QA module)
- Generate 5,000 synthetic chart images
- Generate 2,500 synthetic chart images
- Implement dataset fingerprinting

### Milestone 2: Preprocessing and OCR (Week 1-2)
- Implement image preprocessor
- Implement OCR engine with PaddleOCR and EasyOCR

### Milestone 3: Detection and Feature Extraction (Week 2)
- Implement traditional segmentation
- Implement geometric & structural feature extractor (~23 features)
- Implement OCR-based feature extractor (~4 features)
- Implement histogram-specific feature extractor (~3 features)

### Milestone 4: Classification Baselines (Week 2-3)
- Implement heuristic classifier
- Implement ML classifiers (RF, SVM, GB, XGBoost)
- Implement CNN classifier (EfficientNet-B0) on Kaggle GPU
- Run baseline experiments (random, heuristic, ML, CNN)

### Milestone 5: Classification Ensemble (Week 3-4)
- Implement ensemble classifier
- Run ensemble experiments

### Milestone 6: Full Pipeline Integration (Week 4)
- Integrate all components into pipeline
- Implement experiment runner
- Run classification experiments

### Milestone 7: Ablation Studies (Week 5-6)
- Run feature group ablations (including histogram-specific)
- Run classification method ablations
- Run OCR impact ablations
- Run preprocessing ablations

### Milestone 8: Analysis and Visualization (Week 6-7)
- Perform error analysis (classification)
- Perform statistical analysis (bootstrap, McNemar, t-tests)
- Analyze efficiency (runtime, memory)
- Perform histogram-specific analysis (disambiguation)
- Generate all plots and figures

### Milestone 9: Paper Writing (Week 7-8)
- Write all paper sections
- Create all figures and tables
- Review and revise

---

## Success Criteria

### Classification Success

**Minimum Viable**:
- Overall accuracy >85%
- Per-class accuracy >80%
- Histogram vs. Bar disambiguation >90%
- Macro F1 >0.82
- Statistical significance vs. baselines (p < 0.05)

**Target**:
- Overall accuracy >90%
- Per-class accuracy >85%
- Histogram vs. Bar disambiguation >95%
- Macro F1 >0.88
- Statistical significance vs. all baselines (p < 0.01)

**Stretch**:
- Overall accuracy >92%
- Per-class accuracy >88%
- Histogram vs. Bar disambiguation >97%
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

## Timeline

### Week-by-Week Breakdown

**Week 1**: Infrastructure + Dataset Generation
- Set up folder structure (no QA module)
- Generate synthetic dataset (5,000 images)
- Implement dataset fingerprinting

**Week 1-2**: Preprocessing + OCR
- Implement preprocessing
- Implement OCR engine
- Test OCR quality

**Week 2**: Detection + Feature Extraction
- Implement detection (traditional)
- Implement all feature extractors (~30 features)
- Test feature extraction

**Week 2-3**: Classification Baselines
- Implement all classifiers (heuristic, ML, CNN)
- Run baseline experiments
- CNN training on Kaggle GPU

**Week 3-4**: Classification Ensemble
- Implement ensemble classifier
- Run ensemble experiments

**Week 4**: Full Pipeline Integration
- Integrate all components into pipeline
- Run classification experiments

**Week 5-6**: Ablation Studies
- Run all ablation experiments
- Analyze results

**Week 6-7**: Analysis and Visualization
- Perform comprehensive analysis
- Generate all plots

**Week 7-8**: Paper Writing
- Write all sections
- Create figures and tables

### Critical Path

1. Infrastructure/Dataset (Week 1)
2. Preprocessing/OCR (Week 1-2)
3. Detection/Features (Week 2)
4. Classification Baselines (Week 2-3)
5. Classification Ensemble (Week 3-4)
6. Full Pipeline (Week 4)
7. Ablation Studies (Week 5-6)
8. Analysis/Viz (Week 6-7)
9. Paper Writing (Week 7-8)

### Buffer Time

- Timeline is 8 weeks, can be compressed to 6 weeks if needed
- Focus on critical path items if time constrained

---

## Resources Required

### Hardware

**Minimum Requirements**:
- CPU: 4-8 cores
- RAM: 16GB
- Storage: 50GB
- GPU: Not required (Kaggle GPU for CNN training only)

**Recommended Requirements**:
- CPU: 8+ cores
- RAM: 32GB
- Storage: 100GB
- GPU: Kaggle GPU for CNN training (free)

### Software

**Core Dependencies**:
- Python 3.9+
- numpy >= 1.24.0
- pandas >= 2.0.0
- opencv-python >= 4.8.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- pyyaml >= 6.0
- pillow >= 10.0.0
- scipy >= 1.11.0

**OCR & ML Dependencies**:
- easyocr >= 1.7.0
- paddleocr >= 2.7.0
- torch >= 2.0.0
- torchvision >= 0.15.0
- xgboost >= 2.0.0

### Personnel

**Required Roles**:
- 1 Student Researcher (full-time)

**Time Commitment**:
- Student Researcher: 40 hours/week for 6-8 weeks

### Budget

**Software**: Free (open-source)
**Hardware**: $0 (CPU development, Kaggle GPU free)
**Personnel**: Depends on institution
**Conference Fees**: $500-1,000 (for submission)
**Total**: $500-1,000 (minimal)

---

## Risk Management

### Technical Risks

**Risk 1: CNN training too slow on CPU**
- **Probability**: High
- **Impact**: Medium
- **Mitigation**: Use Kaggle GPU for CNN training only, limit CNN to GPU comparison

**Risk 2: OCR quality too low for real datasets**
- **Probability**: Medium
- **Impact**: High
- **Mitigation**: Implement OCR-free baseline, analyze OCR failure impact, focus on synthetic dataset

**Risk 3: Feature extraction too slow**
- **Probability**: Low
- **Impact**: Medium
- **Mitigation**: Profile and optimize, remove slow features, accept lower accuracy for faster runtime

**Risk 4: Histogram disambiguation inconclusive**
- **Probability**: Medium
- **Impact**: Low
- **Mitigation**: Treat histogram as one of five chart types, emphasize general framework if no significant differences

### Research Risks

**Risk 1: Results not competitive with SOTA**
- **Probability**: Medium
- **Impact**: High
- **Mitigation**: Focus on CPU-only comparison, emphasize efficiency trade-offs, target application-oriented venues

**Risk 2: Novelty insufficient for publication**
- **Probability**: Low
- **Impact**: High
- **Mitigation**: Emphasize CPU-first contribution, histogram disambiguation specialization, comprehensive ablation studies

**Risk 3: Ablation studies inconclusive**
- **Probability**: Low
- **Impact**: Medium
- **Mitigation**: Design clear hypotheses, use statistical testing, report non-significant results honestly

**Risk 4: Dataset limitations**
- **Probability**: Medium
- **Impact**: Medium
- **Mitigation**: Use synthetic dataset with rigorous validation, report limitations clearly

### Timeline Risks

**Risk 1: Experiments take longer than expected**
- **Probability**: Medium
- **Impact**: Medium
- **Mitigation**: Parallel processing where possible, prioritize key experiments, use smaller dataset for initial testing

**Risk 2: Paper writing takes longer than expected**
- **Probability**: Low
- **Impact**: Medium
- **Mitigation**: Start writing early, write sections incrementally, allocate buffer time

### Risk Response Plan

For each risk:
1. **Monitor**: Track risk indicators throughout project
2. **Assess**: Evaluate if risk materializes
3. **Respond**: Implement mitigation strategy
4. **Escalate**: If mitigation fails, escalate to project lead

---

## Deliverables

### Code Deliverables

1. **Complete codebase** with all modules implemented (no QA module)
2. **Configuration files** for all experiments
3. **Trained models** for all classifiers
4. **Dataset generation scripts** for reproducibility
5. **Experiment scripts** for running all experiments
6. **Documentation** for installation and usage

### Data Deliverables

1. **Synthetic dataset** (5,000 images)
2. **Dataset fingerprint** for reproducibility
3. **Ground truth annotations** for all images
4. **Experiment results** (CSV, JSON formats)

### Research Deliverables

1. **Research paper** (8-10 pages)
2. **Supplementary material** (additional experiments, error cases)
3. **Figures and tables** (all publication-ready)
4. **Ablation study results** (detailed analysis)
5. **Statistical analysis report** (all tests and results)

### Documentation Deliverables

1. **README.md** with installation and usage instructions
2. **ARCHITECTURE.md** with system architecture
3. **METHODOLOGY.md** with detailed methodology
4. **EXPERIMENT_PLAN.md** with experiment design
5. **REPRODUCIBILITY.md** with reproduction instructions

---

## Publication Strategy

### Target Venues

**Primary Venues** (Top-tier workshops):
- CVPR Workshop on Chart Understanding
- ECCV Workshop on Document Analysis
- ICDAR (International Conference on Document Analysis and Recognition)
- WACV (Winter Conference on Applications of Computer Vision)

**Secondary Venues** (Application-oriented journals):
- Pattern Recognition
- Expert Systems with Applications
- IEEE Transactions on Pattern Analysis and Machine Intelligence
- Computer Vision and Image Understanding

**Workshop Papers** (Specialized):
- NeurIPS Workshop on AI for Good
- ICCV Workshop on Vision for Document Analysis
- EMNLP Workshop on Document Intelligence

### Paper Structure

**Title**: "Lightweight Chart Classification: Achieving Competitive Accuracy Without GPU Acceleration with Histogram Disambiguation"

**Abstract**: Problem statement, proposed method, results, contributions

**Introduction**: Motivation, problem definition, challenges, contributions

**Related Work**: Chart classification, OCR for documents, lightweight CV, ensemble methods

**Method**: System overview, feature extraction, classification, histogram disambiguation, implementation details

**Experiments**: Datasets, baselines, implementation details, results, statistical analysis, error analysis

**Discussion**: Findings interpretation, limitations, future work

**Conclusion**: Summary of contributions, impact statement

### Required Figures

1. System architecture diagram
2. Feature extraction pipeline
3. Classification accuracy comparison
4. Per-class accuracy
5. Confusion matrix
8. Ablation results
9. Runtime comparison
10. Feature importance
11. Error case examples
12. Histogram-specific analysis

### Required Tables

1. Dataset statistics
2. Baseline comparison (classification)
4. Ablation study results (classification)
6. Statistical significance tests
7. Per-class performance

---

## Conclusion

This research blueprint provides a comprehensive plan for conducting publication-quality research on lightweight chart understanding. The plan is designed to be:

- **Feasible**: Achievable within 10-12 weeks with clear milestones
- **Rigorous**: Systematic methodology with statistical analysis
- **Novel**: CPU-first focus with histogram specialization
- **Reproducible**: Full snapshot and fingerprinting protocol
- **Publication-Ready**: Comprehensive evaluation and analysis

The research addresses the gap in CPU-only chart understanding while providing a unified framework for both classification and question answering. The systematic ablation studies and comprehensive analysis will provide valuable insights for the research community.

---

## Appendix

### A. Detailed Task List

See `new_implementation_milestones.md` for detailed task breakdown.

### B. Configuration Templates

See `ARCHITECTURE_DESIGN.md` for configuration system details.

### C. Statistical Analysis Methods

See `experiment_plan.md` for statistical analysis protocols.

### D. Literature Review

See `literature_review.md` for complete literature review.

### E. Research Question Evolution

See `idea_evaluation.md` and `final_research_question.md` for research question development.

---

## Document Version History

- **v1.0** (2025-01-15): Initial research blueprint
- Based on literature review of 18+ papers
- Incorporates partial redesign (Option 2 from idea evaluation)
- Updated to include QA capabilities
- Timeline: 10-12 weeks

---

## Contact

For questions or clarifications about this research blueprint, refer to the project repository or contact the research lead.
