# Implementation Milestones

## Phase 7: Implementation Milestones (10-12 Weeks)

This document provides detailed implementation milestones for the research project based on the final methodology, which includes both classification and question answering capabilities.

---

## Milestone 1: Infrastructure and Dataset Generation (Week 1-2)

**Objective**: Set up infrastructure and generate synthetic dataset with QA pairs

**Duration**: 2 weeks

### Deliverables

1. **Updated folder structure**
   - Add `src/qa/` module (classifier.py, templates.py, engine.py, histogram.py)
   - Add `src/data/qa_generator.py`
   - Update configuration files to include QA settings

2. **Dataset generation**
   - 5,000 synthetic chart images (1,000 per type)
   - 10,000 QA pairs (2 per image)
   - Dataset fingerprinting
   - Train/val/test splits (70/15/15)

3. **QA templates**
   - 5 question type templates
   - Template filling logic
   - Question type classifier

### Success Criteria

- [ ] All 5,000 images generated successfully
- [ ] All 10,000 QA pairs generated
- [ ] Dataset fingerprint recorded
- [ ] QA templates work for all question types
- [ ] Question type classifier achieves >90% accuracy on template classification

### Tasks

- [ ] Create src/qa/ module structure
- [ ] Implement src/data/qa_generator.py
- [ ] Implement QA templates in config/qa_templates.yaml
- [ ] Implement question type classifier
- [ ] Generate 5,000 synthetic charts
- [ ] Generate 10,000 QA pairs
- [ ] Implement dataset fingerprinting
- [ ] Create train/val/test splits
- [ ] Validate dataset quality
- [ ] Test QA generation on sample charts

### Estimated Effort

- Developer time: 40 hours
- Testing time: 10 hours
- Total: 50 hours

---

## Milestone 2: Preprocessing, OCR, and Detection (Week 2-3)

**Objective**: Implement preprocessing, OCR, and detection components

**Duration**: 1.5 weeks

### Deliverables

1. **Image preprocessor**
   - Grayscale, CLAHE, adaptive threshold
   - Morphological cleanup
   - Configurable operations

2. **OCR engine**
   - PaddleOCR integration
   - EasyOCR integration
   - Ensemble fusion
   - Caching

3. **Detection**
   - YOLO detector (if GPU available)
   - Traditional segmentation fallback
   - Element counting

### Success Criteria

- [ ] Preprocessor applies all operations correctly
- [ ] OCR works with both engines
- [ ] OCR ensemble improves accuracy
- [ ] Detection works for all chart types
- [ ] Fallback to traditional segmentation works

### Tasks

- [ ] Implement image preprocessor
- [ ] Implement OCR engine with both engines
- [ ] Implement OCR ensemble
- [ ] Implement OCR caching
- [ ] Implement YOLO detector
- [ ] Implement traditional segmentation
- [ ] Implement detection fallback logic
- [ ] Test preprocessing on sample images
- [ ] Test OCR on sample images
- [ ] Test detection on sample images

### Estimated Effort

- Developer time: 30 hours
- Testing time: 8 hours
- Total: 38 hours

---

## Milestone 3: Feature Extraction (Week 3)

**Objective**: Implement comprehensive feature extraction

**Duration**: 1 week

### Deliverables

1. **Geometric features** (35+ features)
   - Image dimensions
   - Contour statistics
   - Edge density
   - Line/circle detection

2. **Structural features** (15 features)
   - Axis detection
   - Grid detection
   - Legend detection
   - Layout analysis

3. **OCR-based features** (10 features)
   - Text count
   - Text density
   - Numeric text ratio

4. **Color features** (8 features)
   - Color histogram
   - Dominant colors
   - Color diversity

5. **Detection-based features** (5 features)
   - Element counts
   - Spatial distribution

### Success Criteria

- [ ] All 73 features extracted correctly
- [ ] Feature extraction <100ms per image
- [ ] No duplicate features
- [ ] Feature values are reasonable

### Tasks

- [ ] Implement geometric feature extractor
- [ ] Implement structural feature extractor
- [ ] Implement OCR-based feature extractor
- [ ] Implement color feature extractor
- [ ] Implement detection-based feature extractor
- [ ] Test feature extraction on sample images
- [ ] Profile feature extraction runtime
- [ ] Validate feature values

### Estimated Effort

- Developer time: 24 hours
- Testing time: 6 hours
- Total: 30 hours

---

## Milestone 4: Classification Baselines (Week 3-4)

**Objective**: Implement classification baselines

**Duration**: 1 week

### Deliverables

1. **Heuristic classifier**
   - Rule-based classification
   - Confidence computation

2. **ML classifiers**
   - Random Forest
   - SVM
   - Gradient Boosting
   - XGBoost

3. **CNN classifier** (optional)
   - EfficientNet-B0
   - Transfer learning

4. **Baseline experiments**
   - Random classifier
   - Heuristic-only
   - Feature-only ML
   - CNN-only

### Success Criteria

- [ ] Heuristic classifier works
- [ ] All ML classifiers train successfully
- [ ] CNN classifier trains (if GPU available)
- [ ] Baseline experiments complete
- [ ] Random classifier: ~20% accuracy
- [ ] Heuristic: ~60% accuracy
- [ ] ML: ~85% accuracy

### Tasks

- [ ] Implement heuristic classifier
- [ ] implement Random Forest classifier
- [ ] Implement SVM classifier
- [ ] Implement Gradient Boosting classifier
- [ ] Implement XGBoost classifier
- [ ] Implement CNN classifier (if GPU available)
- [ ] Run random baseline experiment
- [ ] Run heuristic-only experiment
- [ ] Run feature-only ML experiment
- [ ] Run CNN-only experiment
- [ ] Analyze baseline results

### Estimated Effort

- Developer time: 32 hours
- Testing time: 8 hours
- Total: 40 hours

---

## Milestone 5: Value Extraction (Week 4)

**Objective**: Implement chart-specific value extraction

**Duration**: 1 week

### Deliverables

1. **Bar chart extraction**
   - Bar detection
   - Height measurement
   - Axis mapping

2. **Line chart extraction**
   - Line detection
   - Point sampling
   - Axis mapping

3. **Pie chart extraction**
   - Slice detection
   - Angle calculation
   - Percentage mapping

4. **Scatter plot extraction**
   - Point detection
   - Coordinate mapping

5. **Histogram extraction**
   - Bin detection
   - Height calculation
   - Distribution analysis

### Success Criteria

- [ ] Value extraction works for all chart types
- [ ] Extracted values match ground truth within 5%
- [ ] Extraction <200ms per chart
- [ ] Handles edge cases (missing axes, rotated charts)

### Tasks

- [ ] Implement bar chart value extraction
- [ ] Implement line chart value extraction
- [ ] Implement pie chart value extraction
- [ ] Implement scatter plot value extraction
- [ ] Implement histogram value extraction
- [ ] Test extraction on sample charts
- [ ] Validate extracted values against ground truth
- [ ] Profile extraction runtime

### Estimated Effort

- Developer time: 32 hours
- Testing time: 8 hours
- Total: 40 hours

---

## Milestone 6: QA Module (Week 5)

**Objective**: implement question answering module

**Duration**: 1 week

### Deliverables

1. **Question type classifier**
   - Rule-based classification
   - Keyword matching
   - 5 question types

2. **QA templates**
   - Value retrieval templates
   - Comparison templates
   - Trend templates
   - Aggregation templates
   - Histogram-specific templates

3. **QA engine**
   - Template routing
   - Answer generation
   - Value lookup

4. **Histogram-specific QA**
   - Bin analysis
   - Distribution shape
   - Histogram-specific questions

### Success Criteria

- [ ] Question type classifier achieves >90% accuracy
- [ ] All templates work correctly
- [ ] QA engine generates answers correctly
- [ ] Histogram QA works for histogram-specific questions
- [ ] QA overhead <100ms per question

### Tasks

- [ ] Implement question type classifier
- [ ] Implement value retrieval templates
- [ ] Implement comparison templates
- [ ] Implement trend templates
- [ ] Implement aggregation templates
- [ ] Implement histogram-specific templates
- [ ] Implement QA engine
- [ ] Implement histogram-specific QA
- [ ] Test QA on sample questions
- [ ] Validate QA accuracy
- [ ] Profile QA runtime

### Estimated Effort

- Developer time: 32 hours
- Testing time: 8 hours
- Total: 40 hours

---

## Milestone 7: Classification Ensemble (Week 5-6)

**Objective**: Implement ensemble classification system

**Duration**: 1 week

### Deliverables

1. **Ensemble classifier**
   - Weighted voting
   - Confidence-based fusion
   - Multiple weight configurations

2. **Weight optimization**
   - Grid search for optimal weights
   - Cross-validation
   - Oracle weights (upper bound)

3. **Full classification experiments**
   - Ensemble vs. individual classifiers
   - Different weight configurations
   - Statistical analysis

### Success Criteria

- [ ] Ensemble classifier combines predictions correctly
- [ ] Ensemble outperforms best individual classifier
- [ ] Weight optimization finds good configuration
- [ ] Ensemble accuracy >90%
- [ ] Statistical significance vs. baselines

### Tasks

- [ ] Implement ensemble classifier
- [ ] Implement weighted voting
- [ ] Implement confidence-based fusion
- [ ] Implement weight optimization
- [ ] Run ensemble experiments
- [ ] Compare with individual classifiers
- [ ] Perform statistical analysis
- [ ] Analyze ensemble performance

### Estimated Effort

- Developer time: 24 hours
- Testing time: 6 hours
- Total: 30 hours

---

## Milestone 8: Full Pipeline Integration (Week 6)

**Objective**: Integrate all components into full pipeline

**Duration**: 1 week

### Deliverables

1. **Pipeline orchestrator**
   - All 9 stages integrated
   - Optional QA stage
   - Error handling

2. **Experiment runner**
   - Experiment initialization
   - Checkpoint system
   - Result management

3. **Full pipeline experiments**
   - Classification only
   - Classification + QA
   - End-to-end validation

### Success Criteria

- [ ] Pipeline runs end-to-end
- [ ] All stages execute in correct order
- [ ] QA stage works when enabled
- [ ] Experiment runner saves results
- [ ] Full pipeline classification >90%
- [ ] Full pipeline QA >80%

### Tasks

- [ ] Implement pipeline orchestrator
- [ ] Integrate all components
- [ ] Add optional QA stage
- [ ] Implement experiment runner
- [ ] Implement checkpoint system
- [ ] Run classification-only experiments
- [ ] Run classification+QA experiments
- [ ] Validate end-to-end results

### Estimated Effort

- Developer time: 32 hours
- Testing time: 8 hours
- Total: 40 hours

---

## Milestone 9: Ablation Studies (Week 7-8)

**Objective**: Conduct systematic ablation studies

**Duration**: 2 weeks

### Deliverables

1. **Feature group ablations**
   - Geometric only
   - Structural only
   - OCR-based only
   - Color only
   - Detection-based only
   - All features

2. **Classification method ablations**
   - Heuristic only
   - ML only (each model)
   - CNN only
   - YOLO only
   - Ensemble variants

3. **OCR impact ablations**
   - PaddleOCR only
   - EasyOCR only
   - Ensemble
   - No OCR

4. **Detection impact ablations**
   - YOLO only
   - Traditional only
   - No detection

5. **QA component ablations**
   - Template only
   - + question type classification
   - + value extraction
   - Full pipeline

6. **Preprocessing ablations**
   - No preprocessing
   - Grayscale only
   - Full preprocessing

### Success Criteria

- [ ] All ablation experiments complete
- [ ] Clear insights from each ablation
- [ ] Statistical significance for key comparisons
- [ ] Ablation results documented

### Tasks

- [ ] Run feature group ablations
- [ ] Run classification method ablations
- [ ] Run OCR impact ablations
- [ ] Run detection impact ablations
- [ ] Run QA component ablations
- [ ] Run preprocessing ablations
- [ ] Analyze ablation results
- [ ] Perform statistical tests
- [ ] Document ablation insights

### Estimated Effort

- Developer time: 48 hours
- Testing time: 16 hours
- Total: 64 hours

---

## Milestone 10: Analysis and Visualization (Week 8-9)

**Objective**: Comprehensive analysis and visualization

**Duration**: 1.5 weeks

### Deliverables

1. **Error analysis**
   - Per-class error rates
   - Confusion patterns
   - Qualitative failure cases
   - OCR failure impact
   - QA failure analysis

2. **Statistical analysis**
   - Bootstrap confidence intervals
   - McNemar's tests
   - Paired t-tests
   - Bonferroni correction

3. **Efficiency analysis**
   - Per-stage runtime breakdown
   - Memory footprint
   - Accuracy vs. runtime trade-off

4. **Histogram-specific analysis**
   - Histogram vs. other chart types
   - Histogram-specific challenges
   - Histogram QA performance

5. **Visualization**
   - Confusion matrices
   - Performance plots
   - Runtime plots
   - Feature importance
   - Error case examples

### Success Criteria

- [ ] Error analysis complete with patterns identified
- [ ] Statistical analysis with significance tests
- [ ] Efficiency analysis with trade-offs documented
- [ ] Histogram analysis complete
- [ ] All plots generated and saved
- [ ] Results are publication-ready

### Tasks

- [ ] Perform error analysis (classification)
- [ ] Perform error analysis (QA)
- [ ] Compute bootstrap confidence intervals
- [ ] Perform McNemar's tests
- [ ] Perform paired t-tests
- [ ] Apply Bonferroni correction
- [ ] Analyze runtime breakdown
- [ ] Analyze memory footprint
- [ ] Analyze accuracy vs. runtime
- [ ] Perform histogram-specific analysis
- [ ] Generate confusion matrices
- [ ] Generate performance plots
- [ ] Generate runtime plots
- [ ] Generate feature importance plots
- [ ] Generate error case examples

### Estimated Effort

- Developer time: 40 hours
- Testing time: 8 hours
- Total: 48 hours

---

## Milestone 11: Paper Writing (Week 9-10)

**Objective**: Write research paper

**Duration**: 2 weeks

### Deliverables

1. **Paper sections**
   - Abstract
   - Introduction
   - Related Work
   - Method
   - Experiments
   - Results
   - Discussion
   - Conclusion

2. **Figures and tables**
   - System architecture diagram
   - Performance comparison tables
   - Ablation study tables
   - Statistical significance tables
   - All required plots

3. **Supplementary material**
   - Additional ablation results
   - Additional error cases
   - Implementation details

### Success Criteria

- [ ] All paper sections written
- [ ] All figures and tables included
- [ ] Paper flows logically
- [ ] Results are clearly presented
- [ ] Contributions are highlighted
- [ ] Limitations are discussed

### Tasks

- [ ] Write abstract
- [ ] Write introduction
- [ ] Write related work
- [ ] Write method section
- [ ] Write experiments section
- [ ] Write results section
- [ ] Write discussion section
- [ ] Write conclusion
- [ ] Create all figures
- [ ] Create all tables
- [ ] Review and revise
- [ ] Proofread

### Estimated Effort

- Developer time: 48 hours
- Review time: 8 hours
- Total: 56 hours

---

## Milestone 12: Revision and Submission (Week 10-12)

**Objective**: Internal review, revision, and submission

**Duration**: 2 weeks

### Deliverables

1. **Internal review**
   - Peer review
   - Feedback collection
   - Revision plan

2. **Revisions**
   - Address feedback
   - Additional experiments if needed
   - Paper revisions

3. **Submission**
   - Final paper formatting
   - Supplementary material
   - Code repository preparation
   - Submission to venue

### Success Criteria

- [ ] Internal review completed
- [ ] All feedback addressed
- [ ] Paper formatted correctly
- [ ] Code repository ready
- [ ] Paper submitted

### Tasks

- [ ] Conduct internal review
- [ ] Collect feedback
- [ ] Create revision plan
- [ ] Implement revisions
- [ ] Run additional experiments if needed
- [ ] Format paper for venue
- [ ] Prepare supplementary material
- [ ] Prepare code repository
- [ ] Submit paper

### Estimated Effort

- Developer time: 32 hours
- Review time: 16 hours
- Total: 48 hours

---

## Summary

### Total Effort

| Milestone | Duration | Developer Time | Testing Time | Total |
|-----------|----------|----------------|--------------|-------|
| M1: Infrastructure/Dataset | 2 weeks | 40h | 10h | 50h |
| M2: Preprocessing/OCR/Detection | 1.5 weeks | 30h | 8h | 38h |
| M3: Feature Extraction | 1 week | 24h | 6h | 30h |
| M4: Classification Baselines | 1 week | 32h | 8h | 40h |
| M5: Value Extraction | 1 week | 32h | 8h | 40h |
| M6: QA Module | 1 week | 32h | 8h | 40h |
| M7: Classification Ensemble | 1 week | 24h | 6h | 30h |
| M8: Full Pipeline | 1 week | 32h | 8h | 40h |
| M9: Ablation Studies | 2 weeks | 48h | 16h | 64h |
| M10: Analysis/Viz | 1.5 weeks | 40h | 8h | 48h |
| M11: Paper Writing | 2 weeks | 48h | 8h | 56h |
| M12: Revision/Submission | 2 weeks | 32h | 16h | 48h |
| **Total** | **17.5 weeks** | **416h** | **108h** | **524h** |

### Timeline Summary

- **Week 1-2**: Infrastructure + Dataset Generation
- **Week 2-3**: Preprocessing + OCR + Detection
- **Week 3**: Feature Extraction
- **Week 3-4**: Classification Baselines
- **Week 4**: Value Extraction
- **Week 5**: QA Module
- **Week 5-6**: Classification Ensemble
- **Week 6**: Full Pipeline Integration
- **Week 7-8**: Ablation Studies
- **Week 8-9**: Analysis and Visualization
- **Week 9-10**: Paper Writing
- **Week 10-12**: Revision and Submission

### Critical Path

1. M1: Infrastructure/Dataset (must be first)
2. M2: Preprocessing/OCR/Detection (depends on M1)
3. M3: Feature Extraction (depends on M2)
4. M4: Classification Baselines (depends on M3)
5. M5: Value Extraction (depends on M4)
6. M6: QA Module (depends on M5)
7. M7: Classification Ensemble (depends on M4, M6)
8. M8: Full Pipeline (depends on M2, M3, M4, M5, M6, M7)
9. M9: Ablation Studies (depends on M8)
10. M10: Analysis/Viz (depends on M9)
11. M11: Paper Writing (depends on M10)
12. M12: Revision/Submission (depends on M11)

### Risk Mitigation

Each milestone has clear success criteria. If a milestone fails:
1. Investigate failure
2. Adjust timeline if needed
3. Focus on critical path items
4. Defer non-critical items if necessary

### Success Criteria Summary

**Classification**:
- Overall accuracy >90%
- Per-class accuracy >85%
- Statistical significance vs. baselines

**QA**:
- Overall accuracy >80%
- Per-type accuracy >75%
- Runtime overhead <100ms

**Efficiency**:
- Per-image runtime <1s (CPU)
- Memory footprint <2GB

**Analysis**:
- Clear ablation insights
- Statistical significance demonstrated
- Publication-ready figures and tables
