# Comprehensive Migration Document

## Executive Summary

This document presents a complete refactoring plan for the chart understanding research repository. The refactoring aims to improve modularity, maintainability, reproducibility, and publication-readiness while preserving high-quality components from the existing codebase.

### Key Recommendations

1. **Adopt new modular architecture** with 15+ independent modules
2. **Migrate 20 high-quality components** with minimal changes
3. **Delete 9 files** containing dead code or duplicates
4. **Merge 5 component pairs** to eliminate redundancy
5. **Implement new configuration system** using dataclasses and YAML
6. **Add comprehensive testing** with >80% coverage target
7. **Establish research-grade methodology** for publication

### Expected Outcomes

- **Code reduction**: ~8,000 LOC → ~5,000 LOC (37% reduction)
- **File reduction**: 49 files → 35 files (29% reduction)
- **Test coverage**: 0% → >80%
- **Modularity**: Monolithic → 15+ independent modules
- **Reproducibility**: Partial → Full snapshot support
- **Documentation**: Partial → Complete

### Effort Estimate

- **Total duration**: 32 working days (~6 weeks)
- **Total effort**: 324 hours (256h development + 68h testing)
- **Team size**: 1-2 developers
- **Risk level**: Low (incremental migration with rollback)

---

## Current State Analysis

### Repository Structure

The current repository has 49 Python files organized into the following structure:

```
chart_research_project/
├── Root level files (config.py, main.py, debug scripts)
├── src/pipeline/ (orchestration, config, reproducibility)
├── src/ocr/ (OCR engine, ensemble)
├── src/classifier/ (heuristic, ML classifiers)
├── src/extraction/ (value extraction)
├── src/research/ (CNN, YOLO, hybrid, dataset verifier)
├── src/segmentation/ (chart element segmenter)
├── src/feature_extraction/ (geometric, structural features)
├── src/features/ (duplicate feature extraction)
├── src/evaluation/ (evaluators)
├── src/preprocessing/ (image preprocessor)
├── src/caching/ (OCR cache)
├── src/dataset/ (dataset manager, versioning)
├── src/detection/ (YOLO detector wrapper)
├── src/profiling/ (runtime profiler)
├── src/visualization/ (plot generators)
└── ocr_env/ (virtual environment - should not be in repo)
```

### Technical Debt Identified

1. **Monolithic pipelines**: research_pipeline_final.py (829 lines), research_pipeline_v3.py (551 lines)
2. **Duplicate functionality**: Multiple evaluators, feature extractors, ML trainers
3. **Dead code**: Debug scripts, demo scripts, older pipeline versions
4. **Hardcoded configuration**: config.py mixes paths with constants
5. **Module organization**: Research module contains classifiers, detectors (wrong locations)
6. **Missing unit tests**: Only debug_pipeline.py for testing
7. **Virtual environment in repo**: ocr_env/ should not be committed

### High-Value Components

The following components are identified as high-quality and should be preserved:

1. **OCR Engine** - Robust ensemble with fallback, retries, caching
2. **OCR Ensemble** - Token-level fusion with confidence arbitration
3. **Chart Element Segmenter** - Research-grade segmentation with noise filtering
4. **Geometric Feature Extractor** - 35+ comprehensive features
5. **Dataset Manager** - Rich synthetic generation with variety
6. **Dataset Versioning** - Fingerprinting and deterministic splits
7. **Configuration Loader** - Well-designed dataclasses with YAML
8. **Reproducibility Utilities** - Seeding, git commit, library versions
9. **OCR Cache** - Disk-backed cache with metadata tracking
10. **Image Preprocessor** - Configurable preprocessing pipeline
11. **Heuristic Classifier** - Rule-based classification with disambiguation
12. **ML Classifier** - Multiple models with cross-validation
13. **CNN Classifier** - Transfer learning with early stopping
14. **Hybrid Classifier** - Confidence-weighted voting
15. **Value Extractor** - Chart-specific value extraction
16. **Research Evaluator** - Comprehensive research metrics
17. **Research Plot Generator** - Publication-ready plots
18. **Runtime Profiler** - Clean profiling with context manager
19. **YOLO Detector** - Thread-safe wrapper with fallback
20. **Dataset Verifier** - Verification and balancing

---

## Proposed New Architecture

### Design Principles

1. **Single Responsibility**: Each module has one clear purpose
2. **Modularity**: Components are loosely coupled and independently testable
3. **Reproducibility**: All experiments are fully reproducible with snapshots
4. **CPU-First**: Optimized for CPU with optional GPU acceleration
5. **Minimal Dependencies**: Only essential libraries, no technical debt
6. **Publication-Ready**: Built-in artifact generation for papers
7. **Understandable**: New researcher can understand structure in <1 hour

### New Folder Structure

```
chart_research_project/
├── README.md
├── requirements.txt
├── pyproject.toml
├── config/
│   ├── default.yaml
│   ├── smoke_test.yaml
│   └── ablation_variants.yaml
├── src/
│   ├── config/ (settings, constants)
│   ├── core/ (pipeline, experiment, reproducibility)
│   ├── data/ (dataset, versioning, synthetic)
│   ├── preprocessing/ (image processing)
│   ├── ocr/ (engine, ensemble)
│   ├── detection/ (yolo, segmenter)
│   ├── features/ (geometric, structural)
│   ├── classification/ (heuristic, ml, cnn, hybrid)
│   ├── extraction/ (values)
│   ├── evaluation/ (metrics, analysis)
│   ├── visualization/ (plots)
│   ├── utils/ (logging, caching, profiling)
│   └── models/ (loader)
├── models/pretrained/ (yolov8n.pt)
├── data/raw_images/
├── data/datasets/
├── cache/ocr/
├── results/experiments/
├── results/models/
├── results/plots/
├── results/reports/
├── tests/ (unit and integration tests)
├── examples/ (quick start, custom experiment)
└── scripts/ (train, ablation, report generation)
```

### Execution Flow

```
Configuration → Experiment Runner → Pipeline Orchestrator
                                              ↓
┌─────────────────────────────────────────────────────┐
│                  Pipeline Stages                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │Preprocess│→│    OCR   │→│Detection │          │
│  └──────────┘  └──────────┘  └──────────┘          │
│       ↓              ↓              ↓                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ Features │→│Classify  │→│Extract   │          │
│  └──────────┘  └──────────┘  └──────────┘          │
│       ↓              ↓              ↓                │
│  ┌──────────────────────────────────────────┐       │
│  │           Evaluation & Artifacts          │       │
│  └──────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────┘
```

### Configuration System

The new configuration system uses dataclasses for type safety and YAML for human-editable configs:

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

preprocessing:
  enabled: true
  grayscale: true
  clahe: true
  # ... other preprocessing options

ocr:
  enabled_engines: ["paddleocr", "easyocr"]
  ensemble_enabled: true

classification:
  methods: ["heuristic", "ml", "cnn", "hybrid"]
  # ... method-specific configs

cache:
  enabled: true
  root_dir: "cache"
```

---

## Migration Plan

### Strategy

**Incremental migration with legacy support**:
- Create new folder structure alongside existing code
- Migrate components one module at a time
- Maintain backward compatibility during transition
- Delete old code only after validation
- Use feature flags to switch between old/new implementations

### Migration Phases

| Phase | Duration | Description |
|-------|----------|-------------|
| Phase 1 | 2 days | Infrastructure setup (folders, config, modules) |
| Phase 2 | 2 days | Core utilities (logging, caching, reproducibility) |
| Phase 3 | 3 days | Data layer (dataset, versioning, synthetic) |
| Phase 4 | 3 days | Preprocessing and OCR |
| Phase 5 | 4 days | Detection and feature extraction |
| Phase 6 | 6 days | Classification (heuristic, ML, CNN, hybrid) |
| Phase 7 | 3 days | Evaluation and visualization |
| Phase 8 | 4 days | Pipeline orchestration |
| Phase 9 | 3 days | Testing and validation |
| Phase 10 | 2 days | Legacy cleanup |
| **Total** | **32 days** | **Complete migration** |

### Component Mapping

| Component | Current Location | New Location | Changes Required |
|-----------|----------------|--------------|------------------|
| OCR Engine | src/ocr/ocr_engine.py | src/ocr/engine.py | Import updates |
| OCR Ensemble | src/ocr/ensemble.py | src/ocr/ensemble.py | None |
| Segmenter | src/segmentation/ | src/detection/segmenter.py | Import updates |
| Geometric Features | src/feature_extraction/ | src/features/geometric.py | Merge with chart_feature_extractor |
| Dataset Manager | src/dataset/ | src/data/dataset.py | Split into dataset.py + synthetic.py |
| Dataset Versioning | src/dataset/ | src/data/versioning.py | None |
| Config Loader | src/pipeline/config.py | src/config/settings.py | Expand for new architecture |
| Reproducibility | src/pipeline/reproducibility.py | src/core/reproducibility.py | None |
| OCR Cache | src/caching/ocr_cache.py | src/utils/caching.py | Generalize to DiskCache |
| Image Preprocessor | src/preprocessing/ | src/preprocessing/image.py | None |
| Heuristic Classifier | src/classifier/ | src/classification/heuristic.py | Simplify interface |
| ML Classifier | src/classifier/ + src/research/ | src/classification/ml.py | Merge GridSearchCV |
| CNN Classifier | src/research/ | src/classification/cnn.py | Import updates |
| Hybrid Classifier | src/research/ | src/classification/hybrid.py | Import updates |
| Value Extractor | src/extraction/ | src/extraction/values.py | None |
| Research Evaluator | src/evaluation/ | src/evaluation/metrics.py | Rename |
| Research Plots | src/visualization/ | src/visualization/plots.py | None |
| Runtime Profiler | src/profiling/ | src/utils/profiling.py | None |
| YOLO Detector | src/detection/ + src/research/ | src/detection/yolo.py | Merge |
| Dataset Verifier | src/research/ | src/data/dataset_verifier.py | Import updates |

### Files to Delete

1. **Root level**:
   - config.py (replaced by config/)
   - debug_pipeline.py (replace with pytest)
   - demo_experiment.py (move to examples/)
   - research_pipeline_final.py (superseded)
   - research_pipeline_v3.py (superseded)
   - download_test_images.py (functionality in dataset manager)

2. **src/pipeline/**:
   - research_pipeline.py (superseded by modular pipeline)

3. **src/research/**:
   - All files (migrated to appropriate modules)

4. **src/segmentation/**:
   - All files (migrated to detection/)

5. **src/feature_extraction/**:
   - All files (migrated to features/)

6. **src/caching/**:
   - All files (migrated to utils/)

7. **src/profiling/**:
   - All files (migrated to utils/)

8. **src/evaluation/**:
   - evaluator.py (duplicate of research_evaluator.py)

9. **src/visualization/**:
   - plots.py (duplicate of research_plots.py)
   - debug_visualizer.py (move to tools/ or delete)

10. **ocr_env/**:
    - Virtual environment (should not be in repo)

---

## Implementation Milestones

### Milestone 1: Infrastructure Foundation (2 days)

**Deliverables**:
- New folder structure created
- Configuration system implemented (default.yaml, settings.py)
- Module structure initialized
- Testing framework set up (pytest, conftest.py)

**Success Criteria**:
- [ ] All directories created
- [ ] Config loads from YAML
- [ ] All modules importable
- [ ] Pytest runs successfully

### Milestone 2: Core Utilities (2 days)

**Deliverables**:
- Logging system (logging.py, JsonFormatter)
- Caching utilities (caching.py, DiskCache, OCRCache)
- Reproducibility utilities (reproducibility.py)
- Runtime profiler (profiling.py)

**Success Criteria**:
- [ ] Logging works with JSON formatter
- [ ] Cache is thread-safe
- [ ] Reproducibility snapshot generates
- [ ] Profiler tracks time accurately

### Milestone 3: Data Layer (3 days)

**Deliverables**:
- Dataset versioning (versioning.py)
- Dataset manager (dataset.py)
- Synthetic generation (synthetic.py)
- Dataset verifier (dataset_verifier.py)

**Success Criteria**:
- [ ] Fingerprinting works correctly
- [ ] Synthetic generation produces varied charts
- [ ] Dataset manager ensures minimum size
- [ ] Verifier detects corrupt images

### Milestone 4: Preprocessing and OCR (3 days)

**Deliverables**:
- Image preprocessor (preprocessing/image.py)
- OCR engine (ocr/engine.py)
- OCR ensemble (ocr/ensemble.py)

**Success Criteria**:
- [ ] Preprocessor applies all operations
- [ ] OCR works with both engines
- [ ] Ensemble fuses tokens correctly
- [ ] OCR cache reduces redundancy

### Milestone 5: Detection and Feature Extraction (4 days)

**Deliverables**:
- Chart element segmenter (detection/segmenter.py)
- Geometric feature extractor (features/geometric.py)
- YOLO detector (detection/yolo.py)

**Success Criteria**:
- [ ] Segmenter detects all element types
- [ ] Feature extractor produces 35+ features
- [ ] YOLO detector works with pretrained weights
- [ ] YOLO fallback to segmentation works

### Milestone 6: Classification (6 days)

**Deliverables**:
- Heuristic classifier (classification/heuristic.py)
- ML classifier (classification/ml.py)
- CNN classifier (classification/cnn.py)
- Hybrid classifier (classification/hybrid.py)

**Success Criteria**:
- [ ] Heuristic classifier produces chart types
- [ ] ML classifier trains all models
- [ ] CNN classifier trains on GPU/CPU
- [ ] Hybrid classifier combines predictions

### Milestone 7: Evaluation and Visualization (3 days)

**Deliverables**:
- Research evaluator (evaluation/metrics.py)
- Research plot generator (visualization/plots.py)
- Cleanup of duplicate files

**Success Criteria**:
- [ ] Evaluator computes all metrics
- [ ] Plot generator produces all plots
- [ ] No import errors after cleanup

### Milestone 8: Pipeline Orchestration (4 days)

**Deliverables**:
- Pipeline orchestrator (core/pipeline.py)
- Experiment runner (core/experiment.py)
- Entry point scripts (scripts/)
- Updated main.py

**Success Criteria**:
- [ ] Pipeline runs end-to-end
- [ ] Experiment runner creates checkpoints
- [ ] All scripts run without errors
- [ ] main.py produces valid results

### Milestone 9: Testing and Validation (3 days)

**Deliverables**:
- Unit tests for all modules (tests/)
- Integration tests (test_pipeline.py, test_experiment.py)
- Smoke test on small dataset
- Validation against old results

**Success Criteria**:
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Test coverage >80%
- [ ] Results match old pipeline within 1%

### Milestone 10: Legacy Cleanup (2 days)

**Deliverables**:
- Delete all old code files
- Delete old folder structure
- Update documentation (README.md)
- Final validation on full dataset

**Success Criteria**:
- [ ] All old code deleted
- [ ] No import errors
- [ ] All tests still pass
- [ ] Full experiment completes successfully

---

## Research Redesign

### Optimized Research Question

**Primary**: "How can we achieve robust chart type classification under resource-constrained environments (CPU-only, minimal dependencies) while maintaining competitive accuracy with GPU-heavy deep learning approaches?"

**Secondary**:
1. What is the relative contribution of heuristic vs. learned features?
2. Can ensemble methods bridge the CPU-GPU gap?
3. How does OCR quality impact classification?
4. What are the failure modes of lightweight systems?

### Research Contributions

1. **Lightweight Architecture** - CPU-first design with minimal dependencies
2. **Hybrid Feature Engineering** - Novel combination of geometric, structural, texture features
3. **Robust OCR Integration** - Ensemble with confidence-weighted fusion
4. **Comprehensive Evaluation** - Multi-dataset, statistical analysis, error analysis

### Experimental Design

**Baselines**:
- Random baseline
- Heuristic-only
- Feature-only ML
- CNN-only
- Full pipeline

**Ablation Studies**:
- Feature groups (geometric, structural, texture)
- Classification methods (heuristic, ML, CNN, ensemble)
- OCR impact (with/without, different engines)
- Preprocessing (none, minimal, full)
- Ensemble weights (equal, learned, oracle)

**Evaluation Metrics**:
- Overall accuracy, per-class accuracy
- Macro F1, weighted F1
- Precision, recall per class
- Runtime per image
- Memory footprint

**Statistical Analysis**:
- 95% confidence intervals (bootstrap)
- McNemar's test for significance
- Paired t-tests across methods

---

## Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| CNN training too slow on CPU | Medium | Medium | Use pre-trained features, limit CNN to GPU comparison |
| OCR quality too low for real datasets | Medium | Medium | Implement OCR-free baseline, analyze failure impact |
| Feature extraction too slow | Low | Medium | Profile and optimize, remove slow features |
| Real datasets not accessible | Medium | Low | Focus on synthetic dataset with rigorous validation |

### Migration Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Integration errors during migration | Medium | High | Incremental migration, rollback plan, thorough testing |
| Performance degradation | Low | Medium | Validate against old results, profile bottlenecks |
| Data loss during cleanup | Low | Critical | No data deleted until validation, git version control |
| Timeline overrun | Medium | Medium | Buffer time in estimates, prioritize critical path |

### Mitigation Strategies

1. **Git Branch Strategy**: Each phase on separate branch for easy rollback
2. **Incremental Validation**: Test each component before proceeding
3. **Backup**: Keep old code until final validation
4. **Documentation**: Update docs as code is migrated
5. **Communication**: Clear commit messages and PR descriptions

---

## Timeline and Resources

### Timeline

| Week | Milestones |
|------|------------|
| Week 1 | M1: Infrastructure, M2: Core Utilities |
| Week 2 | M3: Data Layer |
| Week 3 | M4: Preprocessing/OCR |
| Week 4 | M5: Detection/Features |
| Week 5-6 | M6: Classification |
| Week 7 | M7: Evaluation/Viz |
| Week 8 | M8: Pipeline |
| Week 9 | M9: Testing |
| Week 10 | M10: Cleanup |

### Resource Requirements

**Personnel**:
- 1-2 developers (Python, computer vision, ML experience)
- Optional: Research scientist for methodology guidance

**Hardware**:
- Development machine: 4-8 CPU cores, 16GB RAM
- Optional GPU: Single consumer GPU (RTX 3060 or similar)
- Storage: 10GB for code + datasets

**Software**:
- Python 3.9+
- Git for version control
- Pytest for testing
- Standard ML/CV libraries (numpy, opencv, scikit-learn, etc.)

### Effort Breakdown

| Phase | Duration | Developer Hours | Testing Hours | Total |
|-------|----------|-----------------|---------------|-------|
| M1: Infrastructure | 2 days | 16h | 2h | 18h |
| M2: Core Utilities | 2 days | 16h | 4h | 20h |
| M3: Data Layer | 3 days | 24h | 6h | 30h |
| M4: Preprocessing/OCR | 3 days | 24h | 6h | 30h |
| M5: Detection/Features | 4 days | 32h | 8h | 40h |
| M6: Classification | 6 days | 48h | 12h | 60h |
| M7: Evaluation/Viz | 3 days | 24h | 6h | 30h |
| M8: Pipeline | 4 days | 32h | 8h | 40h |
| M9: Testing | 3 days | 24h | 12h | 36h |
| M10: Cleanup | 2 days | 16h | 4h | 20h |
| **Total** | **32 days** | **256h** | **68h** | **324h** |

---

## Approval Checklist

### Architecture Approval

- [ ] New folder structure reviewed and approved
- [ ] Module organization approved
- [ ] Configuration system approved
- [ ] Execution flow approved
- [ ] Design principles accepted

### Migration Plan Approval

- [ ] Migration strategy accepted (incremental with rollback)
- [ ] Phase breakdown approved
- [ ] Timeline acceptable (32 days)
- [ ] Resource requirements feasible
- [ ] Risk mitigation plan sufficient

### Component Mapping Approval

- [ ] All high-value components identified
- [ ] Component mapping correct
- [ ] Files to delete list approved
- [ ] Merge operations approved

### Research Redesign Approval

- [ ] Research question clear and focused
- [ ] Contributions well-defined
- [ ] Experimental design sound
- [ ] Evaluation metrics appropriate
- [ ] Statistical analysis plan adequate

### Implementation Milestones Approval

- [ ] All 10 milestones defined
- [ ] Success criteria clear
- [ ] Deliverables specific
- [ ] Effort estimates realistic
- [ ] Dependencies between milestones understood

### Risk Assessment Approval

- [ ] All risks identified
- [ ] Likelihood and impact assessments reasonable
- [ ] Mitigation strategies sufficient
- [ ] Rollback plan acceptable

---

## Next Steps

Upon approval of this migration document:

1. **Create feature branch**: `git checkout -b migration/refactor`
2. **Begin Milestone 1**: Infrastructure setup
3. **Track progress**: Update this document with completion status
4. **Regular reviews**: Weekly status reviews to ensure alignment
5. **Final validation**: End-to-end test before merging to main

---

## Appendices

### Appendix A: Detailed File Inventory

See `MIGRATION_AUDIT.md` for complete file-by-file analysis.

### Appendix B: Architecture Design Details

See `ARCHITECTURE_DESIGN.md` for complete architecture specification.

### Appendix C: Reuse Analysis

See `REUSE_ANALYSIS.md` for detailed component quality assessment.

### Appendix D: Research Methodology

See `RESEARCH_REDESIGN.md` for complete research design.

### Appendix E: Migration Plan Details

See `MIGRATION_PLAN.md` for step-by-step migration procedures.

### Appendix F: Implementation Milestones

See `IMPLEMENTATION_MILESTONES.md` for detailed milestone specifications.

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-01-XX | Cascade | Initial comprehensive migration document |

---

## Approval Signatures

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Project Lead | | | |
| Technical Lead | | | |
| Research Lead | | | |
| | | | |
