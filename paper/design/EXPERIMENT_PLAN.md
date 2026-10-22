# Experiment Plan

## 1. Experimental Philosophy

The experiments should answer four questions:

1. Can the system localize and classify charts reliably?
2. Can specialized parsers recover structure accurately?
3. Does the universal schema generalize across datasets?
4. Does a modular design outperform a monolithic or histogram-only baseline under dataset shift?

## 2. Stage 1: Detection Experiments

Goal:
- Measure chart localization quality on document images.

Metrics:
- mAP
- recall at IoU thresholds
- chart-level F1

Comparisons:
- detector with and without preprocessing
- detector trained on single-dataset versus multi-dataset data

## 3. Stage 2: Classification Experiments

Goal:
- Evaluate chart type recognition for all supported families.

Metrics:
- accuracy
- macro-F1
- confusion matrix
- calibration error

Comparisons:
- CNN baseline
- class-balanced training
- cross-dataset transfer

## 4. Stage 3: Extraction Experiments

Goal:
- Evaluate the structural parsers on each chart family.

Metrics:
- entity localization F1
- numeric MAE
- sequence alignment score
- structural exact match
- schema validity rate

Per chart type:

- Bar: category/value recovery
- Line: ordered point recovery
- Scatter: point coordinate recovery
- Pie: slice proportion recovery
- Histogram: bin edge and count recovery
- Box plot: quartile and whisker recovery

## 5. Cross-Dataset Evaluation

Goal:
- Measure generalization from one dataset to another.

Protocol:

- train on one dataset family
- test on held-out public datasets
- evaluate with unified metrics

Examples:

- PlotQA to ChartQA
- DVQA to FigureQA
- mixed training to Benetech-like chart domains if available

## 6. Generalization Experiments

Goal:
- Test whether the framework learns chart-agnostic primitives.

Design:

- train router and parsers on a subset of chart types
- evaluate zero-shot or few-shot on the held-out family

Key question:

- does structural recovery degrade gracefully when the target chart family was not seen in training?

## 7. Ablation Studies

Mandatory ablations:

1. No router, single parser for all charts
2. No OCR
3. No axis module
4. No legend module
5. No consistency checker
6. Single-dataset training
7. Multi-dataset training
8. With versus without preprocessing

Interpretation:

- These ablations will establish which modules are scientifically necessary rather than just convenient.

## 8. Runtime Analysis

Measure:

- end-to-end latency
- module-wise latency
- memory use
- throughput per page

Why:

- publication reviewers often ask whether the system is practical or merely accurate.

## 9. Error Analysis

Stratify errors by:

- chart family
- image resolution
- OCR difficulty
- axis scale
- legend complexity
- multi-series density
- low-quality scans

Report:

- representative failure cases
- visually similar misclassifications
- numeric deviations
- schema violations

## 10. Comparison With Previous Papers

Compare against:

- histogram-only extraction systems
- chart classification systems
- chart QA systems that do not produce structured outputs
- end-to-end chart understanding systems

Focus on:

- generalization
- interpretability
- schema completeness
- extraction fidelity

## 11. Success Criteria

The system should demonstrate:

- reliable chart localization
- strong classification performance
- valid universal schema generation
- consistent gains from modular parsing
- measurable cross-dataset robustness

