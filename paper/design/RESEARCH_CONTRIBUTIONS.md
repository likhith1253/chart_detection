# Research Contributions

## 1. Core Novelty

The central contribution is a unified chart understanding framework that generalizes a histogram-extraction pipeline into a multi-chart document reasoning system. The novelty is not in detection alone, and not in end-to-end classification alone, but in the decomposition of chart understanding into:

- chart localization
- chart type recognition
- parser routing
- chart-specific structural recovery
- unified schema serialization
- consistency validation

This creates a reproducible scientific pipeline rather than a single-task model.

## 2. Research Hypothesis

Charts of different families share enough structural primitives that a unified intermediate representation can support multiple specialized parsers with better transfer and auditability than histogram-only or monolithic approaches.

More specifically:

- a common detection and classification front-end should improve reuse across chart families
- chart-specific parsers should outperform one-size-fits-all extraction
- a universal JSON schema should make cross-dataset evaluation and downstream use more robust

## 3. Expected Contributions

1. A multi-chart document understanding architecture.
2. A universal chart schema that supports bar, line, scatter, pie, histogram, and box plot outputs.
3. A parser routing framework that separates recognition from structural extraction.
4. A consistency-checked extraction protocol with provenance.
5. A benchmark and evaluation methodology that spans multiple public datasets.
6. A reproducibility-first design that makes error analysis and ablations straightforward.

## 4. Difference From Histogram-Only Approaches

Histogram-only systems typically assume:

- one plot family
- one geometry family
- one extraction grammar
- one narrow set of labels

This proposal instead handles multiple chart grammars with shared infrastructure. The key difference is the presence of a router and a universal representation. That turns the system from a specialized extractor into a general chart reasoning framework.

## 5. Difference From Existing Chart Understanding Systems

Many chart understanding systems are either:

- classification-heavy
- QA-heavy
- retrieval-heavy
- end-to-end generation-heavy

This design is different because it treats structured extraction as the primary object of study. It is not just asking what the chart is; it is recovering the chart semantics in a form usable by machines.

## 6. Why This Is Publishable

This work is publishable if executed well because it addresses several reviewer-valued concerns:

- generalization beyond one chart family
- explicit intermediate supervision
- cross-dataset benchmark evaluation
- provable traceability from pixels to structured outputs
- modularity for ablations and error analysis
- relevance to document AI, chart QA, accessibility, and scientific literature mining

The framing is strong because the system is useful both as a benchmarked method and as a reusable research platform.

## 7. Main Scientific Risk

The biggest risk is that the architecture could be seen as an integration project unless the experiments clearly show:

- transfer across chart families
- improvement from routing
- robustness under dataset shift
- better structured extraction than simpler baselines

So the scientific claim must emphasize generalization and structured reasoning, not only engineering completeness.

