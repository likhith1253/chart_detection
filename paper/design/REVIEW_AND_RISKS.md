# Design Review

## 1. Weaknesses

- The architecture depends heavily on OCR quality, especially for small tick labels and dense legends.
- A universal schema can become too abstract if it is not enforced with strict validation rules.
- Parser specialization increases engineering and research complexity.
- Cross-dataset benchmarks may still reflect dataset-specific visual conventions rather than pure chart reasoning.

## 2. Novelty Risks

- Reviewers may view the work as a modular integration of known components unless the routing and universal schema are shown to improve measurable outcomes.
- Histogram-only extraction ideas may be considered too narrow as a prior baseline, so the paper must frame the novelty as multi-chart generalization and structured output.
- If the contribution does not show gains on unseen chart families, it may be seen as incremental.

## 3. Potential Reviewer Criticisms

1. "This is just a pipeline."
2. "The schema is hand-designed and may not generalize."
3. "OCR dominates the results, so the chart reasoning is not isolated."
4. "The benchmark is too heterogeneous to compare fairly."
5. "The system depends on dataset-specific heuristics."
6. "No evidence that routing is better than a unified parser."

## 4. How to Strengthen the Paper

1. Show measurable gains from routing versus a single parser.
2. Report schema validity, not just raw numeric accuracy.
3. Include strong cross-dataset transfer experiments.
4. Separate OCR errors from parser errors in the analysis.
5. Provide oracle-routing ablations to quantify the value of the classifier.
6. Include qualitative failure cases for each chart family.
7. Demonstrate that the same schema supports all target chart types without special-case output formats.

## 5. Publication Readiness Estimate

At the design level, the project is publishable if the implementation validates three claims:

- multi-chart generalization
- modular interpretability
- schema-consistent structured extraction

Without those results, the system risks reading as a well-organized engineering effort rather than a research contribution.

