# Implementation Roadmap

## Milestone 1: Benchmark Finalization

Objective:
- Finalize dataset inventory, splits, and leakage controls.

Deliverables:
- benchmark manifest
- statistics
- leakage reports
- dataset inventory

Expected output files:
- `data/datasets/benchmark_manifest.csv`
- `data/datasets/benchmark_statistics.json`
- `data/datasets/dataset_inventory.json`
- `data/datasets/benchmark_leakage_report.json`

Dependencies:
- Existing public datasets

Approximate complexity:
- Medium

Paper section supported:
- Experiments
- Dataset

## Milestone 2: Document-to-Chart Localization Design

Objective:
- Define how chart regions are isolated from full pages.

Deliverables:
- detector protocol
- chart crop metadata contract

Expected output files:
- detector evaluation report

Dependencies:
- Benchmark images

Approximate complexity:
- High

Paper section supported:
- Architecture
- Methodology

## Milestone 3: Multi-Chart Schema Design

Objective:
- Fix the universal JSON schema and its validation rules.

Deliverables:
- JSON schema specification
- validation constraints

Expected output files:
- schema document
- schema tests later in implementation

Dependencies:
- Parser design

Approximate complexity:
- Medium

Paper section supported:
- Methodology

## Milestone 4: Parser Routing Specification

Objective:
- Define routing logic from class prediction to parser selection.

Deliverables:
- routing policy
- fallback policy

Expected output files:
- routing matrix

Dependencies:
- Classifier outputs

Approximate complexity:
- Medium

Paper section supported:
- Architecture
- Methodology

## Milestone 5: Chart-Specific Parser Designs

Objective:
- Specify bar, line, scatter, pie, histogram, and box plot parsers.

Deliverables:
- parser contracts
- per-chart failure analysis

Expected output files:
- parser specification documents

Dependencies:
- Router and schema

Approximate complexity:
- High

Paper section supported:
- Methodology
- Experiments

## Milestone 6: Consistency and Validation Layer

Objective:
- Add semantic validation over extracted entities.

Deliverables:
- consistency rules
- invalid-output taxonomy

Expected output files:
- validation report

Dependencies:
- Parsers

Approximate complexity:
- Medium

Paper section supported:
- Methodology
- Discussion

## Milestone 7: Evaluation Protocol

Objective:
- Lock down metrics and benchmarks.

Deliverables:
- cross-dataset evaluation plan
- ablation matrix

Expected output files:
- experiment tables and plots

Dependencies:
- Benchmark and schema

Approximate complexity:
- Medium

Paper section supported:
- Experiments

## Milestone 8: Paper-Ready Analysis

Objective:
- Produce figures, tables, and narrative claims.

Deliverables:
- architecture diagrams
- tables of results
- failure-case gallery

Expected output files:
- paper draft assets

Dependencies:
- All prior milestones

Approximate complexity:
- High

Paper section supported:
- Full paper

