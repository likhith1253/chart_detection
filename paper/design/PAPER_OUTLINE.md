# Publishable Paper Outline

## Title

Unified Chart Structure Recovery from Document Images via Class-Aware Parser Routing and Universal JSON Representation

## Abstract

We present a modular framework for converting chart-bearing document images into structured machine-readable representations across multiple chart families. Unlike histogram-only extraction systems, our approach jointly addresses chart localization, chart classification, parser routing, OCR, axis understanding, legend interpretation, and consistency-checked structure recovery. A universal JSON schema enables cross-dataset evaluation, downstream reuse, and systematic error analysis. We show that a chart-family-aware architecture improves generalization and interpretability while preserving practical runtime efficiency.

## 1. Introduction

- Problem motivation
- Why document charts matter
- Why structured output matters
- Why a multi-chart framework is needed
- Contributions

## 2. Related Work

- Chart detection
- Chart classification
- Chart question answering
- Histogram extraction
- OCR and document understanding
- Structured information extraction

## 3. Methodology

### 3.1 Overview

### 3.2 Document Preprocessing

### 3.3 Chart Detection

### 3.4 Chart Classification

### 3.5 Parser Routing

### 3.6 Chart-Specific Parsers

### 3.7 OCR, Axis, and Legend Understanding

### 3.8 Consistency Validation

### 3.9 Universal JSON Schema

## 4. Architecture

- Overall pipeline
- Module interfaces
- Data flow
- Parser specialization

## 5. Experiments

- Datasets
- Metrics
- Baselines
- Ablations
- Cross-dataset transfer
- Runtime

## 6. Results

- Detection
- Classification
- Extraction
- Generalization
- Runtime

## 7. Discussion

- Why routing helps
- Where the system fails
- What chart families remain hard

## 8. Limitations

- OCR dependency
- Ambiguous visual encodings
- Multi-chart pages
- Dataset bias

## 9. Future Work

- Joint learning of routing and parsing
- More chart families
- PDF-native extraction
- Human-in-the-loop correction

## 10. Conclusion

- Restate contribution
- Restate value for document AI and chart understanding

