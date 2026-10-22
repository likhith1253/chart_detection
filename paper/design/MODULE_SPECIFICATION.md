# Module Specification

## 1. Document Preprocessing

Purpose:
- Normalize page-level artifacts before chart localization.

Inputs:
- Full document image

Outputs:
- Preprocessed document image
- Page quality metadata

Algorithms:
- Orientation correction
- Contrast normalization
- Denoising
- Optional page segmentation

Possible alternatives:
- Classical image processing
- Lightweight learned restoration
- OCR-driven page cleanup

Evaluation metric:
- Chart-detection recall after preprocessing
- Page rotation correction accuracy

Failure modes:
- Over-smoothing chart text
- Removing thin axes or gridlines
- Failing on low-resolution scans

Dependencies:
- None beyond image I/O

Expected runtime:
- Low to moderate per page

## 2. Chart Region Detection

Purpose:
- Localize charts on a page.

Inputs:
- Preprocessed document image

Outputs:
- Bounding boxes or polygons for chart regions

Algorithms:
- Object detection
- Region proposal plus non-maximum suppression
- Layout-aware detection

Possible alternatives:
- DETR-style detectors
- YOLO-family detectors
- Layout parsing with chart-specific heads

Evaluation metric:
- mAP
- Recall at IoU thresholds

Failure modes:
- Missing small inset charts
- Confusing tables with charts
- Merging adjacent figures

Dependencies:
- Document preprocessing

Expected runtime:
- Moderate

## 3. Chart Classification

Purpose:
- Assign a chart family label.

Inputs:
- Cropped chart image

Outputs:
- Chart class probabilities

Algorithms:
- CNN classifier
- Hierarchical class grouping
- Confidence calibration

Possible alternatives:
- ViT-based classifier
- Multi-task classifier jointly predicting class and visual attributes

Evaluation metric:
- Accuracy
- Macro-F1
- Calibration error

Failure modes:
- Confusing bar vs histogram
- Confusing line vs scatter
- Misclassifying box plots as bar charts under clutter

Dependencies:
- Cropped chart image

Expected runtime:
- Low

## 4. Parser Router

Purpose:
- Select the most appropriate structural parser.

Inputs:
- Class probabilities
- Heuristic cues
- Optional OCR/layout signals

Outputs:
- Parser selection decision

Algorithms:
- Confidence thresholding
- Rule-based fallback
- Multi-parser arbitration

Possible alternatives:
- Learned router
- Mixture-of-experts gate

Evaluation metric:
- Routing accuracy
- Extraction improvement over oracle and baseline routing

Failure modes:
- Overconfident misrouting
- Ambiguity in mixed chart styles

Dependencies:
- Classification output

Expected runtime:
- Negligible

## 5. OCR Module

Purpose:
- Read textual elements inside the chart.

Inputs:
- Chart crop or subregions

Outputs:
- OCR tokens with coordinates and confidence

Algorithms:
- Text detection
- Text recognition
- Token aggregation

Possible alternatives:
- End-to-end OCR
- Layout-conditioned OCR

Evaluation metric:
- Character error rate
- Word error rate
- Text localization F1

Failure modes:
- Rotated text
- Small tick labels
- Low-contrast legends

Dependencies:
- Chart crop

Expected runtime:
- Moderate to high

## 6. Axis Understanding

Purpose:
- Infer the data coordinate system.

Inputs:
- OCR tokens
- Chart geometry

Outputs:
- Axis type, scale, ticks, and mapping functions

Algorithms:
- Tick extraction
- Regression from pixel to data coordinates
- Axis orientation inference

Possible alternatives:
- Symbolic axis parsing
- Neural keypoint regression

Evaluation metric:
- Tick localization accuracy
- Axis mapping error

Failure modes:
- Log scales mistaken for linear
- Category order ambiguity
- Missing tick labels

Dependencies:
- OCR
- Chart geometry

Expected runtime:
- Moderate

## 7. Legend Understanding

Purpose:
- Associate series encodings with labels.

Inputs:
- OCR tokens
- Color/shape clues
- Chart geometry

Outputs:
- Legend entries mapped to series identifiers

Algorithms:
- Spatial matching
- Color clustering
- Semantic label matching

Possible alternatives:
- Graph matching over legend and mark candidates
- Joint OCR-layout transformer

Evaluation metric:
- Legend assignment accuracy

Failure modes:
- Legends outside crop
- Ambiguous colors
- Duplicate series names

Dependencies:
- OCR
- Chart image analysis

Expected runtime:
- Moderate

## 8. Bar Parser

Purpose:
- Extract bar geometry and values.

Inputs:
- Chart crop
- Axis semantics
- Legend semantics

Outputs:
- Bar entities

Algorithms:
- Bar detection
- Baseline estimation
- Category mapping

Possible alternatives:
- Instance segmentation
- Keypoint-based bar recovery

Evaluation metric:
- Bar count accuracy
- Value MAE

Failure modes:
- Stacked bars merged incorrectly
- Overlapping bars

Dependencies:
- OCR, axis understanding

Expected runtime:
- Moderate

## 9. Line Parser

Purpose:
- Recover ordered series trajectories.

Inputs:
- Chart crop
- Axis semantics
- Legend semantics

Outputs:
- Ordered line series

Algorithms:
- Polyline tracing
- Marker detection
- Gap handling

Possible alternatives:
- Keypoint sequence decoding
- Graph-based tracing

Evaluation metric:
- Point MAE
- Sequence alignment score

Failure modes:
- Dense overlapping lines
- Missing markers

Dependencies:
- OCR, axis understanding

Expected runtime:
- Moderate

## 10. Scatter Parser

Purpose:
- Recover point clouds and series membership.

Inputs:
- Chart crop
- Axis semantics
- Legend semantics

Outputs:
- Scatter point entities

Algorithms:
- Point detection
- Clustering
- Symbol-to-series assignment

Possible alternatives:
- Instance segmentation
- Density estimation

Evaluation metric:
- Point localization F1
- Series assignment accuracy

Failure modes:
- Dense point overlap
- Tiny markers

Dependencies:
- OCR, axis understanding

Expected runtime:
- Moderate to high

## 11. Pie Parser

Purpose:
- Recover slice proportions and labels.

Inputs:
- Chart crop
- OCR tokens

Outputs:
- Pie slice entities

Algorithms:
- Boundary angle estimation
- Center localization
- Label association

Possible alternatives:
- Polar geometry decoding

Evaluation metric:
- Slice boundary error
- Proportion MAE

Failure modes:
- 3D or exploded pies
- External labels with leader lines

Dependencies:
- OCR

Expected runtime:
- Low to moderate

## 12. Histogram Parser

Purpose:
- Recover bins and distribution statistics.

Inputs:
- Chart crop
- Axis semantics

Outputs:
- Histogram bin entities

Algorithms:
- Bin edge detection
- Height estimation
- Normalization inference

Possible alternatives:
- Bar-like parsing with distribution constraints

Evaluation metric:
- Bin edge error
- Count MAE

Failure modes:
- Variable-width bins
- Mixed orientation

Dependencies:
- OCR, axis understanding

Expected runtime:
- Moderate

## 13. Box Plot Parser

Purpose:
- Recover distribution summaries from box plots.

Inputs:
- Chart crop
- Axis semantics

Outputs:
- Box, whisker, and outlier entities

Algorithms:
- Quartile landmark detection
- Whisker end estimation
- Outlier detection

Possible alternatives:
- Landmark regression

Evaluation metric:
- Quartile MAE
- Outlier detection F1

Failure modes:
- Horizontal box plots
- Notched boxes

Dependencies:
- OCR, axis understanding

Expected runtime:
- Moderate

## 14. Consistency Checker

Purpose:
- Validate extracted structure.

Inputs:
- Parsed entities

Outputs:
- Validated entities
- Error flags

Algorithms:
- Constraint checking
- Range validation
- Cross-field reconciliation

Possible alternatives:
- Probabilistic graphical model over extracted entities

Evaluation metric:
- Constraint satisfaction rate

Failure modes:
- Rejecting valid noisy data
- Allowing semantically wrong but visually plausible outputs

Dependencies:
- All parser outputs

Expected runtime:
- Low

