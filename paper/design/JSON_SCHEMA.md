# Universal JSON Schema

## 1. Design Objectives

The schema must represent all supported chart types without creating type-specific ad hoc formats. It should:

- preserve raw geometry
- preserve semantic interpretation
- encode uncertainty
- support multiple series
- support multiple chart regions per document
- allow direct conversion to tabular form

## 2. Core Schema

```json
{
  "schema_version": "1.0",
  "document": {
    "document_id": "string",
    "source_uri": "string",
    "page_index": 0,
    "page_size": {
      "width": 0,
      "height": 0
    }
  },
  "charts": [
    {
      "chart_id": "string",
      "chart_type": "bar|line|scatter|pie|histogram|box_plot",
      "confidence": 0.0,
      "bbox": {
        "x0": 0,
        "y0": 0,
        "x1": 0,
        "y1": 0
      },
      "coordinate_system": {
        "x_axis": {
          "type": "categorical|numeric|temporal|log",
          "label": "string",
          "ticks": [],
          "transform": "string"
        },
        "y_axis": {
          "type": "categorical|numeric|temporal|log",
          "label": "string",
          "ticks": [],
          "transform": "string"
        }
      },
      "legend": {
        "present": true,
        "entries": []
      },
      "text": {
        "title": "string",
        "subtitle": "string",
        "annotations": [],
        "ocr_tokens": []
      },
      "series": [],
      "marks": [],
      "statistics": {},
      "provenance": {
        "detector": "string",
        "classifier": "string",
        "parser": "string",
        "ocr": "string",
        "axis_solver": "string",
        "legend_solver": "string"
      },
      "quality": {
        "warnings": [],
        "violations": []
      }
    }
  ]
}
```

## 3. Shared Entities

### 3.1 Series

```json
{
  "series_id": "string",
  "label": "string",
  "color": "string",
  "marker": "string",
  "confidence": 0.0
}
```

### 3.2 Mark

```json
{
  "mark_id": "string",
  "mark_type": "bar|point|line_segment|slice|bin|box|whisker|outlier",
  "series_id": "string",
  "geometry": {},
  "value": {
    "x": null,
    "y": null,
    "label": "string",
    "count": null,
    "proportion": null
  },
  "confidence": 0.0,
  "evidence": {
    "bbox": {},
    "token_ids": [],
    "pixel_refs": []
  }
}
```

## 4. Chart-Specific Encoding

### 4.1 Bar Chart

Required mark fields:

- category label
- numeric value
- bar geometry
- orientation
- stacking/grouping flag

### 4.2 Line Chart

Required mark fields:

- ordered points
- line segment geometry
- optional marker geometry

### 4.3 Scatter Plot

Required mark fields:

- point coordinates
- point style
- optional series assignment

### 4.4 Pie Chart

Required mark fields:

- slice angle span
- slice proportion
- slice label

### 4.5 Histogram

Required mark fields:

- bin lower edge
- bin upper edge
- bin count or height
- bin width

### 4.6 Box Plot

Required mark fields:

- minimum
- q1
- median
- q3
- maximum
- outliers

## 5. Validation Rules

- A bar chart must have at least one bar mark.
- A pie chart slice proportions should sum to approximately 1.0 or 100 depending on unit normalization.
- A histogram bin sequence should be ordered by bin edge.
- A box plot must satisfy min <= q1 <= median <= q3 <= max under standard conventions.
- A line chart must preserve point order along the x axis.
- Scatter points must carry coordinates even if series labels are missing.

## 6. Why This Schema Is Universal

The schema is universal because:

- it separates shared chart metadata from chart-specific marks
- it supports uncertainty and provenance
- it can represent dense or sparse geometries
- it can be serialized to JSON, CSV, or relational tables

This makes it suitable for benchmark evaluation and for future parser development.

