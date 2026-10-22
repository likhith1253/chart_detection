from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ground_truth import GroundTruthInstance
from .metrics import (
    category_accuracy,
    geometry_iou,
    ocr_accuracy,
    parsing_success_rate,
    point_accuracy,
    series_accuracy,
    value_mae,
    value_rmse,
)


@dataclass(frozen=True)
class ComparisonResult:
    chart_type: str
    parser_name: str
    metrics: dict[str, float]
    details: dict[str, Any]


class GroundTruthComparator:
    def compare(self, parser_name: str, chart_type: str, prediction: dict[str, Any], ground_truth: GroundTruthInstance) -> ComparisonResult:
        metrics = {
            "parsing_success_rate": parsing_success_rate([bool(prediction.get("success", False))]).value,
            "geometry_iou": geometry_iou(prediction.get("boxes", []), ground_truth.boxes).value,
            "point_accuracy": point_accuracy(prediction.get("points", []), ground_truth.points).value,
            "value_mae": value_mae(prediction.get("values", []), ground_truth.values).value,
            "value_rmse": value_rmse(prediction.get("values", []), ground_truth.values).value,
            "category_accuracy": category_accuracy(prediction.get("categories", []), ground_truth.categories).value,
            "series_accuracy": series_accuracy(prediction.get("series", []), ground_truth.series).value,
            "ocr_accuracy": ocr_accuracy(prediction.get("ocr_text", []), ground_truth.ocr_text).value,
        }
        return ComparisonResult(chart_type, parser_name, metrics, {"chart_id": ground_truth.chart_id, "metadata": ground_truth.metadata})

