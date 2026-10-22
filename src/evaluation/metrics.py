from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float
    support: int = 0
    details: dict[str, Any] | None = None


def parsing_success_rate(successes: list[bool]) -> MetricResult:
    return MetricResult("parsing_success_rate", float(np.mean(successes)) if successes else 0.0, len(successes))


def geometry_iou(pred_boxes: list[dict[str, float]], true_boxes: list[dict[str, float]]) -> MetricResult:
    if not pred_boxes or not true_boxes:
        return MetricResult("geometry_iou", 0.0, max(len(pred_boxes), len(true_boxes)))
    scores = []
    for p, t in zip(pred_boxes, true_boxes):
        scores.append(_iou(p, t))
    return MetricResult("geometry_iou", float(np.mean(scores)), len(scores))


def point_accuracy(pred_points: list[dict[str, float]], true_points: list[dict[str, float]], tolerance: float = 5.0) -> MetricResult:
    if not pred_points or not true_points:
        return MetricResult("point_accuracy", 0.0, max(len(pred_points), len(true_points)))
    matches = 0
    total = min(len(pred_points), len(true_points))
    for p, t in zip(pred_points, true_points):
        if abs(p["x"] - t["x"]) <= tolerance and abs(p["y"] - t["y"]) <= tolerance:
            matches += 1
    return MetricResult("point_accuracy", matches / total if total else 0.0, total)


def value_mae(pred_values: list[float], true_values: list[float]) -> MetricResult:
    if not pred_values or not true_values:
        return MetricResult("value_mae", 0.0, max(len(pred_values), len(true_values)))
    n = min(len(pred_values), len(true_values))
    return MetricResult("value_mae", float(np.mean([abs(pred_values[i] - true_values[i]) for i in range(n)])), n)


def value_rmse(pred_values: list[float], true_values: list[float]) -> MetricResult:
    if not pred_values or not true_values:
        return MetricResult("value_rmse", 0.0, max(len(pred_values), len(true_values)))
    n = min(len(pred_values), len(true_values))
    return MetricResult("value_rmse", sqrt(float(np.mean([(pred_values[i] - true_values[i]) ** 2 for i in range(n)]))), n)


def category_accuracy(pred_categories: list[str], true_categories: list[str]) -> MetricResult:
    if not pred_categories or not true_categories:
        return MetricResult("category_accuracy", 0.0, max(len(pred_categories), len(true_categories)))
    n = min(len(pred_categories), len(true_categories))
    return MetricResult("category_accuracy", sum(str(pred_categories[i]) == str(true_categories[i]) for i in range(n)) / n, n)


def series_accuracy(pred_series: list[dict[str, Any]], true_series: list[dict[str, Any]]) -> MetricResult:
    if not pred_series or not true_series:
        return MetricResult("series_accuracy", 0.0, max(len(pred_series), len(true_series)))
    n = min(len(pred_series), len(true_series))
    correct = 0
    for p, t in zip(pred_series, true_series):
        if p.get("series_id") == t.get("series_id") and p.get("label") == t.get("label"):
            correct += 1
    return MetricResult("series_accuracy", correct / n if n else 0.0, n)


def ocr_accuracy(pred_text: list[str], true_text: list[str]) -> MetricResult:
    if not pred_text or not true_text:
        return MetricResult("ocr_accuracy", 0.0, max(len(pred_text), len(true_text)))
    n = min(len(pred_text), len(true_text))
    return MetricResult("ocr_accuracy", sum(str(pred_text[i]).strip() == str(true_text[i]).strip() for i in range(n)) / n, n)


def runtime_metric(seconds: float) -> MetricResult:
    return MetricResult("runtime", float(seconds), 1)


def memory_metric(megabytes: float) -> MetricResult:
    return MetricResult("memory", float(megabytes), 1)


def _iou(a: dict[str, float], b: dict[str, float]) -> float:
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom else 0.0

