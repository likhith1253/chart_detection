from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .prediction_record import DetectionPrediction


def _iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def apply_nms(predictions: Iterable[DetectionPrediction], iou_threshold: float = 0.5) -> list[DetectionPrediction]:
    grouped: dict[tuple[str, str], list[DetectionPrediction]] = {}
    for pred in predictions:
        backend_name = getattr(pred, "backend_name", "unknown")
        grouped.setdefault((backend_name, pred.class_name), []).append(pred)

    kept: list[DetectionPrediction] = []
    for _, group in grouped.items():
        group = sorted(group, key=lambda p: p.confidence, reverse=True)
        while group:
            best = group.pop(0)
            kept.append(best)
            group = [pred for pred in group if _iou(best.bbox, pred.bbox) < iou_threshold]
    return kept


def build_detection_predictions(
    raw_predictions: Iterable[object],
    backend_name: str,
    image_id: str,
    metadata: dict | None = None,
) -> list[DetectionPrediction]:
    metadata = dict(metadata or {})
    out: list[DetectionPrediction] = []
    for pred in raw_predictions:
        bbox = list(getattr(pred, "bbox"))
        out.append(
            DetectionPrediction(
                bbox=[int(v) for v in bbox],
                confidence=float(getattr(pred, "confidence")),
                class_name=str(getattr(pred, "class_name", "chart")),
                backend_name=backend_name,
                image_id=image_id,
                crop=getattr(pred, "crop", None),
                metadata={**metadata, **dict(getattr(pred, "metadata", {}))},
            )
        )
    return out
