from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .geometry import CurveGeometry, DetectedPoint, DetectedSeries


@dataclass
class SeriesDetector:
    metadata: dict[str, Any] = field(default_factory=dict)

    def detect(self, curves: list[CurveGeometry], points: list[DetectedPoint]) -> list[DetectedSeries]:
        series: list[DetectedSeries] = []
        for idx, curve in enumerate(curves):
            curve_points = [
                DetectedPoint(x=float(x), y=float(y), confidence=curve.confidence, marker="sampled", series_id=idx)
                for x, y in curve.points
            ]
            if points:
                curve_points.extend(DetectedPoint(x=p.x, y=p.y, confidence=p.confidence, marker=p.marker, series_id=idx) for p in points[: max(1, len(points) // max(1, len(curves)))])
            series.append(DetectedSeries(series_id=idx, color=None, line_type=curve.curve_type, points=curve_points, confidence=curve.confidence, metadata={"bbox": curve.bbox}))
        if not series and points:
            series.append(DetectedSeries(series_id=0, color=None, line_type="polyline", points=list(points), confidence=0.5))
        return series

