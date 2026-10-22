from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.parsers.bar.calibration import AxisCalibration

from .geometry import DetectedPoint, DetectedSeries


@dataclass
class ValueExtractor:
    metadata: dict[str, Any] = field(default_factory=dict)

    def estimate_points(
        self,
        series: list[DetectedSeries],
        calibration: AxisCalibration | None = None,
        axis_scale: str = "linear",
        orientation: str = "vertical",
    ) -> list[list[dict[str, float]]]:
        calibration = calibration or AxisCalibration(0.0, 0.0, 1.0, 1.0, 1, 1, scale_type=axis_scale)
        payload: list[list[dict[str, float]]] = []
        for item in series:
            points: list[dict[str, float]] = []
            for p in item.points:
                xv = calibration.pixel_to_value(float(p.x), axis="x")
                yv = calibration.pixel_to_value(float(p.y), axis="y")
                points.append({"x": round(xv, 4), "y": round(yv, 4), "confidence": float(p.confidence)})
            payload.append(points)
        return payload

