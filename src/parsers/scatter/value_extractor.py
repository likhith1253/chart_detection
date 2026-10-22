from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.parsers.bar.calibration import AxisCalibration

from .geometry import DetectedPoint, ScatterSeries


@dataclass
class ValueExtractor:
    metadata: dict[str, Any] = field(default_factory=dict)

    def estimate_points(
        self,
        series: list[ScatterSeries],
        calibration: AxisCalibration | None = None,
        axis_scale: str = "linear",
    ) -> list[list[dict[str, float]]]:
        calibration = calibration or AxisCalibration(0.0, 0.0, 1.0, 1.0, 1, 1, scale_type=axis_scale)
        payload: list[list[dict[str, float]]] = []
        for item in series:
            points: list[dict[str, float]] = []
            for p in item.points:
                points.append(
                    {
                        "x": round(calibration.pixel_to_value(float(p.x), axis="x"), 4),
                        "y": round(calibration.pixel_to_value(float(p.y), axis="y"), 4),
                        "confidence": float(p.confidence),
                    }
                )
            payload.append(points)
        return payload

