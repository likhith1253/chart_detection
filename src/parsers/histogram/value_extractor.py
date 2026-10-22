from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.parsers.bar.calibration import AxisCalibration

from .geometry import DetectedBin, HistogramDistribution


@dataclass
class ValueExtractor:
    metadata: dict[str, Any] = field(default_factory=dict)

    def estimate_bins(
        self,
        bins: list[DetectedBin],
        calibration: AxisCalibration | None = None,
        axis_scale: str = "linear",
    ) -> list[dict[str, float]]:
        calibration = calibration or AxisCalibration(0.0, 0.0, 1.0, 1.0, 1, 1, scale_type=axis_scale)
        payload: list[dict[str, float]] = []
        for idx, bin_ in enumerate(bins):
            g = bin_.geometry
            payload.append(
                {
                    "bin_index": float(idx),
                    "x_min": round(calibration.pixel_to_value(float(g.x1), axis="x"), 4),
                    "x_max": round(calibration.pixel_to_value(float(g.x2), axis="x"), 4),
                    "frequency": round(max(0.0, calibration.pixel_to_value(float(g.y1), axis="y")), 4),
                }
            )
        return payload

