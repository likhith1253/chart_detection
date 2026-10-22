from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.parsers.bar.calibration import AxisCalibration

from .geometry import DetectedBox


@dataclass(frozen=True)
class BoxValues:
    lower_whisker: float
    lower_quartile: float
    median: float
    upper_quartile: float
    upper_whisker: float
    outliers: list[float]


class ValueExtractor:
    def estimate(self, box: DetectedBox, calibration: AxisCalibration | None = None, orientation: str = "vertical", axis_scale: str = "linear") -> BoxValues:
        calibration = calibration or AxisCalibration(0.0, 0.0, 1.0, 1.0, 1, 1, scale_type=axis_scale)
        if orientation == "horizontal":
            lower = calibration.pixel_to_value(box.whiskers.lower, axis="x")
            q1 = calibration.pixel_to_value(box.quartiles.lower_quartile, axis="x")
            med = calibration.pixel_to_value(box.quartiles.median, axis="x")
            q3 = calibration.pixel_to_value(box.quartiles.upper_quartile, axis="x")
            upper = calibration.pixel_to_value(box.whiskers.upper, axis="x")
            outs = [calibration.pixel_to_value(o.x, axis="x") for o in box.outliers]
        else:
            lower = calibration.pixel_to_value(box.whiskers.lower, axis="y")
            q1 = calibration.pixel_to_value(box.quartiles.lower_quartile, axis="y")
            med = calibration.pixel_to_value(box.quartiles.median, axis="y")
            q3 = calibration.pixel_to_value(box.quartiles.upper_quartile, axis="y")
            upper = calibration.pixel_to_value(box.whiskers.upper, axis="y")
            outs = [calibration.pixel_to_value(o.y, axis="y") for o in box.outliers]
        return BoxValues(round(lower, 4), round(q1, 4), round(med, 4), round(q3, 4), round(upper, 4), [round(v, 4) for v in outs])

