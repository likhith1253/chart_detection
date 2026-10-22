from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from .axis_detector import AxisInfo
from .geometry import CurveGeometry, DetectedPoint, DetectedSeries


def create_debug_overlay(
    image: Image.Image,
    curves: Iterable[CurveGeometry],
    points: Iterable[DetectedPoint],
    series: Iterable[DetectedSeries],
    axis_info: AxisInfo | None = None,
    output_path: str | Path | None = None,
) -> Image.Image | Path:
    arr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    if axis_info is not None:
        x1, y1, x2, y2 = axis_info.plot_area
        cv2.rectangle(arr, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.circle(arr, axis_info.origin, 4, (255, 0, 0), -1)
    for idx, curve in enumerate(curves):
        for a, b in zip(curve.points, curve.points[1:]):
            cv2.line(arr, a, b, (0, 0, 255), 2)
        if curve.points:
            cv2.putText(arr, f"C{idx}", curve.points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    for idx, point in enumerate(points):
        cv2.circle(arr, (int(point.x), int(point.y)), 3, (0, 255, 0), -1)
        cv2.putText(arr, f"P{idx}", (int(point.x), int(point.y)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    for s in series:
        if s.points:
            cv2.putText(arr, f"S{s.series_id}", (int(s.points[0].x), int(s.points[0].y)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
    out = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    if output_path is None:
        return out
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
    return path

