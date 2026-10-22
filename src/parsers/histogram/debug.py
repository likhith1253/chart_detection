from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from .axis_detector import AxisInfo
from .geometry import DetectedBin, DistributionStatistics


def create_debug_overlay(
    image: Image.Image,
    bins: Iterable[DetectedBin],
    distribution: DistributionStatistics | None = None,
    axis_info: AxisInfo | None = None,
    output_path: str | Path | None = None,
) -> Image.Image | Path:
    arr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    if axis_info is not None:
        x1, y1, x2, y2 = axis_info.plot_area
        cv2.rectangle(arr, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.circle(arr, axis_info.origin, 4, (255, 0, 0), -1)
    for idx, bin_ in enumerate(bins):
        g = bin_.geometry
        cv2.rectangle(arr, (g.x1, g.y1), (g.x2, g.y2), (0, 0, 255), 2)
        cx, cy = map(int, g.centroid)
        cv2.circle(arr, (cx, cy), 3, (0, 255, 0), -1)
        cv2.putText(arr, f"B{idx}", (g.x1, max(0, g.y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    if distribution is not None:
        cv2.putText(arr, f"cont={distribution.continuity:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    out = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    if output_path is None:
        return out
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
    return path

