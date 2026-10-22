from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from .axis_detector import AxisInfo
from .geometry import BarGeometry


def create_debug_overlay(
    image: Image.Image,
    bars: Iterable[BarGeometry],
    axis_info: AxisInfo | None = None,
    output_path: str | Path | None = None,
) -> Image.Image | Path:
    arr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    if axis_info is not None:
        x1, y1, x2, y2 = axis_info.plot_area
        cv2.rectangle(arr, (x1, y1), (x2, y2), (0, 255, 255), 2)
        ox, oy = axis_info.origin
        cv2.circle(arr, (ox, oy), 4, (255, 0, 0), -1)
    for idx, bar in enumerate(bars):
        cv2.rectangle(arr, (bar.x1, bar.y1), (bar.x2, bar.y2), (0, 0, 255), 2)
        cx, cy = map(int, bar.centroid)
        cv2.circle(arr, (cx, cy), 3, (0, 255, 0), -1)
        cv2.putText(arr, str(idx), (bar.x1, max(0, bar.y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    out = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    if output_path is None:
        return out
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
    return path

