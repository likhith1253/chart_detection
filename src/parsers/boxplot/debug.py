from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from .geometry import DetectedBox, OutlierPoint


def create_debug_overlay(image: Image.Image, boxes: Iterable[DetectedBox], outliers: Iterable[OutlierPoint], output_path: str | Path | None = None) -> Image.Image | Path:
    arr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    for idx, box in enumerate(boxes):
        g = box.geometry
        cv2.rectangle(arr, (g.x1, g.y1), (g.x2, g.y2), (0, 0, 255), 2)
        cv2.putText(arr, f"B{idx}", (g.x1, max(0, g.y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        cv2.line(arr, (g.x1, int(box.quartiles.lower_quartile)), (g.x2, int(box.quartiles.lower_quartile)), (0, 255, 255), 1)
        cv2.line(arr, (g.x1, int(box.quartiles.median)), (g.x2, int(box.quartiles.median)), (0, 255, 0), 1)
        cv2.line(arr, (g.x1, int(box.quartiles.upper_quartile)), (g.x2, int(box.quartiles.upper_quartile)), (255, 0, 255), 1)
    for idx, outlier in enumerate(outliers):
        cv2.circle(arr, (int(outlier.x), int(outlier.y)), 3, (0, 128, 255), -1)
        cv2.putText(arr, f"O{idx}", (int(outlier.x), int(outlier.y)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    out = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    if output_path is None:
        return out
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
    return path

