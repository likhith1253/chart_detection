from __future__ import annotations

from math import cos, radians, sin
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from .circle_detector import CircleDetectionResult
from .geometry import DetectedSlice


def create_debug_overlay(
    image: Image.Image,
    slices: Iterable[DetectedSlice],
    circle: CircleDetectionResult | None = None,
    labels: list[dict[str, object]] | None = None,
    output_path: str | Path | None = None,
) -> Image.Image | Path:
    arr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    if circle is not None:
        cv2.circle(arr, circle.center, circle.radius, (0, 255, 255), 2)
        if circle.inner_radius:
            cv2.circle(arr, circle.center, circle.inner_radius, (255, 0, 255), 2)
    cx, cy = circle.center if circle is not None else (arr.shape[1] // 2, arr.shape[0] // 2)
    for idx, slice_ in enumerate(slices):
        g = slice_.geometry
        for ang in (g.start_angle, g.end_angle):
            x2 = int(cx + g.radius * cos(radians(ang)))
            y2 = int(cy + g.radius * sin(radians(ang)))
            cv2.line(arr, (cx, cy), (x2, y2), (0, 0, 255), 2)
        mid = radians((g.start_angle + g.end_angle) / 2.0)
        tx = int(cx + (g.radius * 0.6) * cos(mid))
        ty = int(cy + (g.radius * 0.6) * sin(mid))
        cv2.putText(arr, f"S{idx}", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        cv2.putText(arr, f"{g.sweep_angle:.1f}", (tx, ty + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    if labels:
        for idx, item in enumerate(labels[: len(list(slices))]):
            pass
    out = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    if output_path is None:
        return out
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
    return path

