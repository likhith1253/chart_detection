from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .geometry import DetectedPoint


@dataclass
class PointDetector:
    metadata: dict[str, Any] = field(default_factory=dict)

    def detect(self, image: Image.Image) -> list[DetectedPoint]:
        arr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        mask = cv2.inRange(sat, 40, 255)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points: list[DetectedPoint] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < 6 or area > 200:
                continue
            cx = x + w / 2.0
            cy = y + h / 2.0
            points.append(DetectedPoint(x=cx, y=cy, confidence=min(0.99, area / 100.0), marker="explicit"))
        return points

