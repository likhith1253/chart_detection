from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass
class CircleDetectionResult:
    center: tuple[int, int]
    radius: int
    inner_radius: int = 0
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CircleDetector:
    dp: float = 1.2
    min_dist: int = 30
    param1: int = 100
    param2: int = 30
    metadata: dict[str, Any] = field(default_factory=dict)

    def detect(self, image: Image.Image) -> CircleDetectionResult:
        arr = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        blur = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, self.dp, self.min_dist, param1=self.param1, param2=self.param2, minRadius=10)
        if circles is not None and len(circles[0]) > 0:
            x, y, r = circles[0][0]
            return CircleDetectionResult(center=(int(x), int(y)), radius=int(r), confidence=0.9, metadata={"method": "hough"})
        h, w = gray.shape[:2]
        return CircleDetectionResult(center=(w // 2, h // 2), radius=int(min(w, h) * 0.35), confidence=0.35, metadata={"method": "fallback"})

