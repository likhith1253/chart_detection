from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .geometry import MarkerGeometry, DetectedPoint


@dataclass
class PointDetector:
    min_area: int = 8
    max_area: int = 500
    metadata: dict[str, Any] = field(default_factory=dict)

    def detect(self, image: Image.Image) -> list[MarkerGeometry]:
        arr = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        markers: list[MarkerGeometry] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < self.min_area or area > self.max_area:
                continue
            cx, cy = self._subpixel_centroid(contour)
            shape = self._shape_name(contour, w, h)
            markers.append(
                MarkerGeometry(
                    bbox=(x, y, x + w, y + h),
                    centroid=(cx, cy),
                    shape=shape,
                    confidence=min(0.99, area / float(self.max_area)),
                )
            )
        return sorted(markers, key=lambda m: (m.bbox[0], m.bbox[1]))

    def _subpixel_centroid(self, contour) -> tuple[float, float]:
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])
        x, y, w, h = cv2.boundingRect(contour)
        return x + w / 2.0, y + h / 2.0

    def _shape_name(self, contour, w: int, h: int) -> str:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        if len(approx) <= 4:
            return "square" if abs(w - h) < max(2, min(w, h) * 0.3) else "diamond"
        if len(approx) > 7:
            return "circle"
        return "marker"

