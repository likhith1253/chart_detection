from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .geometry import CurveGeometry


@dataclass
class CurveDetector:
    edge_low: int = 40
    edge_high: int = 120
    morph_kernel: tuple[int, int] = (3, 3)
    metadata: dict[str, Any] = field(default_factory=dict)

    def detect_edges(self, image: Image.Image) -> np.ndarray:
        arr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(arr, (3, 3), 0)
        edges = cv2.Canny(blurred, self.edge_low, self.edge_high)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.morph_kernel)
        return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    def detect(self, image: Image.Image) -> list[CurveGeometry]:
        edges = self.detect_edges(image)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        curves: list[CurveGeometry] = []
        for contour in contours:
            if len(contour) < 6:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < 8 or h < 8:
                continue
            pts = [(int(p[0][0]), int(p[0][1])) for p in contour[:: max(1, len(contour) // 40)]]
            ar = w / float(h + 1e-6)
            curve_type = "polyline"
            if len(pts) > 20 and cv2.arcLength(contour, False) > 0:
                curve_type = "smooth" if ar > 0.7 else "polyline"
            curves.append(CurveGeometry(points=pts, bbox=(x, y, x + w, y + h), curve_type=curve_type, confidence=min(0.99, len(pts) / 50.0)))
        return sorted(curves, key=lambda c: (c.bbox[0], c.bbox[1]))

