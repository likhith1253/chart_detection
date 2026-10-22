from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .geometry import BinGeometry, DetectedBin


@dataclass
class BinDetector:
    min_area: int = 40
    min_width: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)

    def detect(self, image: Image.Image) -> list[DetectedBin]:
        arr = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 11)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        refined = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bins: list[DetectedBin] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < self.min_area or w < self.min_width:
                continue
            if self._touches_axes(x, y, w, h, image.size):
                score = min(0.99, area / 5000.0 + 0.2)
            else:
                score = min(0.95, area / 5000.0)
            geom = BinGeometry(
                x1=x,
                y1=y,
                x2=x + w,
                y2=y + h,
                width=w,
                height=h,
                area=area,
                centroid=(x + w / 2.0, y + h / 2.0),
                confidence=score,
                metadata={"source": "opencv", "contour_points": len(contour)},
            )
            bins.append(DetectedBin(geometry=geom, score=score))
        bins = self._merge_touching_bins(sorted(bins, key=lambda b: (b.geometry.x1, b.geometry.y1)))
        bins = self._split_overwide_bins(bins)
        return bins

    def _touches_axes(self, x: int, y: int, w: int, h: int, image_size: tuple[int, int]) -> bool:
        img_w, img_h = image_size
        return x <= img_w * 0.15 or y + h >= img_h * 0.8

    def _merge_touching_bins(self, bins: list[DetectedBin]) -> list[DetectedBin]:
        if not bins:
            return []
        merged: list[DetectedBin] = [bins[0]]
        for current in bins[1:]:
            prev = merged[-1]
            if self._overlaps_or_touches(prev.geometry, current.geometry):
                x1 = min(prev.geometry.x1, current.geometry.x1)
                y1 = min(prev.geometry.y1, current.geometry.y1)
                x2 = max(prev.geometry.x2, current.geometry.x2)
                y2 = max(prev.geometry.y2, current.geometry.y2)
                w = x2 - x1
                h = y2 - y1
                area = w * h
                centroid = (x1 + w / 2.0, y1 + h / 2.0)
                geom = BinGeometry(x1, y1, x2, y2, w, h, area, centroid, confidence=max(prev.geometry.confidence, current.geometry.confidence), metadata={"merged": True})
                merged[-1] = DetectedBin(geometry=geom, score=max(prev.score, current.score), merged_from=[*(prev.merged_from or [prev.geometry.bin_index or 0]), *(current.merged_from or [current.geometry.bin_index or 0])])
            else:
                merged.append(current)
        return merged

    def _split_overwide_bins(self, bins: list[DetectedBin]) -> list[DetectedBin]:
        result: list[DetectedBin] = []
        for item in bins:
            geom = item.geometry
            if geom.width > geom.height * 2.5 and geom.width > 20:
                mid = geom.x1 + geom.width // 2
                left = BinGeometry(geom.x1, geom.y1, mid, geom.y2, mid - geom.x1, geom.height, (mid - geom.x1) * geom.height, (geom.x1 + mid) / 2.0, geom.centroid[1], confidence=geom.confidence * 0.9, metadata={"split": "left"})
                right = BinGeometry(mid, geom.y1, geom.x2, geom.y2, geom.x2 - mid, geom.height, (geom.x2 - mid) * geom.height, (mid + geom.x2) / 2.0, geom.centroid[1], confidence=geom.confidence * 0.9, metadata={"split": "right"})
                result.extend([DetectedBin(left, item.score * 0.9, merged_from=item.merged_from), DetectedBin(right, item.score * 0.9, merged_from=item.merged_from)])
            else:
                result.append(item)
        return result

    def _overlaps_or_touches(self, a: BinGeometry, b: BinGeometry) -> bool:
        return not (a.x2 < b.x1 - 2 or b.x2 < a.x1 - 2)

