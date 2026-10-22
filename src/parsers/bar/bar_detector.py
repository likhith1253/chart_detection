from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .geometry import BarGeometry


@dataclass(frozen=True)
class DetectedBar:
    geometry: BarGeometry
    score: float
    series_index: int = 0


@dataclass(frozen=True)
class DetectionMaskResult:
    threshold: np.ndarray
    edges: np.ndarray
    mask: np.ndarray
    components: list[dict]


class BarDetector:
    def __init__(self, min_area: int = 60) -> None:
        self.min_area = min_area

    def _preprocess(self, image: Image.Image) -> DetectionMaskResult:
        arr = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 11
        )
        edges = cv2.Canny(blur, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
        components = []
        for idx in range(1, num_labels):
            x, y, w, h, area = stats[idx]
            components.append({"bbox": (x, y, x + w, y + h), "area": int(area), "centroid": tuple(centroids[idx])})
        return DetectionMaskResult(threshold=thresh, edges=edges, mask=closed, components=components)

    def _contour_boxes(self, mask: np.ndarray) -> list[dict]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[dict] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < self.min_area:
                continue
            boxes.append({"bbox": (x, y, x + w, y + h), "area": area, "contour": contour})
        return boxes

    def _geometric_filter(self, bbox: tuple[int, int, int, int], image_size: tuple[int, int]) -> bool:
        x1, y1, x2, y2 = bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        img_w, img_h = image_size
        if width * height < self.min_area:
            return False
        if width > img_w * 0.95 and height > img_h * 0.95:
            return False
        return True

    def _orientation_confidence(self, bbox: tuple[int, int, int, int]) -> tuple[str, float]:
        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        if h >= w:
            return "vertical", min(1.0, h / (w + h))
        return "horizontal", min(1.0, w / (w + h))

    def detect(self, image: Image.Image) -> list[DetectedBar]:
        mask_result = self._preprocess(image)
        candidates = self._contour_boxes(mask_result.mask)
        candidates.extend({"bbox": comp["bbox"], "area": comp["area"], "centroid": comp["centroid"]} for comp in mask_result.components)
        img_w, img_h = image.size
        bars: list[DetectedBar] = []
        for idx, candidate in enumerate(candidates):
            bbox = candidate["bbox"]
            if not self._geometric_filter(bbox, (img_w, img_h)):
                continue
            orientation, score = self._orientation_confidence(bbox)
            x1, y1, x2, y2 = bbox
            cx = float(candidate.get("centroid", ((x1 + x2) / 2.0, (y1 + y2) / 2.0))[0])
            cy = float(candidate.get("centroid", ((x1 + x2) / 2.0, (y1 + y2) / 2.0))[1])
            bars.append(
                DetectedBar(
                    geometry=BarGeometry(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        orientation=orientation,
                        centroid_x=cx,
                        centroid_y=cy,
                        series_id=None,
                        stack_id=None,
                        confidence=score,
                        metadata={"component_index": idx, "source": "opencv"},
                    ),
                    score=score,
                )
            )
        if not bars:
            width, height = image.size
            bars.append(DetectedBar(BarGeometry(int(width * 0.1), int(height * 0.2), int(width * 0.2), int(height * 0.8), "vertical", confidence=0.05, metadata={"fallback": True}), 0.05))
        return self._deduplicate(bars)

    def _deduplicate(self, bars: list[DetectedBar]) -> list[DetectedBar]:
        unique: list[DetectedBar] = []
        seen: set[tuple[int, int, int, int]] = set()
        for bar in bars:
            bbox = (bar.geometry.x1, bar.geometry.y1, bar.geometry.x2, bar.geometry.y2)
            if bbox in seen:
                continue
            seen.add(bbox)
            unique.append(bar)
        return sorted(unique, key=lambda b: (b.geometry.x1, b.geometry.y1))

