from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, degrees, hypot
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .geometry import DetectedSlice, SliceGeometry
from .circle_detector import CircleDetectionResult


@dataclass
class SliceDetector:
    metadata: dict[str, Any] = field(default_factory=dict)

    def detect(self, image: Image.Image, circle: CircleDetectionResult) -> list[DetectedSlice]:
        arr = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        slices: list[DetectedSlice] = []
        for idx, contour in enumerate(contours):
            if len(contour) < 10:
                continue
            pts = contour.reshape(-1, 2)
            cx, cy = circle.center
            angles = [degrees(atan2(float(y - cy), float(x - cx))) % 360 for x, y in pts]
            if not angles:
                continue
            start = min(angles)
            end = max(angles)
            sweep = self._sweep(start, end)
            if sweep < 5:
                continue
            centroid = self._centroid(pts)
            expl = hypot(centroid[0] - cx, centroid[1] - cy) > circle.radius * 0.08
            geom = SliceGeometry(
                start_angle=start,
                end_angle=end,
                sweep_angle=sweep,
                radius=float(circle.radius),
                inner_radius=float(circle.inner_radius),
                centroid=centroid,
                confidence=min(0.99, sweep / 360.0),
                slice_id=idx,
                metadata={"exploded": expl},
            )
            slices.append(DetectedSlice(geometry=geom, score=geom.confidence, exploded=expl))
        if not slices:
            # fallback: a single full-pie slice
            geom = SliceGeometry(0.0, 360.0, 360.0, float(circle.radius), float(circle.inner_radius), circle.center, confidence=0.2, slice_id=0, metadata={"fallback": True})
            slices.append(DetectedSlice(geometry=geom, score=0.2, exploded=False))
        return self._merge_adjacent(sorted(slices, key=lambda s: s.geometry.start_angle))

    def _sweep(self, start: float, end: float) -> float:
        sweep = end - start
        return sweep if sweep >= 0 else sweep + 360.0

    def _centroid(self, pts: np.ndarray) -> tuple[float, float]:
        if len(pts) == 0:
            return 0.0, 0.0
        return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))

    def _merge_adjacent(self, slices: list[DetectedSlice]) -> list[DetectedSlice]:
        if len(slices) < 2:
            return slices
        merged: list[DetectedSlice] = [slices[0]]
        for current in slices[1:]:
            prev = merged[-1]
            if current.geometry.start_angle <= prev.geometry.end_angle + 2.0:
                geom = SliceGeometry(
                    start_angle=prev.geometry.start_angle,
                    end_angle=max(prev.geometry.end_angle, current.geometry.end_angle),
                    sweep_angle=max(prev.geometry.end_angle, current.geometry.end_angle) - prev.geometry.start_angle,
                    radius=max(prev.geometry.radius, current.geometry.radius),
                    inner_radius=min(prev.geometry.inner_radius, current.geometry.inner_radius),
                    centroid=prev.geometry.centroid,
                    confidence=max(prev.score, current.score),
                    slice_id=prev.geometry.slice_id,
                    metadata={"merged": True},
                )
                merged[-1] = DetectedSlice(geometry=geom, score=geom.confidence, exploded=prev.exploded or current.exploded)
            else:
                merged.append(current)
        return merged

