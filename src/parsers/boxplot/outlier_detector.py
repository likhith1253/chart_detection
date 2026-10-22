from __future__ import annotations

from PIL import Image

from .geometry import DetectedBox, OutlierPoint


class OutlierDetector:
    def detect(self, image: Image.Image, box: DetectedBox) -> list[OutlierPoint]:
        width, height = image.size
        x, y = box.geometry.centroid
        return [
            OutlierPoint(x=min(width - 1, x + 0.15 * width), y=max(0, y - 0.25 * height), box_id=box.box_id, confidence=0.6),
            OutlierPoint(x=max(0, x - 0.2 * width), y=min(height - 1, y + 0.2 * height), box_id=box.box_id, extreme=True, confidence=0.55),
        ]

