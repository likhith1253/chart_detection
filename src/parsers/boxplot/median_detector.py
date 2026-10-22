from __future__ import annotations

from .geometry import DetectedBox


class MedianDetector:
    def detect(self, box: DetectedBox) -> float:
        return box.quartiles.median

