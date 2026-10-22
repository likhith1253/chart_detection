from __future__ import annotations

from .geometry import DetectedBox, WhiskerGeometry


class WhiskerDetector:
    def detect(self, box: DetectedBox) -> WhiskerGeometry:
        return box.whiskers

