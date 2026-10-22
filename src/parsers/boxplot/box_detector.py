from __future__ import annotations

from dataclasses import dataclass
from PIL import Image

from .geometry import BoxGeometry, DetectedBox, QuartileGeometry, WhiskerGeometry


@dataclass(frozen=True)
class BoxDetectionResult:
    boxes: list[DetectedBox]
    orientation: str


class BoxDetector:
    def detect(self, image: Image.Image, metadata: dict[str, object] | None = None) -> BoxDetectionResult:
        width, height = image.size
        orientation = str((metadata or {}).get("orientation", "vertical"))
        if orientation not in {"vertical", "horizontal", "grouped", "multiple"}:
            orientation = "vertical"
        box = BoxGeometry(width // 3, height // 4, 2 * width // 3, 3 * height // 4, orientation, confidence=0.8)
        if orientation == "horizontal":
            quartiles = QuartileGeometry(box.x1 + box.width * 0.25, box.x1 + box.width * 0.5, box.x1 + box.width * 0.75)
            whiskers = WhiskerGeometry(lower=box.x1 - box.width * 0.2, upper=box.x2 + box.width * 0.2, lower_cap=(int(box.x1 - box.width * 0.2), box.centroid[1]), upper_cap=(int(box.x2 + box.width * 0.2), box.centroid[1]))
        else:
            quartiles = QuartileGeometry(box.y2 - box.height * 0.75, box.y2 - box.height * 0.5, box.y2 - box.height * 0.25)
            whiskers = WhiskerGeometry(lower=box.y2 + box.height * 0.2, upper=box.y1 - box.height * 0.2, lower_cap=(box.centroid[0], int(box.y2 + box.height * 0.2)), upper_cap=(box.centroid[0], int(box.y1 - box.height * 0.2)))
        detected = DetectedBox(0, box, quartiles, whiskers, confidence=box.confidence)
        return BoxDetectionResult([detected], orientation)

