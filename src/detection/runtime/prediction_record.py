from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class PredictionRecord:
    bbox: list[int]
    confidence: float
    class_name: str
    backend_name: str
    image_id: str
    crop: Image.Image | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectionPrediction(PredictionRecord):
    pass

