from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from PIL import Image

from .config import DetectionConfig
from .model import AbstractDetector, DetectionPrediction, create_detector
from .utils import load_image


class DetectionPredictor:
    def __init__(self, detector: AbstractDetector | None = None, config: DetectionConfig | None = None):
        self.config = config or DetectionConfig()
        self.detector = detector or create_detector(self.config)

    def predict(self, image: Image.Image | str | Path) -> list[dict[str, Any]]:
        preds = self.detector.predict(image)
        return [
            {"bbox": pred.bbox, "confidence": float(pred.confidence), "class": pred.class_name}
            for pred in preds
        ]

    def crop_predictions(self, image: Image.Image | str | Path, min_confidence: float = 0.0) -> list[Image.Image]:
        return self.detector.crop_chart_images(image, min_confidence=min_confidence)


def predict_chart_regions(image: Image.Image | str | Path, detector: AbstractDetector | None = None, config: DetectionConfig | None = None) -> list[dict[str, Any]]:
    return DetectionPredictor(detector=detector, config=config).predict(image)

