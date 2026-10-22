from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image

from .config import DetectionConfig, YoloBackendConfig
from .utils import BoundingBox, crop_image, load_image


class DetectorBackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class DetectionPrediction:
    bbox: list[int]
    confidence: float
    class_name: str = "chart"


@dataclass(frozen=True)
class DetectionResult:
    image_path: Path
    predictions: list[DetectionPrediction]


class AbstractDetector(ABC):
    @abstractmethod
    def predict(self, image: Image.Image | str | Path) -> list[DetectionPrediction]:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    def predict_path(self, image_path: str | Path) -> list[DetectionPrediction]:
        return self.predict(load_image(image_path))

    def crop_chart_images(self, image: Image.Image | str | Path, min_confidence: float = 0.0) -> list[Image.Image]:
        if not isinstance(image, Image.Image):
            image = load_image(image)
        crops: list[Image.Image] = []
        for pred in self.predict(image):
            if pred.confidence < min_confidence:
                continue
            crops.append(crop_image(image, pred.bbox))
        return crops


class YoloDetector(AbstractDetector):
    def __init__(self, config: YoloBackendConfig | None = None):
        self.config = config or YoloBackendConfig()
        self._model = None

    def load(self) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise DetectorBackendError("Ultralytics is required for the YOLO backend") from exc
        self._model = YOLO(self.config.model_name)

    def predict(self, image: Image.Image | str | Path) -> list[DetectionPrediction]:
        if self._model is None:
            try:
                self.load()
            except DetectorBackendError:
                return []
        img = load_image(image) if not isinstance(image, Image.Image) else image
        results = self._model.predict(
            source=img,
            imgsz=self.config.image_size,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            device=self.config.device,
            verbose=False,
        )
        predictions: list[DetectionPrediction] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                xyxy = [int(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0].item()) if hasattr(box.conf[0], "item") else float(box.conf[0])
                cls_name = self.config.classes[0] if self.config.classes else "chart"
                predictions.append(DetectionPrediction(bbox=xyxy, confidence=conf, class_name=cls_name))
        return predictions


def create_detector(config: DetectionConfig | None = None) -> AbstractDetector:
    cfg = config or DetectionConfig()
    backend = cfg.backend.lower()
    if backend == "yolo":
        return YoloDetector(cfg.backend_config)
    raise DetectorBackendError(f"Unsupported detector backend: {cfg.backend}")

