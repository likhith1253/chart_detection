from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

from src.detection.manifest import load_manifest
from src.detection.model import AbstractDetector, create_detector
from src.detection.adapters.base_adapter import DetectionDatasetManifest, DetectionSample

from .cropper import ChartCrop, crop_chart_regions
from .postprocess import apply_nms, build_detection_predictions
from .prediction_record import DetectionPrediction


@dataclass
class DetectionRunner:
    detector: AbstractDetector
    backend_name: str = "yolo"
    nms_iou_threshold: float = 0.5

    @classmethod
    def from_backend(cls, detector: AbstractDetector | None = None, backend_name: str = "yolo") -> "DetectionRunner":
        return cls(detector=detector or create_detector(), backend_name=backend_name)

    def _predict_image(self, image: Image.Image | str | Path, image_id: str, save_crops_dir: str | Path | None = None) -> tuple[list[DetectionPrediction], list[ChartCrop]]:
        raw_preds = self.detector.predict(image)
        preds = build_detection_predictions(raw_preds, backend_name=self.backend_name, image_id=image_id)
        preds = apply_nms(preds, iou_threshold=self.nms_iou_threshold)
        crops = crop_chart_regions(image, preds, save_dir=save_crops_dir)
        enriched: list[DetectionPrediction] = []
        for pred, crop in zip(preds, crops):
            enriched.append(
                DetectionPrediction(
                    bbox=pred.bbox,
                    confidence=pred.confidence,
                    class_name=pred.class_name,
                    backend_name=pred.backend_name,
                    image_id=pred.image_id,
                    crop=crop.crop_image,
                    metadata={**pred.metadata, "crop_path": crop.transform_information.get("saved_path")},
                )
            )
        return enriched, crops

    def run_on_image(self, image: Image.Image | str | Path, image_id: str | None = None, save_crops_dir: str | Path | None = None) -> tuple[list[DetectionPrediction], list[ChartCrop]]:
        image_id = image_id or (Path(image).stem if isinstance(image, (str, Path)) else "image")
        return self._predict_image(image, image_id=image_id, save_crops_dir=save_crops_dir)

    def run_on_folder(self, folder: str | Path, save_crops_dir: str | Path | None = None) -> tuple[list[DetectionPrediction], list[ChartCrop]]:
        folder_path = Path(folder)
        all_preds: list[DetectionPrediction] = []
        all_crops: list[ChartCrop] = []
        for image_path in sorted(folder_path.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
                preds, crops = self.run_on_image(image_path, image_id=image_path.stem, save_crops_dir=save_crops_dir)
                all_preds.extend(preds)
                all_crops.extend(crops)
        return all_preds, all_crops

    def run_on_manifest(self, split: str = "train", save_crops_dir: str | Path | None = None, manifest_csv: str | Path | None = None) -> tuple[list[DetectionPrediction], list[ChartCrop]]:
        _, samples, _ = load_manifest(split=split, manifest_csv=manifest_csv)
        all_preds: list[DetectionPrediction] = []
        all_crops: list[ChartCrop] = []
        for sample in samples:
            preds, crops = self.run_on_image(sample.image_path, image_id=sample.image_id, save_crops_dir=save_crops_dir)
            all_preds.extend(preds)
            all_crops.extend(crops)
        return all_preds, all_crops

