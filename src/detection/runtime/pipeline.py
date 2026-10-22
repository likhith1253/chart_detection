from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .runner import DetectionRunner
from .prediction_record import DetectionPrediction
from .cropper import ChartCrop


@dataclass
class DetectionPipeline:
    runner: DetectionRunner

    def run_image(self, image: Image.Image | str | Path, image_id: str | None = None, save_crops_dir: str | Path | None = None) -> tuple[list[DetectionPrediction], list[ChartCrop]]:
        return self.runner.run_on_image(image, image_id=image_id, save_crops_dir=save_crops_dir)

    def run_folder(self, folder: str | Path, save_crops_dir: str | Path | None = None) -> tuple[list[DetectionPrediction], list[ChartCrop]]:
        return self.runner.run_on_folder(folder, save_crops_dir=save_crops_dir)

    def run_manifest(self, split: str = "train", save_crops_dir: str | Path | None = None, manifest_csv: str | Path | None = None) -> tuple[list[DetectionPrediction], list[ChartCrop]]:
        return self.runner.run_on_manifest(split=split, save_crops_dir=save_crops_dir, manifest_csv=manifest_csv)

