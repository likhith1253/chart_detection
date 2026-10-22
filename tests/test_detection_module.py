from __future__ import annotations

from pathlib import Path
import csv

import pandas as pd
import pytest
from PIL import Image

from src.detection.config import DetectionConfig, YoloBackendConfig, load_detection_config
from src.detection.dataset import DetectionDataset, DetectionSample, load_detection_dataset
from src.detection.evaluate import DetectionEvaluator
from src.detection.model import AbstractDetector, DetectionPrediction, DetectorBackendError, create_detector
from src.detection.predict import DetectionPredictor, predict_chart_regions
from src.detection.train import DetectionTrainer, TrainingConfig
from src.detection.utils import BoundingBox, crop_image, image_size


class StubDetector(AbstractDetector):
    def load(self) -> None:
        return None

    def predict(self, image):
        return [DetectionPrediction(bbox=[1, 2, 9, 10], confidence=0.98, class_name="chart")]


def _write_image(path: Path) -> None:
    Image.new("RGB", (12, 14), color="white").save(path)


def test_load_detection_config_defaults(tmp_path) -> None:
    cfg = load_detection_config(overrides={"output_root": str(tmp_path / "out"), "backend_config": {"device": "cpu"}})
    assert cfg.backend == "yolo"
    assert cfg.output_root == tmp_path / "out"
    assert cfg.backend_config.device == "cpu"


def test_detection_dataset_loading_and_cropping(tmp_path) -> None:
    img_path = tmp_path / "page.png"
    _write_image(img_path)
    csv_path = tmp_path / "dataset.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_id", "image_path", "bbox", "class", "split", "source"])
        writer.writeheader()
        writer.writerow(
            {
                "image_id": "img1",
                "image_path": str(img_path),
                "bbox": "1,2,9,10",
                "class": "chart",
                "split": "train",
                "source": "benchmark",
            }
        )

    ds = load_detection_dataset(csv_path)
    assert len(ds) == 1

    sample = ds.samples[0]
    assert sample.class_name == "chart"
    cropped = ds.crop(0)
    assert cropped.size[0] > 0
    assert cropped.size[1] > 0


def test_predictor_returns_expected_api(tmp_path) -> None:
    img_path = tmp_path / "page.png"
    _write_image(img_path)
    predictor = DetectionPredictor(detector=StubDetector(), config=DetectionConfig())
    preds = predictor.predict(img_path)
    assert preds == [{"bbox": [1, 2, 9, 10], "confidence": 0.98, "class": "chart"}]
    assert len(predict_chart_regions(img_path, detector=StubDetector())) == 1


def test_crop_image_and_image_size(tmp_path) -> None:
    img_path = tmp_path / "page.png"
    _write_image(img_path)
    assert image_size(img_path) == (12, 14)
    image = Image.open(img_path)
    cropped = crop_image(image, [2, 3, 8, 11])
    assert cropped.size == (6, 8)


def test_trainer_saves_checkpoint_without_training(tmp_path) -> None:
    img_path = tmp_path / "page.png"
    _write_image(img_path)
    csv_path = tmp_path / "dataset.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_id", "image_path", "bbox", "class", "split", "source"])
        writer.writeheader()
        writer.writerow(
            {
                "image_id": "img1",
                "image_path": str(img_path),
                "bbox": "1,2,9,10",
                "class": "chart",
                "split": "train",
                "source": "benchmark",
            }
        )
    trainer = DetectionTrainer(TrainingConfig(data_csv=csv_path, output_dir=tmp_path / "out"))
    state = trainer.train()
    assert state.last_checkpoint is not None
    assert state.last_checkpoint.exists()


def test_create_detector_uses_yolo_backend() -> None:
    detector = create_detector(DetectionConfig(backend="yolo", backend_config=YoloBackendConfig()))
    assert detector is not None
