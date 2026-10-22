from __future__ import annotations

from pathlib import Path

import yaml
from PIL import Image

from src.detection import DetectorTrainingConfig, YoloTrainingBridge
from src.detection.manifest.doclaynet import load_yolo_manifest


def _write_image(path: Path, size: tuple[int, int] = (64, 48)) -> None:
    Image.new("RGB", size, color="white").save(path)


def test_doclaynet_yolo_manifest_is_loaded(tmp_path) -> None:
    root = tmp_path / "doclaynet_yolo"
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        _write_image(root / "images" / split / "page_1.png")
        (root / "labels" / split / "page_1.txt").write_text("0 0.500000 0.500000 0.250000 0.250000\n", encoding="utf-8")
    (root / "dataset.yaml").write_text(
        yaml.safe_dump({"path": str(root), "train": "images/train", "val": "images/train", "test": "images/train", "names": {0: "chart"}}),
        encoding="utf-8",
    )

    manifest, samples, annotations = load_yolo_manifest(root / "dataset.yaml")
    assert manifest.dataset_name == "doclaynet_yolo"
    assert len(samples) == 1
    assert len(annotations) == 1
    assert annotations[0].label == "chart"


def test_training_bridge_auto_recognizes_doclaynet_yaml(tmp_path) -> None:
    root = tmp_path / "doclaynet_yolo"
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        _write_image(root / "images" / split / "page_1.png")
        (root / "labels" / split / "page_1.txt").write_text("0 0.500000 0.500000 0.250000 0.250000\n", encoding="utf-8")
    (root / "dataset.yaml").write_text(
        yaml.safe_dump({"path": str(root), "train": "images/train", "val": "images/train", "test": "images/train", "names": {0: "chart"}}),
        encoding="utf-8",
    )

    class StubTrainer:
        def train(self, **kwargs):
            return {"trained": True}

    bridge = YoloTrainingBridge(
        DetectorTrainingConfig(
            manifest_csv=root / "dataset.yaml",
            output_dir=tmp_path / "out",
            epochs=1,
            batch_size=1,
            model_name="yolov11n.pt",
        ),
        trainer_factory=lambda cfg: StubTrainer(),
    )
    state = bridge.train()
    assert state.best_checkpoint is not None and state.best_checkpoint.exists()
    assert (tmp_path / "out" / "metrics.json").exists()
