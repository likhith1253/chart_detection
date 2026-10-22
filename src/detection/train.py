from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .config import DetectionConfig
from .dataset import DetectionDataset, load_detection_dataset
from .model import AbstractDetector, DetectorBackendError, create_detector


@dataclass
class TrainingConfig:
    data_csv: Path
    output_dir: Path = Path("results") / "detection"
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-3
    seed: int = 42
    backend: str = "yolo"
    augmentations: list[str] = field(default_factory=list)
    checkpoint_path: Path | None = None
    resume: bool = False


@dataclass
class TrainingState:
    epoch: int = 0
    best_metric: float = 0.0
    last_checkpoint: Path | None = None


class DetectionTrainer:
    def __init__(
        self,
        config: TrainingConfig,
        detector: AbstractDetector | None = None,
        augmentation_hook: Callable[[Any], Any] | None = None,
        evaluation_hook: Callable[[DetectionDataset], Mapping[str, float]] | None = None,
    ):
        self.config = config
        self.detector = detector or create_detector(DetectionConfig(backend=config.backend))
        self.augmentation_hook = augmentation_hook
        self.evaluation_hook = evaluation_hook
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset(self) -> DetectionDataset:
        return load_detection_dataset(self.config.data_csv)

    def apply_augmentations(self, image):
        if self.augmentation_hook is None:
            return image
        return self.augmentation_hook(image)

    def save_checkpoint(self, state: TrainingState) -> Path:
        payload = {
            "epoch": state.epoch,
            "best_metric": state.best_metric,
            "backend": self.config.backend,
        }
        path = self.checkpoint_dir / f"checkpoint_epoch_{state.epoch:04d}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        state.last_checkpoint = path
        return path

    def load_checkpoint(self, path: str | Path) -> TrainingState:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return TrainingState(
            epoch=int(data.get("epoch", 0)),
            best_metric=float(data.get("best_metric", 0.0)),
            last_checkpoint=Path(path),
        )

    def train(self) -> TrainingState:
        dataset = self.load_dataset()
        state = TrainingState()
        if self.config.resume and self.config.checkpoint_path is not None and Path(self.config.checkpoint_path).exists():
            state = self.load_checkpoint(self.config.checkpoint_path)
        if self.evaluation_hook is not None:
            metrics = self.evaluation_hook(dataset)
            state.best_metric = float(metrics.get("mAP", metrics.get("precision", 0.0)))
        self.save_checkpoint(state)
        return state

