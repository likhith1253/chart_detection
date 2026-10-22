from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class YoloBackendConfig:
    model_name: str = "yolov8n.pt"
    image_size: int = 640
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "cpu"
    classes: tuple[str, ...] = ("chart",)


@dataclass(frozen=True)
class DetectionConfig:
    backend: str = "yolo"
    data_root: Path = Path("data")
    benchmark_root: Path = Path("data") / "benchmark"
    output_root: Path = Path("results") / "detection"
    image_size: int = 640
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    num_workers: int = 0
    seed: int = 42
    backend_config: YoloBackendConfig = field(default_factory=YoloBackendConfig)


def _to_path(value: Any, default: Path) -> Path:
    if value is None:
        return default
    return Path(value)


def _load_mapping(data: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(data or {})


def load_detection_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> DetectionConfig:
    raw: dict[str, Any] = {}
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
            if not isinstance(loaded, dict):
                raise TypeError("Detection config file must contain a mapping")
            raw.update(loaded)
    raw.update(_load_mapping(overrides))

    backend = str(raw.get("backend", "yolo")).lower()
    backend_cfg = _load_mapping(raw.get("backend_config"))
    yolo = YoloBackendConfig(
        model_name=str(backend_cfg.get("model_name", "yolov8n.pt")),
        image_size=int(backend_cfg.get("image_size", raw.get("image_size", 640))),
        confidence_threshold=float(backend_cfg.get("confidence_threshold", raw.get("confidence_threshold", 0.25))),
        iou_threshold=float(backend_cfg.get("iou_threshold", raw.get("iou_threshold", 0.45))),
        device=str(backend_cfg.get("device", "cpu")),
        classes=tuple(backend_cfg.get("classes", ("chart",))),
    )
    return DetectionConfig(
        backend=backend,
        data_root=_to_path(raw.get("data_root"), Path("data")),
        benchmark_root=_to_path(raw.get("benchmark_root"), Path("data") / "benchmark"),
        output_root=_to_path(raw.get("output_root"), Path("results") / "detection"),
        image_size=int(raw.get("image_size", yolo.image_size)),
        confidence_threshold=float(raw.get("confidence_threshold", yolo.confidence_threshold)),
        iou_threshold=float(raw.get("iou_threshold", yolo.iou_threshold)),
        num_workers=int(raw.get("num_workers", 0)),
        seed=int(raw.get("seed", 42)),
        backend_config=yolo,
    )

