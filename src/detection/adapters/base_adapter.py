from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def is_valid(self) -> bool:
        return self.x2 > self.x1 and self.y2 > self.y1

    def has_negative_coordinates(self) -> bool:
        return self.x1 < 0 or self.y1 < 0 or self.x2 < 0 or self.y2 < 0

    def to_list(self) -> list[float]:
        return [float(self.x1), float(self.y1), float(self.x2), float(self.y2)]

    def as_int_list(self) -> list[int]:
        return [int(round(v)) for v in self.to_list()]


@dataclass(frozen=True)
class DetectionAnnotation:
    annotation_id: str
    image_id: str
    bbox: BoundingBox
    label: str = "chart"
    confidence: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectionSample:
    image_id: str
    image_path: Path
    split: str = "train"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectionDatasetManifest:
    dataset_name: str
    root_dir: Path
    image_dir: Path
    annotation_dir: Path | None
    annotation_format: str
    split: str = "train"
    image_count: int | None = None
    annotation_count: int | None = None
    classes: tuple[str, ...] = ("chart",)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetRecord:
    sample: DetectionSample
    annotations: tuple[DetectionAnnotation, ...]


class DatasetAdapter(ABC):
    format_name: str

    @abstractmethod
    def load_manifest(self, root_dir: str | Path, split: str = "train", **kwargs: Any) -> DetectionDatasetManifest:
        raise NotImplementedError

    @abstractmethod
    def iter_records(self, manifest: DetectionDatasetManifest) -> Iterable[DatasetRecord]:
        raise NotImplementedError

    @abstractmethod
    def export_records(self, manifest: DetectionDatasetManifest, records: Iterable[DatasetRecord], output_path: str | Path) -> Path:
        raise NotImplementedError

