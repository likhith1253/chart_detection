from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable

from src.detection.adapters.base_adapter import DetectionAnnotation, DetectionDatasetManifest, DetectionSample


@dataclass(frozen=True)
class ManifestValidationReport:
    valid: bool
    missing_images: list[str] = field(default_factory=list)
    duplicate_image_ids: list[str] = field(default_factory=list)
    duplicate_paths: list[str] = field(default_factory=list)
    invalid_labels: list[str] = field(default_factory=list)
    invalid_splits: list[str] = field(default_factory=list)
    missing_metadata: list[str] = field(default_factory=list)

    def to_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return out


def validate_manifest(
    manifest: DetectionDatasetManifest,
    samples: Iterable[DetectionSample],
    annotations: Iterable[DetectionAnnotation],
) -> ManifestValidationReport:
    allowed_splits = {"train", "validation", "test"}
    samples = list(samples)
    annotations = list(annotations)
    missing_images: list[str] = []
    duplicate_image_ids: list[str] = []
    duplicate_paths: list[str] = []
    invalid_labels: list[str] = []
    invalid_splits: list[str] = []
    missing_metadata: list[str] = []

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    labels = {c.lower() for c in manifest.classes}
    ann_by_image = {ann.image_id: ann for ann in annotations}

    for sample in samples:
        if sample.image_id in seen_ids:
            duplicate_image_ids.append(sample.image_id)
        seen_ids.add(sample.image_id)

        path_str = str(sample.image_path)
        if path_str in seen_paths:
            duplicate_paths.append(path_str)
        seen_paths.add(path_str)

        if not sample.image_path.exists():
            missing_images.append(sample.image_id)

        if sample.split not in allowed_splits:
            invalid_splits.append(sample.image_id)

        if "dataset" not in sample.metadata or "chart_type" not in sample.metadata or "width" not in sample.metadata or "height" not in sample.metadata:
            missing_metadata.append(sample.image_id)

        ann = ann_by_image.get(sample.image_id)
        if ann is not None and ann.label.lower() not in labels:
            invalid_labels.append(ann.annotation_id)

    return ManifestValidationReport(
        valid=not any([missing_images, duplicate_image_ids, duplicate_paths, invalid_labels, invalid_splits, missing_metadata]),
        missing_images=missing_images,
        duplicate_image_ids=duplicate_image_ids,
        duplicate_paths=duplicate_paths,
        invalid_labels=invalid_labels,
        invalid_splits=invalid_splits,
        missing_metadata=missing_metadata,
    )

