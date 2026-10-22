from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

from .base_adapter import DetectionAnnotation, DetectionDatasetManifest, DetectionSample, DatasetRecord


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    missing_images: list[str] = field(default_factory=list)
    missing_annotations: list[str] = field(default_factory=list)
    duplicate_ids: list[str] = field(default_factory=list)
    duplicate_boxes: list[str] = field(default_factory=list)
    invalid_labels: list[str] = field(default_factory=list)
    negative_coordinates: list[str] = field(default_factory=list)
    invalid_bounding_boxes: list[str] = field(default_factory=list)
    total_records: int = 0

    def to_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return out


class DatasetValidator:
    def validate(self, manifest: DetectionDatasetManifest, records: Iterable[DatasetRecord]) -> ValidationReport:
        missing_images: list[str] = []
        missing_annotations: list[str] = []
        duplicate_ids: list[str] = []
        duplicate_boxes: list[str] = []
        invalid_labels: list[str] = []
        negative_coordinates: list[str] = []
        invalid_bboxes: list[str] = []

        seen_ids: set[str] = set()
        seen_boxes: set[tuple[str, tuple[float, float, float, float], str]] = set()
        total = 0
        allowed_labels = {label.lower() for label in manifest.classes}

        for record in records:
            total += 1
            sample = record.sample
            if not sample.image_path.exists():
                missing_images.append(sample.image_id)
            if not record.annotations:
                missing_annotations.append(sample.image_id)
            if sample.image_id in seen_ids:
                duplicate_ids.append(sample.image_id)
            seen_ids.add(sample.image_id)
            for ann in record.annotations:
                key = (ann.image_id, tuple(ann.bbox.to_list()), ann.label.lower())
                if key in seen_boxes:
                    duplicate_boxes.append(ann.annotation_id)
                seen_boxes.add(key)
                if ann.label.lower() not in allowed_labels:
                    invalid_labels.append(ann.annotation_id)
                if ann.bbox.has_negative_coordinates():
                    negative_coordinates.append(ann.annotation_id)
                if not ann.bbox.is_valid():
                    invalid_bboxes.append(ann.annotation_id)

        return ValidationReport(
            valid=not any([missing_images, missing_annotations, duplicate_ids, duplicate_boxes, invalid_labels, negative_coordinates, invalid_bboxes]),
            missing_images=missing_images,
            missing_annotations=missing_annotations,
            duplicate_ids=duplicate_ids,
            duplicate_boxes=duplicate_boxes,
            invalid_labels=invalid_labels,
            negative_coordinates=negative_coordinates,
            invalid_bounding_boxes=invalid_bboxes,
            total_records=total,
        )

