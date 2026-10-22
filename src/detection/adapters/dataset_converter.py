from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .annotation_parser import AnnotationParser
from .base_adapter import DetectionAnnotation, DetectionDatasetManifest, DetectionSample, DatasetRecord


class DatasetConverter:
    def __init__(self, parser: AnnotationParser | None = None):
        self.parser = parser or AnnotationParser()

    def to_internal_records(
        self,
        manifest: DetectionDatasetManifest,
        annotation_path: str | Path,
        image_lookup: dict[str, Path],
    ) -> list[DatasetRecord]:
        annotations = self.parser.parse(annotation_path, manifest.annotation_format)
        grouped: dict[str, list[DetectionAnnotation]] = {}
        for ann in annotations:
            grouped.setdefault(ann.image_id, []).append(ann)

        records: list[DatasetRecord] = []
        for image_id, image_path in image_lookup.items():
            sample = DetectionSample(image_id=image_id, image_path=image_path, split=manifest.split, metadata={"dataset": manifest.dataset_name})
            records.append(DatasetRecord(sample=sample, annotations=tuple(grouped.get(image_id, []))))
        return records

    def export_csv(self, records: Iterable[DatasetRecord], output_path: str | Path) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["image_id", "image_path", "annotation_id", "bbox", "label", "confidence", "split"],
            )
            writer.writeheader()
            for record in records:
                for ann in record.annotations:
                    writer.writerow(
                        {
                            "image_id": record.sample.image_id,
                            "image_path": str(record.sample.image_path),
                            "annotation_id": ann.annotation_id,
                            "bbox": ",".join(map(str, ann.bbox.to_list())),
                            "label": ann.label,
                            "confidence": "" if ann.confidence is None else ann.confidence,
                            "split": record.sample.split,
                        }
                    )
        return out

    def export_json(self, records: Iterable[DatasetRecord], output_path: str | Path) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = []
        for record in records:
            payload.append(
                {
                    "sample": {
                        "image_id": record.sample.image_id,
                        "image_path": str(record.sample.image_path),
                        "split": record.sample.split,
                        "metadata": record.sample.metadata,
                    },
                    "annotations": [
                        {
                            "annotation_id": ann.annotation_id,
                            "image_id": ann.image_id,
                            "bbox": ann.bbox.to_list(),
                            "label": ann.label,
                            "confidence": ann.confidence,
                            "extra": ann.extra,
                        }
                        for ann in record.annotations
                    ],
                }
            )
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out

