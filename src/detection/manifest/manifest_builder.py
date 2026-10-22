from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.detection.adapters.base_adapter import (
    BoundingBox,
    DetectionAnnotation,
    DetectionDatasetManifest,
    DetectionSample,
    DatasetRecord,
)


def _default_annotation(sample: DetectionSample, width: int, height: int, label: str) -> DetectionAnnotation:
    return DetectionAnnotation(
        annotation_id=f"{sample.image_id}_chart",
        image_id=sample.image_id,
        bbox=BoundingBox(0.0, 0.0, float(width), float(height)),
        label=label,
        extra={"source": "benchmark_manifest"},
    )


def build_manifest(df: pd.DataFrame, split: str) -> tuple[DetectionDatasetManifest, list[DetectionSample], list[DatasetRecord]]:
    required = {"image_id", "dataset", "image_path", "chart_type", "split", "license", "width", "height"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Benchmark manifest missing columns: {sorted(missing)}")

    split_df = df[df["split"] == split].copy()
    if split_df.empty:
        raise ValueError(f"No rows found for split={split!r}")

    dataset_name = f"benchmark_{split}"
    root_dir = Path("data") / "benchmark"
    image_dir = root_dir
    manifest = DetectionDatasetManifest(
        dataset_name=dataset_name,
        root_dir=root_dir,
        image_dir=image_dir,
        annotation_dir=None,
        annotation_format="benchmark",
        split=split,
        image_count=int(len(split_df)),
        annotation_count=int(len(split_df)),
        classes=tuple(sorted({str(v) for v in split_df["chart_type"].dropna().unique()})),
        extra={
            "source_manifest": str(Path("data") / "datasets" / "benchmark_manifest.csv"),
            "splits": ("train", "validation", "test"),
        },
    )

    samples: list[DetectionSample] = []
    records: list[DatasetRecord] = []
    for _, row in split_df.iterrows():
        sample = DetectionSample(
            image_id=str(row["image_id"]),
            image_path=Path(str(row["image_path"])),
            split=str(row["split"]),
            metadata={
                "dataset": str(row["dataset"]),
                "chart_type": str(row["chart_type"]),
                "license": str(row["license"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
            },
        )
        annotation = _default_annotation(sample, int(row["width"]), int(row["height"]), str(row["chart_type"]))
        samples.append(sample)
        records.append(DatasetRecord(sample=sample, annotations=(annotation,)))

    return manifest, samples, records

