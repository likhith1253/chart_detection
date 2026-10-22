from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.detection.adapters.base_adapter import DetectionAnnotation, DetectionDatasetManifest, DetectionSample
from .manifest_builder import build_manifest
from .doclaynet import load_doclaynet_manifest, load_yolo_manifest


def _load_benchmark_dataframe(manifest_csv: str | Path | None = None) -> pd.DataFrame:
    csv_path = Path(manifest_csv or Path("data") / "datasets" / "benchmark_manifest.csv")
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    return pd.read_csv(csv_path)


def load_manifest(split: str = "train", manifest_csv: str | Path | None = None) -> tuple[DetectionDatasetManifest, list[DetectionSample], list[DetectionAnnotation]]:
    path = Path(manifest_csv or Path("data") / "datasets" / "benchmark_manifest.csv")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return load_yolo_manifest(path, split=split)
    if path.is_dir() and (path / "dataset.yaml").exists():
        return load_yolo_manifest(path / "dataset.yaml", split=split)
    if path.exists():
        try:
            if path.parent.name == "doclaynet" and (path / "COCO").exists():
                return load_doclaynet_manifest(path, split=split)
        except Exception:
            pass
    df = _load_benchmark_dataframe(path)
    manifest, samples, records = build_manifest(df, split=split)
    annotations = [annotation for record in records for annotation in record.annotations]
    return manifest, samples, annotations
