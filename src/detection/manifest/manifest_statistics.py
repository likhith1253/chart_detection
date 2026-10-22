from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.detection.adapters.base_adapter import DetectionAnnotation, DetectionDatasetManifest, DetectionSample


@dataclass(frozen=True)
class ManifestStatistics:
    images_per_split: dict[str, int]
    images_per_class: dict[str, int]
    images_per_dataset: dict[str, int]
    class_imbalance: dict[str, float]
    missing_metadata: int

    def to_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return out


def build_manifest_statistics(
    manifest: DetectionDatasetManifest,
    samples: Iterable[DetectionSample],
    annotations: Iterable[DetectionAnnotation],
) -> ManifestStatistics:
    samples = list(samples)
    annotations = list(annotations)
    split_counts = pd.Series([s.split for s in samples]).value_counts().to_dict()
    class_counts = pd.Series([a.label for a in annotations]).value_counts().to_dict()
    dataset_counts = pd.Series([s.metadata.get("dataset", "unknown") for s in samples]).value_counts().to_dict()
    total = sum(class_counts.values()) or 1
    class_imbalance = {cls: round(count / total, 6) for cls, count in class_counts.items()}
    missing_metadata = sum(1 for s in samples if not {"dataset", "chart_type", "width", "height"}.issubset(s.metadata))
    return ManifestStatistics(
        images_per_split={k: int(v) for k, v in split_counts.items()},
        images_per_class={k: int(v) for k, v in class_counts.items()},
        images_per_dataset={k: int(v) for k, v in dataset_counts.items()},
        class_imbalance=class_imbalance,
        missing_metadata=int(missing_metadata),
    )

