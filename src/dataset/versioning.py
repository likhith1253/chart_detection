"""
Dataset fingerprinting, source tracking, and deterministic splits.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class DatasetImage:
    path: Path
    name: str
    source: str
    chart_type: str


def infer_source(file_name: str) -> str:
    lower = file_name.lower()
    if lower.startswith("plotqa"):
        return "plotqa"
    if lower.startswith("dvqa"):
        return "dvqa"
    if lower.startswith("chartqa"):
        return "chartqa"
    if lower.startswith("synthetic"):
        return "synthetic"
    return "unknown"


def infer_chart_type(file_name: str) -> str:
    lower = file_name.lower()
    if "hist" in lower:
        return "histogram"
    if "scatter" in lower:
        return "scatter_plot"
    if "line" in lower:
        return "line_chart"
    if "pie" in lower:
        return "pie_chart"
    if "bar" in lower:
        return "bar_chart"
    return "unknown"


def discover_images(raw_images_dir: str | Path) -> List[DatasetImage]:
    base = Path(raw_images_dir)
    files = [p for p in sorted(base.iterdir()) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    return [
        DatasetImage(
            path=p,
            name=p.name,
            source=infer_source(p.name),
            chart_type=infer_chart_type(p.name),
        )
        for p in files
    ]


def dataset_fingerprint(images: List[DatasetImage]) -> str:
    digest = hashlib.sha256()
    for item in sorted(images, key=lambda x: x.name):
        stat = item.path.stat()
        digest.update(item.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    return digest.hexdigest()


def deterministic_split(
    images: List[DatasetImage],
    ratio: Dict[str, float],
    seed: int,
) -> Dict[str, List[str]]:
    keys = ["train", "val", "test"]
    total_ratio = sum(float(ratio.get(k, 0.0)) for k in keys)
    if total_ratio <= 0:
        ratio = {"train": 0.7, "val": 0.15, "test": 0.15}

    names = [item.name for item in images]
    rnd = random.Random(seed)
    rnd.shuffle(names)

    n = len(names)
    n_train = int(n * ratio.get("train", 0.7))
    n_val = int(n * ratio.get("val", 0.15))
    n_test = max(0, n - n_train - n_val)

    train = names[:n_train]
    val = names[n_train : n_train + n_val]
    test = names[n_train + n_val : n_train + n_val + n_test]
    return {"train": train, "val": val, "test": test}


def build_dataset_manifest(
    raw_images_dir: str | Path,
    manifest_path: str | Path,
    ratio: Dict[str, float],
    seed: int,
) -> Dict:
    images = discover_images(raw_images_dir)
    fingerprint = dataset_fingerprint(images)
    splits = deterministic_split(images, ratio, seed)
    source_counts = Counter(item.source for item in images)
    type_counts = Counter(item.chart_type for item in images)

    payload = {
        "dataset_root": str(Path(raw_images_dir).resolve()),
        "total_images": len(images),
        "fingerprint": fingerprint,
        "source_counts": dict(source_counts),
        "chart_type_counts": dict(type_counts),
        "split_seed": seed,
        "splits": splits,
        "files": [
            {
                "image_name": item.name,
                "image_path": str(item.path.resolve()),
                "source": item.source,
                "chart_type": item.chart_type,
            }
            for item in images
        ],
    }
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
