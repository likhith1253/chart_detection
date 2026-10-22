from __future__ import annotations

import json
from dataclasses import dataclass
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


DOCLAYNET_CLASSES = (
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
)


def _dataset_root_from_yaml(yaml_path: Path) -> Path:
    return yaml_path.parent


def load_yolo_manifest(yaml_path: str | Path, split: str = "train") -> tuple[DetectionDatasetManifest, list[DetectionSample], list[DetectionAnnotation]]:
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(yaml_path)
    root_dir = _dataset_root_from_yaml(yaml_path)
    split_alias = {"validation": "val"}.get(split, split)
    image_dir = root_dir / "images" / split_alias
    label_dir = root_dir / "labels" / split_alias
    if not image_dir.exists():
        raise FileNotFoundError(image_dir)
    if not label_dir.exists():
        raise FileNotFoundError(label_dir)

    import yaml

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = data.get("names", {})
    if isinstance(names, dict):
        class_names = tuple(names[k] for k in sorted(names))
    else:
        class_names = tuple(names)
    if not class_names:
        class_names = ("chart",)

    samples: list[DetectionSample] = []
    annotations: list[DetectionAnnotation] = []
    for image_path in sorted(image_dir.glob("*")):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
            continue
        sample = DetectionSample(image_id=image_path.stem, image_path=image_path, split=split, metadata={"dataset": "doclaynet"})
        samples.append(sample)
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        for idx, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            cls_idx, cx, cy, bw, bh = line.split()[:5]
            cls_name = class_names[int(cls_idx)] if int(cls_idx) < len(class_names) else str(cls_idx)
            x1 = float(cx) - float(bw) / 2.0
            y1 = float(cy) - float(bh) / 2.0
            x2 = float(cx) + float(bw) / 2.0
            y2 = float(cy) + float(bh) / 2.0
            annotations.append(
                DetectionAnnotation(
                    annotation_id=f"{image_path.stem}_{idx}",
                    image_id=image_path.stem,
                    bbox=BoundingBox(x1, y1, x2, y2),
                    label=cls_name,
                )
            )

    manifest = DetectionDatasetManifest(
        dataset_name="doclaynet_yolo",
        root_dir=root_dir,
        image_dir=root_dir / "images",
        annotation_dir=root_dir / "labels",
        annotation_format="yolo",
        split=split_alias,
        image_count=len(samples),
        annotation_count=len(annotations),
        classes=class_names,
        extra={"source_manifest": str(yaml_path), "splits": ("train", "val", "test")},
    )
    return manifest, samples, annotations


def load_doclaynet_manifest(root_dir: str | Path, split: str = "train") -> tuple[DetectionDatasetManifest, list[DetectionSample], list[DetectionAnnotation]]:
    root_dir = Path(root_dir)
    coco_path = root_dir / "COCO" / f"{split}.json"
    if not coco_path.exists():
        raise FileNotFoundError(coco_path)
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    categories = {int(cat["id"]): str(cat["name"]) for cat in data.get("categories", [])}
    images = {int(img["id"]): img for img in data.get("images", [])}

    samples: list[DetectionSample] = []
    for img in images.values():
        samples.append(
            DetectionSample(
                image_id=str(img["id"]),
                image_path=root_dir / "PNG" / str(img["file_name"]),
                split=split,
                metadata={k: img.get(k) for k in ["width", "height", "collection", "doc_name", "page_no", "precedence", "doc_category"]},
            )
        )

    annotations: list[DetectionAnnotation] = []
    for ann in data.get("annotations", []):
        x, y, w, h = map(float, ann["bbox"])
        annotations.append(
            DetectionAnnotation(
                annotation_id=str(ann["id"]),
                image_id=str(ann["image_id"]),
                bbox=BoundingBox(x, y, x + w, y + h),
                label=categories.get(int(ann["category_id"]), "Picture"),
                extra={k: v for k, v in ann.items() if k not in {"id", "image_id", "bbox", "category_id"}},
            )
        )

    manifest = DetectionDatasetManifest(
        dataset_name="doclaynet",
        root_dir=root_dir,
        image_dir=root_dir / "PNG",
        annotation_dir=root_dir / "COCO",
        annotation_format="coco",
        split=split,
        image_count=len(samples),
        annotation_count=len(annotations),
        classes=DOCLAYNET_CLASSES,
        extra={"source_manifest": str(coco_path), "splits": ("train", "val", "test")},
    )
    return manifest, samples, annotations
