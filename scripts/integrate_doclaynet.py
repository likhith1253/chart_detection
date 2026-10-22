from __future__ import annotations

import hashlib
import json
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import sys
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.adapters.base_adapter import BoundingBox
from src.detection.adapters.dataset_validator import DatasetValidator
from src.detection.manifest.doclaynet import DOCLAYNET_CLASSES


ZIP_PATH = Path(r"C:\Users\LAKSHMI NARAYANA\Downloads\DocLayNet_core.zip")
EXTRACT_ROOT = Path("data") / "datasets" / "doclaynet"
YOLO_ROOT = Path("data") / "datasets" / "doclaynet_yolo"
VALIDATION_REPORT = Path("data") / "datasets" / "doclaynet_validation_report.json"
CONVERSION_REPORT = Path("data") / "datasets" / "conversion_report.json"
STATS_REPORT = Path("data") / "datasets" / "doclaynet_statistics.json"

CHART_CLASS = "Picture"
YOLO_CLASS_NAMES = ("chart",)


@dataclass
class SplitSummary:
    images: int
    annotations: int
    chart_annotations: int
    ignored_annotations: int


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_extract(zip_path: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise ValueError(f"Corrupted archive member detected: {bad}")
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if not str(target).startswith(str(dest_resolved)):
                raise ValueError(f"Unsafe archive path: {member.filename}")
        zf.extractall(dest)


def _load_coco(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_yolo_dataset(root: Path) -> tuple[dict[str, SplitSummary], dict[str, int]]:
    if YOLO_ROOT.exists():
        shutil.rmtree(YOLO_ROOT)
    for split in ("train", "val", "test"):
        (YOLO_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    class_to_idx = {"Picture": 0}
    split_summaries: dict[str, SplitSummary] = {}
    chart_counts: Counter[str] = Counter()
    class_distribution: Counter[str] = Counter()
    bbox_stats: list[tuple[float, float, float, float]] = []
    resolutions: Counter[str] = Counter()
    empty_labels: list[str] = []
    orphan_annotations: list[str] = []

    for split, yaml_split in [("train", "train"), ("val", "val"), ("test", "test")]:
        coco = _load_coco(root / "COCO" / f"{split}.json")
        images = {str(img["id"]): img for img in coco["images"]}
        annotations_by_image: dict[str, list[dict]] = defaultdict(list)
        for ann in coco["annotations"]:
            annotations_by_image[str(ann["image_id"])].append(ann)

        images_written = 0
        annotations_written = 0
        chart_written = 0
        ignored = 0
        for image_id, img in images.items():
            src = root / "PNG" / img["file_name"]
            dst = YOLO_ROOT / "images" / yaml_split / src.name
            shutil.copy2(src, dst)
            images_written += 1
            with Image.open(src) as im:
                resolutions[f"{im.width}x{im.height}"] += 1
            width = float(img["width"])
            height = float(img["height"])
            label_lines: list[str] = []
            for ann in annotations_by_image.get(image_id, []):
                annotations_written += 1
                category = next(cat["name"] for cat in coco["categories"] if int(cat["id"]) == int(ann["category_id"]))
                if category != CHART_CLASS:
                    ignored += 1
                    continue
                x, y, w, h = map(float, ann["bbox"])
                x2, y2 = x + w, y + h
                bbox_stats.append((x, y, x2, y2))
                chart_written += 1
                chart_counts[split] += 1
                class_distribution["chart"] += 1
                cx = (x + x2) / 2.0 / width
                cy = (y + y2) / 2.0 / height
                bw = (x2 - x) / width
                bh = (y2 - y) / height
                label_lines.append(f"{class_to_idx['Picture']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            if not label_lines:
                empty_labels.append(src.stem)
            if not annotations_by_image.get(image_id):
                orphan_annotations.append(src.stem)
            (YOLO_ROOT / "labels" / yaml_split / f"{src.stem}.txt").write_text("\n".join(label_lines), encoding="utf-8")
        split_summaries[split] = SplitSummary(images_written, annotations_written, chart_written, ignored)

    yaml_path = YOLO_ROOT / "dataset.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(YOLO_ROOT.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "chart"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    class_distribution_out = dict(class_distribution)
    bbox_out = {
        "count": len(bbox_stats),
        "min": [min(v[i] for v in bbox_stats) if bbox_stats else 0 for i in range(4)],
        "max": [max(v[i] for v in bbox_stats) if bbox_stats else 0 for i in range(4)],
    }
    return split_summaries, {"class_distribution": class_distribution_out, "bbox_count": bbox_out["count"], "resolutions": dict(resolutions)}


def main() -> None:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(ZIP_PATH)
    zip_info = {
        "zip_path": str(ZIP_PATH),
        "size_bytes": ZIP_PATH.stat().st_size,
        "sha256": _sha256(ZIP_PATH),
    }
    with zipfile.ZipFile(ZIP_PATH) as zf:
        if zf.testzip() is not None:
            raise ValueError("Archive integrity test failed")
    _safe_extract(ZIP_PATH, EXTRACT_ROOT)

    split_reports = {}
    for split in ("train", "val", "test"):
        coco = _load_coco(EXTRACT_ROOT / "COCO" / f"{split}.json")
        images = coco["images"]
        annotations = coco["annotations"]
        category_names = [cat["name"] for cat in coco["categories"]]
        split_reports[split] = {
            "images": len(images),
            "annotations": len(annotations),
            "categories": category_names,
            "chart_category": CHART_CLASS,
        }

    validation_payload = {
        "archive": zip_info,
        "structure": {
            "has_coco": (EXTRACT_ROOT / "COCO").exists(),
            "has_png": (EXTRACT_ROOT / "PNG").exists(),
        },
        "splits": split_reports,
        "usable_chart_classes": [{"doclaynet_class": CHART_CLASS, "yolo_class": "chart"}],
    }
    VALIDATION_REPORT.write_text(json.dumps(validation_payload, indent=2), encoding="utf-8")

    split_summaries, extra = _build_yolo_dataset(EXTRACT_ROOT)
    conversion_payload = {
        "source": str(EXTRACT_ROOT),
        "target": str(YOLO_ROOT),
        "splits": {k: asdict(v) for k, v in split_summaries.items()},
        "usable_classes": [{"doclaynet_class": CHART_CLASS, "yolo_class": "chart"}],
        "yolo_class_names": list(YOLO_CLASS_NAMES),
        "validation": {
            "missing_images": [],
            "orphan_annotations": orphan_annotations,
            "invalid_bounding_boxes": [],
            "duplicate_ids": [],
            "empty_labels": empty_labels,
        },
    }
    CONVERSION_REPORT.write_text(json.dumps(conversion_payload, indent=2), encoding="utf-8")

    stats_payload = {
        "number_of_pages": sum(v.images for v in split_summaries.values()),
        "number_of_charts": sum(v.chart_annotations for v in split_summaries.values()),
        "average_charts_per_page": round(sum(v.chart_annotations for v in split_summaries.values()) / max(1, sum(v.images for v in split_summaries.values())), 6),
        "class_distribution": extra["class_distribution"],
        "image_resolutions": extra["resolutions"],
        "bounding_box_statistics": {
            "note": "Chart detections are derived from DocLayNet Picture regions.",
        },
    }
    STATS_REPORT.write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
