from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ET

from .base_adapter import BoundingBox, DetectionAnnotation


def _annotation_from_mapping(payload: dict[str, Any], default_image_id: str = "") -> DetectionAnnotation:
    bbox = payload.get("bbox") or payload.get("box") or payload.get("bounding_box")
    if isinstance(bbox, dict):
        bbox = [bbox.get("x1"), bbox.get("y1"), bbox.get("x2"), bbox.get("y2")]
    if isinstance(bbox, str):
        bbox = [part.strip() for part in bbox.split(",")]
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("Annotation payload must contain a 4-value bbox")
    return DetectionAnnotation(
        annotation_id=str(payload.get("annotation_id") or payload.get("id") or default_image_id),
        image_id=str(payload.get("image_id") or payload.get("image") or default_image_id),
        bbox=BoundingBox(*map(float, bbox)),
        label=str(payload.get("label") or payload.get("class") or "chart"),
        confidence=float(payload["confidence"]) if payload.get("confidence") is not None else None,
        extra={k: v for k, v in payload.items() if k not in {"annotation_id", "id", "image_id", "image", "bbox", "box", "bounding_box", "label", "class", "confidence"}},
    )


def parse_coco(path: str | Path) -> list[DetectionAnnotation]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    images = {str(img["id"]): img for img in data.get("images", [])}
    categories = {str(cat["id"]): str(cat["name"]) for cat in data.get("categories", [])}
    anns: list[DetectionAnnotation] = []
    for ann in data.get("annotations", []):
        x, y, w, h = map(float, ann["bbox"])
        anns.append(
            DetectionAnnotation(
                annotation_id=str(ann.get("id")),
                image_id=str(ann["image_id"]),
                bbox=BoundingBox(x, y, x + w, y + h),
                label=categories.get(str(ann.get("category_id")), "chart"),
                confidence=float(ann["score"]) if ann.get("score") is not None else None,
                extra={k: v for k, v in ann.items() if k not in {"id", "image_id", "bbox", "category_id", "score"}},
            )
        )
    return anns


def parse_yolo(path: str | Path, image_id: str | None = None, image_size: tuple[float, float] | None = None) -> list[DetectionAnnotation]:
    ann_path = Path(path)
    anns: list[DetectionAnnotation] = []
    for idx, line in enumerate(ann_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) not in {5, 6}:
            raise ValueError(f"Invalid YOLO line: {line!r}")
        cls, cx, cy, w, h = parts[:5]
        score = float(parts[5]) if len(parts) == 6 else None
        cx, cy, w, h = map(float, (cx, cy, w, h))
        if image_size is not None:
            img_w, img_h = image_size
            x1 = (cx - w / 2.0) * img_w
            y1 = (cy - h / 2.0) * img_h
            x2 = (cx + w / 2.0) * img_w
            y2 = (cy + h / 2.0) * img_h
        else:
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0
        anns.append(
            DetectionAnnotation(
                annotation_id=f"{ann_path.stem}_{idx}",
                image_id=image_id or ann_path.stem,
                bbox=BoundingBox(x1, y1, x2, y2),
                label=str(cls),
                confidence=score,
            )
        )
    return anns


def parse_pascal_voc(path: str | Path) -> list[DetectionAnnotation]:
    root = ET.parse(str(path)).getroot()
    image_id = root.findtext("filename") or Path(path).stem
    anns: list[DetectionAnnotation] = []
    for idx, obj in enumerate(root.findall("object")):
        name = obj.findtext("name") or "chart"
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin", "0"))
        ymin = float(box.findtext("ymin", "0"))
        xmax = float(box.findtext("xmax", "0"))
        ymax = float(box.findtext("ymax", "0"))
        anns.append(
            DetectionAnnotation(
                annotation_id=f"{image_id}_{idx}",
                image_id=image_id,
                bbox=BoundingBox(xmin, ymin, xmax, ymax),
                label=name,
            )
        )
    return anns


def parse_csv(path: str | Path) -> list[DetectionAnnotation]:
    anns: list[DetectionAnnotation] = []
    with Path(path).open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            anns.append(_annotation_from_mapping(row))
    return anns


def parse_json(path: str | Path) -> list[DetectionAnnotation]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [_annotation_from_mapping(item) for item in data]
    if isinstance(data, dict):
        if "annotations" in data:
            return [_annotation_from_mapping(item) for item in data["annotations"]]
        return [_annotation_from_mapping(data)]
    raise TypeError("Unsupported JSON annotation structure")


class AnnotationParser:
    def parse(self, path: str | Path, format_name: str, **kwargs: Any) -> list[DetectionAnnotation]:
        fmt = format_name.lower()
        if fmt == "coco":
            return parse_coco(path)
        if fmt == "yolo":
            return parse_yolo(path, image_id=kwargs.get("image_id"))
        if fmt in {"voc", "pascal_voc", "xml"}:
            return parse_pascal_voc(path)
        if fmt == "csv":
            return parse_csv(path)
        if fmt == "json":
            return parse_json(path)
        raise ValueError(f"Unsupported annotation format: {format_name}")


def parse_annotation_file(path: str | Path, format_name: str, **kwargs: Any) -> list[DetectionAnnotation]:
    return AnnotationParser().parse(path, format_name, **kwargs)
