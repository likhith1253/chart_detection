from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from src.detection.adapters import (
    AnnotationParser,
    BoundingBox,
    DatasetConverter,
    DatasetRecord,
    DatasetValidator,
    DetectionDatasetManifest,
    DetectionSample,
    get_adapter_registry,
    parse_annotation_file,
)
from src.detection.adapters.annotation_parser import parse_coco, parse_csv, parse_json, parse_pascal_voc, parse_yolo


def _write_image(path: Path, size: tuple[int, int] = (32, 24)) -> None:
    Image.new("RGB", size, color="white").save(path)


def test_parse_coco_yolo_voc_csv_json(tmp_path) -> None:
    img_path = tmp_path / "page.png"
    _write_image(img_path)

    coco_path = tmp_path / "ann.json"
    coco_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "page.png"}],
                "categories": [{"id": 2, "name": "chart"}],
                "annotations": [{"id": 10, "image_id": 1, "category_id": 2, "bbox": [1, 2, 3, 4]}],
            }
        ),
        encoding="utf-8",
    )
    anns = parse_coco(coco_path)
    assert anns[0].bbox.to_list() == [1.0, 2.0, 4.0, 6.0]

    yolo_path = tmp_path / "ann.txt"
    yolo_path.write_text("0 0.5 0.5 0.25 0.5 0.9\n", encoding="utf-8")
    anns = parse_yolo(yolo_path, image_id="page", image_size=(32, 24))
    assert anns[0].bbox.to_list() == [12.0, 6.0, 20.0, 18.0]

    voc_path = tmp_path / "ann.xml"
    voc_path.write_text(
        "<annotation><filename>page.png</filename><object><name>chart</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>4</xmax><ymax>6</ymax></bndbox></object></annotation>",
        encoding="utf-8",
    )
    anns = parse_pascal_voc(voc_path)
    assert anns[0].image_id == "page.png"

    csv_path = tmp_path / "ann.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["annotation_id", "image_id", "bbox", "label", "confidence"])
        writer.writeheader()
        writer.writerow({"annotation_id": "1", "image_id": "page", "bbox": "1,2,4,6", "label": "chart", "confidence": "0.8"})
    anns = parse_csv(csv_path)
    assert anns[0].confidence == 0.8

    json_path = tmp_path / "ann2.json"
    json_path.write_text(
        json.dumps([{"annotation_id": "1", "image_id": "page", "bbox": [1, 2, 4, 6], "label": "chart"}]),
        encoding="utf-8",
    )
    anns = parse_json(json_path)
    assert anns[0].label == "chart"

    assert len(parse_annotation_file(json_path, "json")) == 1


def test_dataset_converter_and_validator(tmp_path) -> None:
    img_path = tmp_path / "page.png"
    _write_image(img_path)
    manifest = DetectionDatasetManifest(
        dataset_name="demo",
        root_dir=tmp_path,
        image_dir=tmp_path,
        annotation_dir=tmp_path,
        annotation_format="json",
        split="train",
        classes=("chart",),
    )
    ann_path = tmp_path / "ann.json"
    ann_path.write_text(
        json.dumps([{"annotation_id": "a1", "image_id": "img1", "bbox": [1, 2, 10, 12], "label": "chart"}]),
        encoding="utf-8",
    )
    converter = DatasetConverter()
    records = converter.to_internal_records(manifest, ann_path, {"img1": img_path})
    assert len(records) == 1
    assert records[0].annotations[0].bbox.is_valid()

    validator = DatasetValidator()
    report = validator.validate(manifest, records)
    assert report.valid
    report_path = report.to_json(tmp_path / "validation_report.json")
    assert report_path.exists()


def test_validation_detects_problems(tmp_path) -> None:
    manifest = DetectionDatasetManifest(
        dataset_name="demo",
        root_dir=tmp_path,
        image_dir=tmp_path,
        annotation_dir=tmp_path,
        annotation_format="json",
        split="train",
        classes=("chart",),
    )
    records = [
        DatasetRecord(
            sample=DetectionSample(image_id="dup", image_path=tmp_path / "missing.png"),
            annotations=(
                parse_annotation_file(
                    tmp_path / "does_not_matter.json",
                    "json",
                )
                if False
                else ()
            ),
        ),
        DatasetRecord(
            sample=DetectionSample(image_id="dup", image_path=tmp_path / "missing.png"),
            annotations=(
                tuple(
                    [
                        __import__("src.detection.adapters.base_adapter", fromlist=["BoundingBox"]).BoundingBox(-1, 2, 0, 1)
                    ]
                )
                if False
                else ()
            ),
        ),
    ]
    # Use a minimal valid annotation with an invalid bbox and label by hand.
    from src.detection.adapters.base_adapter import DetectionAnnotation, BoundingBox

    records = [
        DatasetRecord(
            sample=DetectionSample(image_id="dup", image_path=tmp_path / "missing.png"),
            annotations=(DetectionAnnotation(annotation_id="a1", image_id="dup", bbox=BoundingBox(-1, 2, 0, 1), label="bad"),),
        ),
        DatasetRecord(
            sample=DetectionSample(image_id="dup", image_path=tmp_path / "missing.png"),
            annotations=(DetectionAnnotation(annotation_id="a1", image_id="dup", bbox=BoundingBox(-1, 2, 0, 1), label="bad"),),
        ),
    ]
    report = DatasetValidator().validate(manifest, records)
    assert not report.valid
    assert report.missing_images
    assert report.duplicate_ids
    assert report.duplicate_boxes
    assert report.invalid_labels
    assert report.negative_coordinates
    assert report.invalid_bounding_boxes


def test_registry_is_backend_agnostic() -> None:
    registry = get_adapter_registry()
    assert isinstance(registry.known_formats(), tuple)

