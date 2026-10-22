from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image

from src.detection.manifest import build_manifest_statistics, load_manifest, validate_manifest


def test_load_manifest_returns_samples_annotations_and_manifest() -> None:
    manifest, samples, annotations = load_manifest(split="train")
    assert manifest.split == "train"
    assert samples
    assert annotations
    assert all(sample.split == "train" for sample in samples)
    assert all(ann.label for ann in annotations)


def test_manifest_statistics_are_built_from_benchmark() -> None:
    manifest, samples, annotations = load_manifest(split="validation")
    stats = build_manifest_statistics(manifest, samples, annotations)
    assert stats.images_per_split["validation"] == len(samples)
    assert stats.images_per_class
    assert stats.images_per_dataset
    assert stats.missing_metadata == 0


def test_manifest_validation_report(tmp_path) -> None:
    manifest, samples, annotations = load_manifest(split="test")
    report = validate_manifest(manifest, samples, annotations)
    assert report.valid
    out = report.to_json(tmp_path / "manifest_validation_report.json")
    assert out.exists()


def test_manifest_loader_respects_split_only() -> None:
    train_manifest, train_samples, _ = load_manifest(split="train")
    test_manifest, test_samples, _ = load_manifest(split="test")
    assert train_manifest.split == "train"
    assert test_manifest.split == "test"
    assert {s.split for s in train_samples} == {"train"}
    assert {s.split for s in test_samples} == {"test"}

