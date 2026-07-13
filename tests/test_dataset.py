"""Tests for dataset generation and metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config.constants import LABELS_CSV_COLUMNS
from src.config.settings import ProjectConfig, resolve_path
from src.data.dataset import DatasetGenerator, count_images, validate_dataset_integrity


def test_directory_creation(smoke_config: ProjectConfig, tmp_path: Path) -> None:
    """Dataset generation should create required directories."""
    output_dir = resolve_path(smoke_config, smoke_config.dataset.output_dir)
    metadata_dir = resolve_path(smoke_config, smoke_config.dataset.metadata_dir)

    assert not output_dir.exists()
    assert not metadata_dir.exists()

    generator = DatasetGenerator(smoke_config)
    summary = generator.generate()

    assert output_dir.exists()
    assert metadata_dir.exists()
    assert summary.output_dir == output_dir


def test_dataset_generation(smoke_config: ProjectConfig) -> None:
    """Dataset generator should create the configured number of images."""
    generator = DatasetGenerator(smoke_config)
    summary = generator.generate()

    assert summary.total_images == smoke_config.dataset.total_images
    assert count_images(
        summary.output_dir,
        filename_prefix=smoke_config.dataset.filename_prefix,
        chart_types=smoke_config.dataset.chart_types,
    ) == smoke_config.dataset.total_images
    assert set(summary.chart_type_counts) == set(smoke_config.dataset.chart_types)

    for chart_type, count in summary.chart_type_counts.items():
        assert count == smoke_config.dataset.images_per_type


def test_metadata_generation(smoke_config: ProjectConfig) -> None:
    """labels.csv should contain required metadata columns and valid JSON."""
    generator = DatasetGenerator(smoke_config)
    summary = generator.generate()

    labels = pd.read_csv(summary.labels_csv_path)
    assert list(labels.columns) == list(LABELS_CSV_COLUMNS)
    assert len(labels) == smoke_config.dataset.total_images

    for _, row in labels.iterrows():
        params = json.loads(row["generation_parameters"])
        assert isinstance(params, dict)
        assert row["chart_type"] in smoke_config.dataset.chart_types
        assert isinstance(row["random_seed"], (int, float))
        assert (summary.output_dir / row["filename"]).exists()


def test_dataset_integrity_validation(smoke_config: ProjectConfig) -> None:
    """Integrity checks should pass for a freshly generated dataset."""
    generator = DatasetGenerator(smoke_config)
    summary = generator.generate()
    validation = validate_dataset_integrity(smoke_config, summary)

    assert validation["passed"] is True
    assert validation["image_count_matches_config"] is True
    assert validation["labels_row_count_matches_images"] is True
    assert validation["all_chart_types_present"] is True
    assert validation["fingerprint_exists"] is True
