"""Tests for dataset fingerprint generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.config.constants import FINGERPRINT_FIELDS
from src.config.settings import ProjectConfig, configuration_hash, resolve_path
from src.data.dataset import DatasetGenerator
from src.data.versioning import (
    build_dataset_fingerprint,
    collect_software_versions,
    compute_dataset_hash,
    load_dataset_fingerprint,
    write_dataset_fingerprint,
)


def test_fingerprint_generation(smoke_config: ProjectConfig) -> None:
    """Fingerprint file should include all required fields."""
    generator = DatasetGenerator(smoke_config)
    summary = generator.generate()

    fingerprint = load_dataset_fingerprint(smoke_config)
    for field_name in FINGERPRINT_FIELDS:
        assert field_name in fingerprint

    assert fingerprint["generation_seed"] == smoke_config.seed
    assert fingerprint["configuration_hash"] == configuration_hash(smoke_config)
    assert len(fingerprint["dataset_hash"]) == 64
    assert isinstance(fingerprint["software_versions"], dict)
    assert "python" in fingerprint["software_versions"]


def test_dataset_hash_is_stable(smoke_config: ProjectConfig) -> None:
    """Dataset hash should remain stable for unchanged files."""
    generator = DatasetGenerator(smoke_config)
    summary = generator.generate()

    first_hash = compute_dataset_hash(summary.output_dir)
    second_hash = compute_dataset_hash(summary.output_dir)
    assert first_hash == second_hash


def test_write_dataset_fingerprint(smoke_config: ProjectConfig) -> None:
    """Fingerprint writer should persist valid JSON."""
    generator = DatasetGenerator(smoke_config)
    summary = generator.generate()

    fingerprint_path = resolve_path(smoke_config, smoke_config.dataset.fingerprint_path)
    payload = json.loads(fingerprint_path.read_text(encoding="utf-8"))

    assert payload["dataset_hash"] == compute_dataset_hash(summary.output_dir)
    assert payload["timestamp"]


def test_build_dataset_fingerprint_timestamp_override(smoke_config: ProjectConfig) -> None:
    """Fingerprint builder should accept explicit timestamps."""
    generator = DatasetGenerator(smoke_config)
    summary = generator.generate()

    fixed_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    fingerprint = build_dataset_fingerprint(
        smoke_config,
        summary.output_dir,
        generation_seed=smoke_config.seed,
        timestamp=fixed_time,
    )

    assert fingerprint["timestamp"] == fixed_time.isoformat()


def test_collect_software_versions() -> None:
    """Software version collector should return package versions."""
    versions = collect_software_versions()
    assert "python" in versions
    assert "numpy" in versions
    assert "matplotlib" in versions
