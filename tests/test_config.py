"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.constants import CHART_TYPES, DEFAULT_CONFIG_PATH
from src.config.settings import (
    configuration_hash,
    load_config,
    resolve_path,
)


def test_load_default_config(project_root: Path) -> None:
    """Default configuration should load and validate."""
    config_path = project_root / DEFAULT_CONFIG_PATH
    config = load_config(config_path, project_root=project_root)

    assert config.seed == 42
    assert config.dataset.total_images == 2500
    assert config.dataset.images_per_type == 500
    assert config.dataset.chart_types == CHART_TYPES


def test_load_smoke_test_config(project_root: Path) -> None:
    """Smoke-test configuration should load with reduced dataset size."""
    config_path = project_root / "configs" / "smoke_test.yaml"
    config = load_config(config_path, project_root=project_root)

    assert config.dataset.total_images == 10
    assert config.dataset.images_per_type == 2
    assert len(config.dataset.chart_types) == 5


def test_configuration_hash_is_stable(project_root: Path) -> None:
    """Configuration hash should be deterministic for the same config."""
    config_path = project_root / DEFAULT_CONFIG_PATH
    first = load_config(config_path, project_root=project_root)
    second = load_config(config_path, project_root=project_root)

    assert configuration_hash(first) == configuration_hash(second)
    assert len(configuration_hash(first)) == 64


def test_resolve_path_uses_project_root(project_root: Path) -> None:
    """Relative paths should resolve against project root."""
    config = load_config(project_root / DEFAULT_CONFIG_PATH, project_root=project_root)
    resolved = resolve_path(config, config.dataset.output_dir)

    assert resolved == project_root / "data" / "raw_images"


def test_invalid_total_images_raises(project_root: Path, tmp_path: Path) -> None:
    """Mismatched total_images should fail validation."""
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text(
        "\n".join(
            [
                "seed: 1",
                "dataset:",
                "  total_images: 100",
                "  images_per_type: 2",
                "  chart_types: [bar, line]",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="total_images"):
        load_config(invalid_config, project_root=project_root)
