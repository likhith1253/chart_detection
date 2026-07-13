"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config.settings import ProjectConfig, load_config


@pytest.fixture
def project_root() -> Path:
    """Return repository root path."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def smoke_config(project_root: Path, tmp_path: Path) -> ProjectConfig:
    """Load smoke-test configuration with isolated output directories."""
    config_path = project_root / "configs" / "smoke_test.yaml"
    config = load_config(config_path, project_root=project_root)

    output_dir = tmp_path / "raw_images"
    metadata_dir = tmp_path / "datasets"

    updated = {
        "seed": config.seed,
        "project_root": str(project_root),
        "dataset": {
            "output_dir": str(output_dir),
            "metadata_dir": str(metadata_dir),
            "labels_csv": str(metadata_dir / "labels.csv"),
            "fingerprint_path": str(metadata_dir / "dataset_fingerprint.json"),
            "total_images": config.dataset.total_images,
            "images_per_type": config.dataset.images_per_type,
            "chart_types": list(config.dataset.chart_types),
            "filename_prefix": config.dataset.filename_prefix,
            "overwrite": True,
        },
        "synthetic": {
            "font_families": list(config.synthetic.font_families),
            "font_sizes": list(config.synthetic.font_sizes),
            "fig_sizes": [list(size) for size in config.synthetic.fig_sizes],
            "dpi_options": list(config.synthetic.dpi_options),
            "palette_names": list(config.synthetic.palette_names),
            "scatter_colormaps": list(config.synthetic.scatter_colormaps),
            "background_colors": list(config.synthetic.background_colors),
            "noise_levels": list(config.synthetic.noise_levels),
            "orientations": list(config.synthetic.orientations),
            "histogram_distributions": list(config.synthetic.histogram_distributions),
        },
        "logging": {
            "level": "WARNING",
            "format": config.logging.format,
            "file_name": None,
        },
    }

    test_config_path = tmp_path / "smoke_test.yaml"
    test_config_path.write_text(yaml.dump(updated), encoding="utf-8")
    return load_config(test_config_path, project_root=project_root)
