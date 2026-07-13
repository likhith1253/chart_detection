"""YAML-based configuration loader with typed dataclasses."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.config.constants import CHART_TYPES


@dataclass(frozen=True)
class SyntheticConfig:
    """Parameters controlling synthetic chart appearance."""

    font_families: tuple[str, ...] = (
        "DejaVu Sans",
        "DejaVu Serif",
        "DejaVu Sans Mono",
    )
    font_sizes: tuple[int, ...] = (10, 11, 12, 13, 14)
    fig_sizes: tuple[tuple[float, float], ...] = (
        (8.0, 5.0),
        (10.0, 6.0),
        (9.0, 5.0),
        (10.0, 7.0),
        (12.0, 6.0),
    )
    dpi_options: tuple[int, ...] = (100, 150)
    palette_names: tuple[str, ...] = (
        "Paired",
        "Set2",
        "Set3",
        "tab10",
        "Pastel1",
        "Dark2",
        "Accent",
        "tab20",
    )
    scatter_colormaps: tuple[str, ...] = (
        "viridis",
        "plasma",
        "coolwarm",
        "RdYlBu",
    )
    background_colors: tuple[str, ...] = (
        "white",
        "#f5f5f5",
        "#eef2f7",
        "#fff8e7",
        "#f0fff0",
    )
    noise_levels: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05)
    orientations: tuple[str, ...] = ("vertical", "horizontal")
    histogram_distributions: tuple[str, ...] = (
        "normal",
        "uniform",
        "exponential",
        "lognormal",
    )
    bar_category_pools: tuple[tuple[str, ...], ...] = (
        ("Sales", "Profit", "Revenue", "Cost", "Tax"),
        ("Q1", "Q2", "Q3", "Q4"),
        ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"),
        ("Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"),
        ("Jan", "Feb", "Mar", "Apr", "May", "Jun"),
        ("USA", "UK", "India", "China", "Brazil", "Germany"),
    )
    title_pools: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "bar": (
                "Quarterly Revenue by Region",
                "Annual Sales Performance",
                "Budget Allocation by Department",
                "Product Comparison Results",
            ),
            "histogram": (
                "Exam Score Distribution",
                "Response Time Distribution",
                "Age Distribution of Users",
                "Salary Distribution",
            ),
            "line": (
                "Temperature Over Time",
                "Stock Price Trend",
                "Monthly Website Traffic",
                "CPU Usage Over 24 Hours",
            ),
            "scatter": (
                "Height vs Weight",
                "Income vs Education",
                "Temperature vs Pressure",
                "Study Hours vs Test Score",
            ),
            "pie": (
                "Market Share Distribution",
                "Budget Breakdown",
                "Resource Allocation",
                "Traffic Source Distribution",
            ),
        }
    )


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset paths and generation targets."""

    output_dir: str = "data/raw_images"
    metadata_dir: str = "data/datasets"
    labels_csv: str = "data/datasets/labels.csv"
    fingerprint_path: str = "data/datasets/dataset_fingerprint.json"
    total_images: int = 2500
    images_per_type: int = 500
    chart_types: tuple[str, ...] = CHART_TYPES
    filename_prefix: str = "synthetic"
    overwrite: bool = False


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    file_name: str | None = None


@dataclass(frozen=True)
class ProjectConfig:
    """Root configuration object."""

    seed: int = 42
    project_root: str = "."
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _to_tuple(value: Any) -> tuple:
    """Convert YAML sequences to tuples for immutable config objects."""
    if isinstance(value, list):
        return tuple(_to_tuple(item) for item in value)
    if isinstance(value, dict):
        return {key: _to_tuple(item) for key, item in value.items()}
    return value


def _build_dataset_config(data: dict[str, Any]) -> DatasetConfig:
    """Build DatasetConfig from a YAML mapping."""
    kwargs = {key: _to_tuple(value) for key, value in data.items()}
    return DatasetConfig(**kwargs)


def _build_synthetic_config(data: dict[str, Any]) -> SyntheticConfig:
    """Build SyntheticConfig from a YAML mapping."""
    kwargs = {key: _to_tuple(value) for key, value in data.items()}
    return SyntheticConfig(**kwargs)


def _build_logging_config(data: dict[str, Any]) -> LoggingConfig:
    """Build LoggingConfig from a YAML mapping."""
    return LoggingConfig(**data)


def load_config(config_path: str | Path, project_root: str | Path | None = None) -> ProjectConfig:
    """
    Load and validate project configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.
        project_root: Optional project root override for resolving relative paths.

    Returns:
        Validated ProjectConfig instance.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If required fields are missing or invalid.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")

    root = Path(project_root or raw.get("project_root", ".")).resolve()

    dataset_data = raw.get("dataset", {})
    synthetic_data = raw.get("synthetic", {})
    logging_data = raw.get("logging", {})

    config = ProjectConfig(
        seed=int(raw.get("seed", 42)),
        project_root=str(root),
        dataset=_build_dataset_config(dataset_data),
        synthetic=_build_synthetic_config(synthetic_data),
        logging=_build_logging_config(logging_data),
    )
    _validate_config(config)
    return config


def _validate_config(config: ProjectConfig) -> None:
    """Validate configuration values."""
    if config.seed < 0:
        raise ValueError("seed must be non-negative")

    if config.dataset.total_images <= 0:
        raise ValueError("dataset.total_images must be positive")

    if config.dataset.images_per_type <= 0:
        raise ValueError("dataset.images_per_type must be positive")

    expected_total = config.dataset.images_per_type * len(config.dataset.chart_types)
    if config.dataset.total_images != expected_total:
        raise ValueError(
            "dataset.total_images must equal images_per_type * number of chart_types "
            f"({expected_total}), got {config.dataset.total_images}"
        )

    for chart_type in config.dataset.chart_types:
        if chart_type not in CHART_TYPES:
            raise ValueError(f"Unsupported chart type in config: {chart_type}")


def resolve_path(config: ProjectConfig, relative_path: str | Path) -> Path:
    """Resolve a path relative to the configured project root."""
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return Path(config.project_root) / path


def config_to_canonical_dict(config: ProjectConfig) -> dict[str, Any]:
    """Return a canonical JSON-serializable configuration dictionary."""
    payload = asdict(config)
    return json.loads(json.dumps(payload, sort_keys=True))


def configuration_hash(config: ProjectConfig) -> str:
    """Compute a stable hash of the active configuration."""
    import hashlib

    canonical = json.dumps(config_to_canonical_dict(config), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
