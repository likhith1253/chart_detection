"""Global constants for chart classification research."""

from __future__ import annotations

CHART_TYPES: tuple[str, ...] = ("bar", "histogram", "line", "scatter", "pie")

CHART_TYPE_ALIASES: dict[str, str] = {
    "bar": "bar",
    "histogram": "histogram",
    "hist": "histogram",
    "line": "line",
    "scatter": "scatter",
    "pie": "pie",
}

SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
)

DEFAULT_CONFIG_PATH: str = "configs/default.yaml"
SMOKE_TEST_CONFIG_PATH: str = "configs/smoke_test.yaml"

LABELS_CSV_COLUMNS: tuple[str, ...] = (
    "filename",
    "chart_type",
    "generation_parameters",
    "random_seed",
)

FINGERPRINT_FIELDS: tuple[str, ...] = (
    "dataset_hash",
    "configuration_hash",
    "timestamp",
    "software_versions",
    "generation_seed",
)
