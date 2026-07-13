from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from pathlib import PureWindowsPath
from time import perf_counter
from typing import Iterable

import pandas as pd
from PIL import Image


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class DatasetRoots:
    metadata_root: Path
    image_root: Path


@dataclass(frozen=True)
class ImageResolutionStats:
    indexed_images: int
    resolution_time_sec: float
    missing_images: int


def resolve_dataset_roots(
    metadata_root: str | Path | None = None,
    image_root: str | Path | None = None,
) -> DatasetRoots:
    return DatasetRoots(
        metadata_root=Path(metadata_root or "data/datasets"),
        image_root=Path(image_root or "data/raw_images"),
    )


def default_split_csv(metadata_root: str | Path) -> Path:
    return Path(metadata_root) / "unified_dataset_splits.csv"


def list_metadata_files(metadata_root: str | Path) -> list[Path]:
    root = Path(metadata_root)
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_file())


def count_images(image_root: str | Path) -> int:
    root = Path(image_root)
    if not root.exists():
        return 0
    return sum(
        1
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    )


def build_image_index(image_root: str | Path) -> dict[str, Path]:
    root = Path(image_root)
    if not root.exists():
        return {}

    image_index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        key = path.name.casefold()
        if key in image_index:
            duplicates.setdefault(key, [image_index[key]]).append(path)
            continue
        image_index[key] = path

    if duplicates:
        details = "; ".join(
            f"{name}: {', '.join(str(p) for p in paths[:3])}"
            for name, paths in sorted(duplicates.items())
        )
        raise ValueError(f"Duplicate image filenames detected under {root}: {details}")

    return image_index


def _image_name_only(raw_value: str) -> str:
    value = str(raw_value).strip()
    if "\\" in value:
        return PureWindowsPath(value).name
    return Path(value).name


def resolve_split_dataframe(
    split_df: pd.DataFrame,
    metadata_root: str | Path,
    image_root: str | Path,
    return_stats: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, ImageResolutionStats]:
    roots = resolve_dataset_roots(metadata_root=metadata_root, image_root=image_root)
    df = split_df.copy()
    if "image_path" not in df.columns:
        raise ValueError("Split CSV must contain an image_path column")

    start = perf_counter()
    image_index = build_image_index(roots.image_root)
    resolved_paths: list[str] = []
    missing_paths: list[str] = []

    for raw_value in df["image_path"].astype(str).tolist():
        image_name = _image_name_only(raw_value).casefold()
        resolved = image_index.get(image_name)
        if resolved is None:
            missing_paths.append(image_name)
            resolved_paths.append(raw_value)
            continue
        resolved_paths.append(str(resolved))

    resolution_time_sec = perf_counter() - start

    if missing_paths:
        sample = ", ".join(missing_paths[:5])
        raise FileNotFoundError(f"Could not resolve {len(missing_paths)} image paths. Sample: {sample}")

    df["image_path"] = resolved_paths
    stats = ImageResolutionStats(
        indexed_images=len(image_index),
        resolution_time_sec=resolution_time_sec,
        missing_images=len(missing_paths),
    )
    if return_stats:
        return df, stats
    return df


def validate_images(image_paths: Iterable[str | Path]) -> None:
    bad_paths: list[str] = []
    for raw_path in image_paths:
        path = Path(raw_path)
        if not path.exists():
            bad_paths.append(str(path))
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            bad_paths.append(str(path))

    if bad_paths:
        sample = ", ".join(bad_paths[:5])
        raise ValueError(f"Image validation failed for {len(bad_paths)} files. Sample: {sample}")
