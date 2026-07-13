from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class DatasetRoots:
    metadata_root: Path
    image_root: Path


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


def _candidate_image_paths(raw_value: str, image_root: Path) -> Iterable[Path]:
    raw_path = Path(str(raw_value).strip())
    if raw_path.is_absolute():
        yield raw_path
        return

    yield raw_path
    yield image_root / raw_path
    yield image_root / raw_path.name


def resolve_split_dataframe(
    split_df: pd.DataFrame,
    metadata_root: str | Path,
    image_root: str | Path,
) -> pd.DataFrame:
    roots = resolve_dataset_roots(metadata_root=metadata_root, image_root=image_root)
    df = split_df.copy()
    if "image_path" not in df.columns:
        raise ValueError("Split CSV must contain an image_path column")

    resolved_paths: list[str] = []
    missing_paths: list[str] = []

    for raw_value in df["image_path"].astype(str).tolist():
        resolved = None
        for candidate in _candidate_image_paths(raw_value, roots.image_root):
            if candidate.exists():
                resolved = candidate
                break
        if resolved is None:
            missing_paths.append(raw_value)
            resolved_paths.append(raw_value)
        else:
            resolved_paths.append(str(resolved))

    if missing_paths:
        sample = ", ".join(missing_paths[:5])
        raise FileNotFoundError(f"Could not resolve {len(missing_paths)} image paths. Sample: {sample}")

    df["image_path"] = resolved_paths
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
