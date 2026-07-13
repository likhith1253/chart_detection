from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running this file directly from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.dataset_paths import build_image_resolution_index, _image_name_only, resolve_dataset_roots


def main() -> int:
    parser = argparse.ArgumentParser(description="Check chart image resolution coverage.")
    parser.add_argument("--metadata-root", default="data/datasets", help="Root containing original dataset folders.")
    parser.add_argument("--image-root", default="data/raw_images", help="Primary raw image directory.")
    parser.add_argument("--split-csv", default=None, help="CSV with an image_path column to validate.")
    args = parser.parse_args()

    roots = resolve_dataset_roots(args.metadata_root, args.image_root)
    index = build_image_resolution_index(roots.metadata_root, roots.image_root)

    if args.split_csv is None:
        split_csv = roots.metadata_root / "unified_dataset_splits.csv"
    else:
        split_csv = Path(args.split_csv)

    if not split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")

    df = pd.read_csv(split_csv)
    if "image_path" not in df.columns:
        raise ValueError("Split CSV must contain an image_path column")

    raw_resolved: list[str] = []
    original_resolved: list[str] = []
    missing: list[str] = []

    for raw_value in df["image_path"].astype(str).tolist():
        name = _image_name_only(raw_value).casefold()
        if name in index.raw_images:
            raw_resolved.append(name)
        elif name in index.original_datasets:
            original_resolved.append(name)
        else:
            missing.append(name)

    print("Images resolved from raw_images:")
    for name in raw_resolved:
        print(name)

    print("Images resolved from original datasets:")
    for name in original_resolved:
        print(name)

    print("Missing images:")
    for name in missing:
        print(name)

    print("PASS" if not missing else "FAIL")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
