"""
Dataset verification and balancing utilities for research pipeline v3.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List

import cv2
import pandas as pd

import config
from src.dataset.dataset_manager import _GENERATORS

logger = logging.getLogger(__name__)


CHART_LABELS = [
    "bar_chart",
    "histogram",
    "line_chart",
    "scatter_plot",
    "pie_chart",
]

GENERATOR_KEY_BY_LABEL = {
    "bar_chart": "bar",
    "histogram": "hist",
    "line_chart": "line",
    "scatter_plot": "scatter",
    "pie_chart": "pie",
}


def infer_label_from_filename(filename: str) -> str:
    """Infer chart label from filename."""
    name = filename.lower()
    if "bar" in name:
        return "bar_chart"
    if "hist" in name:
        return "histogram"
    if "line" in name:
        return "line_chart"
    if "scatter" in name:
        return "scatter_plot"
    if "pie" in name:
        return "pie_chart"
    return "unknown"


def _next_synthetic_index(raw_dir: Path, generator_key: str) -> int:
    max_idx = -1
    pattern = f"synthetic_{generator_key}_*.png"
    for path in raw_dir.glob(pattern):
        try:
            idx = int(path.stem.split("_")[-1])
            if idx > max_idx:
                max_idx = idx
        except ValueError:
            continue
    return max_idx + 1


def _verify_images(raw_dir: Path) -> List[Dict]:
    rows: List[Dict] = []
    valid_exts = {".png", ".jpg", ".jpeg"}

    for path in sorted(raw_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in valid_exts:
            continue

        label = infer_label_from_filename(path.name)
        img = cv2.imread(str(path))
        is_corrupt = img is None or img.size == 0
        h = int(img.shape[0]) if img is not None else 0
        w = int(img.shape[1]) if img is not None else 0

        rows.append(
            {
                "image_name": path.name,
                "image_path": str(path.resolve()),
                "label": label,
                "is_corrupt": int(is_corrupt),
                "width": w,
                "height": h,
                "file_size": int(path.stat().st_size),
            }
        )
    return rows


def _generate_missing_classes(raw_dir: Path, min_per_class: int) -> Dict[str, int]:
    rows = _verify_images(raw_dir)
    df = pd.DataFrame(rows)
    valid = df[(df["is_corrupt"] == 0) & (df["label"].isin(CHART_LABELS))]
    counts = Counter(valid["label"].tolist())

    generated = {label: 0 for label in CHART_LABELS}
    for label in CHART_LABELS:
        need = max(0, min_per_class - counts.get(label, 0))
        if need == 0:
            continue

        generator_key = GENERATOR_KEY_BY_LABEL[label]
        generator = _GENERATORS[generator_key]
        start_idx = _next_synthetic_index(raw_dir, generator_key)
        logger.info("Generating %s synthetic %s samples for balancing", need, label)

        for offset in range(need):
            idx = start_idx + offset
            out_name = f"synthetic_{generator_key}_{idx:05d}.png"
            out_path = raw_dir / out_name
            try:
                generator(out_path, idx)
                generated[label] += 1
            except Exception as exc:
                logger.warning("Synthetic generation failed for %s: %s", out_name, exc)

    return generated


def verify_and_balance_dataset(
    raw_dir: Path | None = None,
    target_total: int = 3000,
    min_per_class: int = 600,
    remove_corrupt: bool = True,
) -> Dict:
    """
    Verify labels, remove corrupted images, and balance classes with synthetic generation.
    Returns summary dictionary and saves CSV/JSON reports.
    """
    raw_dir = raw_dir or config.RAW_IMAGE_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows = _verify_images(raw_dir)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No images found in {raw_dir}")

    corrupt_files = df[df["is_corrupt"] == 1]["image_name"].tolist()
    if remove_corrupt and corrupt_files:
        for name in corrupt_files:
            try:
                (raw_dir / name).unlink(missing_ok=True)
            except Exception as exc:
                logger.warning("Failed to remove corrupt file %s: %s", name, exc)

    generated = _generate_missing_classes(raw_dir, min_per_class=min_per_class)

    # Re-scan after cleanup/generation.
    rows = _verify_images(raw_dir)
    df = pd.DataFrame(rows)
    df["is_verified"] = ((df["is_corrupt"] == 0) & (df["label"] != "unknown")).astype(int)

    known = df[df["label"].isin(CHART_LABELS) & (df["is_corrupt"] == 0)].copy()
    class_counts = known["label"].value_counts().to_dict()

    # Balanced training manifest uses equal samples per known class.
    balanced_count = int(min(class_counts.values())) if class_counts else 0
    balanced_parts = []
    for label in CHART_LABELS:
        part = known[known["label"] == label].sort_values("image_name").head(balanced_count)
        balanced_parts.append(part)
    balanced_df = pd.concat(balanced_parts, ignore_index=True) if balanced_parts else pd.DataFrame()

    labels_csv = config.DATASETS_DIR / "verified_labels.csv"
    balanced_csv = config.DATASETS_DIR / "balanced_training_manifest.csv"
    report_json = config.RESULT_DIR / "dataset_verification_report.json"

    df.to_csv(labels_csv, index=False)
    balanced_df.to_csv(balanced_csv, index=False)

    total_images = int(len(df))
    summary = {
        "total_images": total_images,
        "target_total": int(target_total),
        "corrupt_removed": int(len(corrupt_files)),
        "generated_for_balance": generated,
        "class_counts_verified": class_counts,
        "unknown_label_count": int((df["label"] == "unknown").sum()),
        "balanced_samples_per_class": int(balanced_count),
        "balanced_manifest_size": int(len(balanced_df)),
        "labels_csv": str(labels_csv),
        "balanced_manifest_csv": str(balanced_csv),
        "meets_target_total": bool(total_images >= target_total),
    }

    report_json.write_text(json.dumps(summary, indent=2))
    logger.info("Dataset verification report written to %s", report_json)
    return summary

