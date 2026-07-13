"""
Unified multi-dataset loader for chart understanding research.

Supported datasets:
- PlotQA
- ChartQA
- DVQA
- FigureQA

The loader maps heterogeneous directory structures into a standard schema and
produces cleaned, balanced, stratified train/validation/test manifests.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import config

logger = logging.getLogger(__name__)


STANDARD_LABELS = [
    "bar_chart",
    "histogram",
    "line_chart",
    "scatter_plot",
    "pie_chart",
]

_LABEL_SYNONYMS = {
    "bar": "bar_chart",
    "bar_chart": "bar_chart",
    "vertical_bar": "bar_chart",
    "horizontal_bar": "bar_chart",
    "vbar": "bar_chart",
    "hbar": "bar_chart",
    "hist": "histogram",
    "histogram": "histogram",
    "line": "line_chart",
    "line_chart": "line_chart",
    "dot_line": "line_chart",
    "scatter": "scatter_plot",
    "scatter_plot": "scatter_plot",
    "dot": "scatter_plot",
    "point": "scatter_plot",
    "pie": "pie_chart",
    "pie_chart": "pie_chart",
    "donut": "pie_chart",
}

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_SUPPORTED_DATASETS = ("plotqa", "chartqa", "dvqa", "figureqa")


def normalize_label(label: str) -> str:
    raw = str(label or "").strip().lower()
    if not raw:
        return "unknown"
    if raw in _LABEL_SYNONYMS:
        return _LABEL_SYNONYMS[raw]
    compact = raw.replace("-", "_").replace(" ", "_")
    if compact in _LABEL_SYNONYMS:
        return _LABEL_SYNONYMS[compact]
    return "unknown"


def infer_label_from_name(name_or_path: str) -> str:
    text = str(name_or_path).lower()
    tokens = re.split(r"[^a-z0-9]+", text)
    ordered_tokens = [t for t in tokens if t]

    # Preserve histogram before bar to avoid accidental bar capture in text.
    priorities = [
        ("hist", "histogram"),
        ("histogram", "histogram"),
        ("scatter", "scatter_plot"),
        ("dot", "scatter_plot"),
        ("point", "scatter_plot"),
        ("pie", "pie_chart"),
        ("donut", "pie_chart"),
        ("line", "line_chart"),
        ("dot_line", "line_chart"),
        ("vbar", "bar_chart"),
        ("hbar", "bar_chart"),
        ("bar", "bar_chart"),
    ]
    joined = "_".join(ordered_tokens)
    for key, target in priorities:
        if key in ordered_tokens or key in joined:
            return target
    return "unknown"


def _infer_split_from_path(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    if "train" in parts:
        return "train"
    if "val" in parts or "valid" in parts or "validation" in parts:
        return "validation"
    if "test" in parts:
        return "test"
    return "unspecified"


@dataclass
class UnifiedLoaderConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42
    balance_classes: bool = True


class UnifiedChartDatasetLoader:
    """Load and standardize chart datasets into a single schema."""

    def __init__(
        self,
        datasets_root: Path | None = None,
        raw_root: Path | None = None,
        output_dir: Path | None = None,
    ):
        self.datasets_root = datasets_root or config.DATASETS_DIR
        self.raw_root = raw_root or config.RAW_IMAGE_DIR
        self.output_dir = output_dir or config.DATASETS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _iter_images(root: Path) -> Iterable[Path]:
        if not root.exists():
            return []
        return (
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        )

    @staticmethod
    def _image_size(path: Path) -> Tuple[int, int]:
        try:
            with Image.open(path) as img:
                w, h = img.size
            return int(w), int(h)
        except Exception:
            return 0, 0

    def _collect_dataset_records(self, dataset_name: str) -> List[Dict]:
        ds_root = self.datasets_root / dataset_name
        if not ds_root.exists():
            return []

        rows: List[Dict] = []
        for idx, img_path in enumerate(self._iter_images(ds_root)):
            inferred = infer_label_from_name(str(img_path))
            width, height = self._image_size(img_path)
            rows.append(
                {
                    "image_id": f"{dataset_name}_{idx:07d}",
                    "image_name": img_path.name,
                    "image_path": str(img_path.resolve()),
                    "dataset_name": dataset_name,
                    "original_split": _infer_split_from_path(img_path),
                    "label_raw": inferred,
                    "label": normalize_label(inferred),
                    "width": width,
                    "height": height,
                    "file_size": int(img_path.stat().st_size),
                    "label_source": "filename_heuristic",
                }
            )
        return rows

    def _collect_local_raw_records(self) -> List[Dict]:
        if not self.raw_root.exists():
            return []
        rows: List[Dict] = []
        for idx, img_path in enumerate(self._iter_images(self.raw_root)):
            inferred = infer_label_from_name(img_path.name)
            width, height = self._image_size(img_path)
            rows.append(
                {
                    "image_id": f"local_raw_{idx:07d}",
                    "image_name": img_path.name,
                    "image_path": str(img_path.resolve()),
                    "dataset_name": "local_raw",
                    "original_split": "unspecified",
                    "label_raw": inferred,
                    "label": normalize_label(inferred),
                    "width": width,
                    "height": height,
                    "file_size": int(img_path.stat().st_size),
                    "label_source": "filename_heuristic",
                }
            )
        return rows

    def collect_unified_records(self) -> pd.DataFrame:
        rows: List[Dict] = []
        for dataset_name in _SUPPORTED_DATASETS:
            rows.extend(self._collect_dataset_records(dataset_name))
        rows.extend(self._collect_local_raw_records())

        if not rows:
            raise RuntimeError("No chart images discovered across configured dataset roots")

        df = pd.DataFrame(rows)
        df["label"] = df["label"].map(normalize_label)
        df["is_known_label"] = df["label"].isin(STANDARD_LABELS).astype(int)
        return df

    @staticmethod
    def _balanced_subset(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
        counts = df["label"].value_counts().to_dict()
        min_count = int(min(counts.values())) if counts else 0
        if min_count == 0:
            return df

        balanced = (
            df.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(n=min_count, random_state=random_state))
            .reset_index(drop=True)
        )
        return balanced

    @staticmethod
    def _safe_stratified_split(
        df: pd.DataFrame,
        cfg: UnifiedLoaderConfig,
    ) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        if len(df["label"].unique()) < 2:
            out = df.copy()
            out["split"] = "train"
            return out

        try:
            train_df, temp_df = train_test_split(
                df,
                test_size=(1.0 - cfg.train_ratio),
                random_state=cfg.random_state,
                stratify=df["label"],
            )

            # Re-normalize val split inside temp.
            val_share = cfg.val_ratio / (cfg.val_ratio + cfg.test_ratio)
            val_df, test_df = train_test_split(
                temp_df,
                test_size=(1.0 - val_share),
                random_state=cfg.random_state,
                stratify=temp_df["label"],
            )
        except Exception:
            # Fallback when classes are too small for strict stratification.
            shuffled = df.sample(frac=1.0, random_state=cfg.random_state).reset_index(drop=True)
            n = len(shuffled)
            n_train = int(n * cfg.train_ratio)
            n_val = int(n * cfg.val_ratio)
            train_df = shuffled.iloc[:n_train]
            val_df = shuffled.iloc[n_train : n_train + n_val]
            test_df = shuffled.iloc[n_train + n_val :]

        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        train_df["split"] = "train"
        val_df["split"] = "validation"
        test_df["split"] = "test"
        return pd.concat([train_df, val_df, test_df], ignore_index=True)

    def build_manifests(self, cfg: UnifiedLoaderConfig | None = None) -> Dict:
        cfg = cfg or UnifiedLoaderConfig()
        raw_df = self.collect_unified_records()
        cleaned = raw_df[raw_df["is_known_label"] == 1].copy()

        if cfg.balance_classes:
            cleaned = self._balanced_subset(cleaned, random_state=cfg.random_state)

        split_df = self._safe_stratified_split(cleaned, cfg=cfg)

        unified_csv = self.output_dir / "unified_dataset_manifest.csv"
        splits_csv = self.output_dir / "unified_dataset_splits.csv"
        dist_csv = self.output_dir / "unified_dataset_distribution.csv"
        summary_json = self.output_dir / "unified_dataset_summary.json"

        raw_df.to_csv(unified_csv, index=False)
        split_df.to_csv(splits_csv, index=False)

        dist = (
            split_df.groupby(["split", "label"])
            .size()
            .reset_index(name="count")
            .sort_values(["split", "label"])
        )
        dist.to_csv(dist_csv, index=False)

        class_counts = split_df["label"].value_counts().to_dict()
        summary = {
            "total_discovered": int(len(raw_df)),
            "known_label_count": int(len(cleaned)),
            "removed_unknown_count": int((raw_df["is_known_label"] == 0).sum()),
            "class_counts_after_balance": {k: int(v) for k, v in class_counts.items()},
            "splits": {
                "train": int((split_df["split"] == "train").sum()),
                "validation": int((split_df["split"] == "validation").sum()),
                "test": int((split_df["split"] == "test").sum()),
            },
            "datasets_present": sorted(split_df["dataset_name"].dropna().unique().tolist()),
            "labels": STANDARD_LABELS,
            "manifests": {
                "unified_dataset_manifest_csv": str(unified_csv),
                "unified_dataset_splits_csv": str(splits_csv),
                "unified_dataset_distribution_csv": str(dist_csv),
            },
        }
        summary_json.write_text(json.dumps(summary, indent=2))

        logger.info("Unified dataset manifests written to %s", self.output_dir)
        return {
            "raw": raw_df,
            "splits": split_df,
            "summary": summary,
            "summary_path": summary_json,
        }
