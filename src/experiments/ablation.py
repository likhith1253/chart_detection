"""
Ablation study helpers for automated experimental comparisons.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List

import pandas as pd

from src.pipeline.config import PipelineConfig


def apply_variant(base_cfg: PipelineConfig, variant: Dict[str, Any]) -> PipelineConfig:
    cfg = copy.deepcopy(base_cfg)
    force_engine = variant.get("ocr_force_engine", None)
    prep_enabled = variant.get("preprocessing_enabled", cfg.preprocessing.enabled)
    cfg.ocr.force_engine = force_engine
    cfg.preprocessing.enabled = bool(prep_enabled)
    return cfg


class AblationRunner:
    """Executes variants and produces a comparison table."""

    def __init__(self, base_cfg: PipelineConfig, variants: List[Dict[str, Any]]):
        self.base_cfg = base_cfg
        self.variants = variants

    def run(
        self,
        execute_fn: Callable[[str, PipelineConfig], Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for variant in self.variants:
            name = str(variant.get("name", "variant"))
            cfg = apply_variant(self.base_cfg, variant)
            output = execute_fn(name, cfg)
            rows.append({"variant": name, "config": asdict(cfg), **output})
        return rows

    @staticmethod
    def save_table(rows: List[Dict[str, Any]], csv_path: str | Path) -> Path:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            pd.DataFrame().to_csv(path, index=False)
            return path

        flat_rows = []
        for row in rows:
            classification = row.get("classification", {})
            ocr = row.get("ocr", {})
            extraction = row.get("value_extraction", {})
            flat_rows.append(
                {
                    "variant": row.get("variant"),
                    "classification_accuracy": classification.get("accuracy", 0.0),
                    "classification_f1": classification.get("f1_score", 0.0),
                    "ocr_average_confidence": ocr.get("average_ocr_confidence", 0.0),
                    "ocr_token_recall": ocr.get("token_recall", 0.0),
                    "value_extraction_success_rate": extraction.get("extraction_success_rate", 0.0),
                    "numeric_extraction_accuracy": extraction.get("numeric_extraction_accuracy", 0.0),
                    "mean_pipeline_time_sec": row.get("mean_pipeline_time_sec", 0.0),
                }
            )

        pd.DataFrame(flat_rows).to_csv(path, index=False)
        return path
