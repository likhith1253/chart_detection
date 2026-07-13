"""
Experiment reproducibility metadata utilities.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.pipeline.config import PipelineConfig


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def _safe_version(package_name: str) -> str:
    try:
        from importlib.metadata import version

        return version(package_name)
    except Exception:
        return "not-installed"


def build_experiment_snapshot(
    cfg: PipelineConfig,
    dataset_fingerprint: str,
) -> Dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": cfg.seed,
        "git_commit_hash": _get_git_commit(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "library_versions": {
            "numpy": _safe_version("numpy"),
            "pandas": _safe_version("pandas"),
            "opencv-python": _safe_version("opencv-python"),
            "scikit-learn": _safe_version("scikit-learn"),
            "easyocr": _safe_version("easyocr"),
            "paddleocr": _safe_version("paddleocr"),
            "ultralytics": _safe_version("ultralytics"),
            "matplotlib": _safe_version("matplotlib"),
            "seaborn": _safe_version("seaborn"),
            "pyyaml": _safe_version("PyYAML"),
        },
        "dataset_fingerprint": dataset_fingerprint,
        "configuration_snapshot": asdict(cfg),
    }


def save_experiment_snapshot(snapshot: Dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return output_path
