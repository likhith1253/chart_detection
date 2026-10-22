from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.integrate_figureqa_benchmark import _build_leakage_report


def test_duplicate_metadata_audit_uses_full_identity_fields() -> None:
    df = pd.DataFrame(
        [
            {
                "dataset": "figureqa",
                "image_id": "fig_1",
                "image_path": "/tmp/fig_1.png",
                "split": "train",
                "chart_type": "bar",
                "width": 800,
                "height": 400,
            },
            {
                "dataset": "figureqa",
                "image_id": "fig_2",
                "image_path": "/tmp/fig_2.png",
                "split": "train",
                "chart_type": "bar",
                "width": 800,
                "height": 400,
            },
        ]
    )
    report = _build_leakage_report(df)
    assert report["duplicate_metadata_count"] == 0


@pytest.mark.skipif(not Path("data/datasets/benchmark_manifest.csv").exists(), reason="needs the local (git-ignored) benchmark manifest")
def test_duplicate_metadata_audit_is_zero_on_benchmark_manifest() -> None:
    df = pd.read_csv(Path("data/datasets/benchmark_manifest.csv"))
    report = _build_leakage_report(df)
    assert report["duplicate_metadata_count"] == 0

