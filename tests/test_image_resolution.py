from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.dataset_paths import resolve_split_dataframe


def test_resolve_split_dataframe_uses_image_name_for_plotqa(tmp_path: Path) -> None:
    metadata_root = tmp_path / "datasets"
    image_root = tmp_path / "raw_images"
    plotqa_dir = metadata_root / "plotqa" / "PlotQA-master" / "images"
    plotqa_dir.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)

    filenames = [
        "10_dot_line.png",
        "11_dot_line.png",
        "12_dot_line.png",
        "13_dot_line.png",
        "14_dot_line.png",
        "15_dot_line.png",
    ]
    for name in filenames:
        (plotqa_dir / name).write_bytes(b"fake-image-data")

    df = pd.DataFrame(
        {
            "image_name": filenames,
            "image_path": [f"raw_images/{name}" for name in filenames],
        }
    )

    resolved, stats = resolve_split_dataframe(
        df,
        metadata_root=metadata_root,
        image_root=image_root,
        return_stats=True,
    )

    assert list(resolved["image_path"]) == [str(plotqa_dir / name) for name in filenames]
    assert stats.resolved_from_raw_images == 0
    assert stats.resolved_from_original_datasets == 6
    assert stats.missing_images == 0
