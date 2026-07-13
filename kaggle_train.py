from __future__ import annotations

import argparse
from pathlib import Path

from src.research.cnn_classifier import CNNChartClassifier, TrainConfig
from src.utils.dataset_paths import count_images, default_split_csv, list_metadata_files, resolve_dataset_roots

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-csv", type=str, default="")
    parser.add_argument("--metadata-root", type=str, default="")
    parser.add_argument("--image-root", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="results/cnn_baseline")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=160)
    args = parser.parse_args()

    roots = resolve_dataset_roots(
        metadata_root=args.metadata_root or None,
        image_root=args.image_root or None,
    )
    split_csv = Path(args.split_csv) if args.split_csv else default_split_csv(roots.metadata_root)
    if not split_csv.exists():
        raise FileNotFoundError(
            f"Missing dataset split file: {split_csv}."
        )

    metadata_files = list_metadata_files(roots.metadata_root)
    print(f"Metadata root: {roots.metadata_root}")
    print(f"Image root: {roots.image_root}")
    print(f"Metadata files found: {len(metadata_files)}")
    print(f"Images found: {count_images(roots.image_root)}")

    runner = CNNChartClassifier(output_dir=args.output_dir)
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, image_size=args.image_size)
    result = runner.fit_from_split(
        split_csv,
        cfg,
        metadata_root=roots.metadata_root,
        image_root=roots.image_root,
    )
    print(result["best_val_macro_f1"])


if __name__ == "__main__":
    main()
