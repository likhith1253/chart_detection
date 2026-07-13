#!/usr/bin/env python3
"""CLI entry point for synthetic dataset generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.constants import DEFAULT_CONFIG_PATH
from src.config.settings import load_config
from src.data.dataset import DatasetGenerator, validate_dataset_integrity
from src.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic chart dataset for research experiments.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Optional project root override.",
    )
    return parser.parse_args()


def main() -> int:
    """Run dataset generation."""
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    project_root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    config = load_config(config_path, project_root=project_root)
    logger = setup_logging(config)
    logger.info("Starting dataset generation with config: %s", config_path)

    generator = DatasetGenerator(config)
    summary = generator.generate()
    validation = validate_dataset_integrity(config, summary)

    report = {
        "total_images": summary.total_images,
        "chart_type_counts": summary.chart_type_counts,
        "labels_csv": str(summary.labels_csv_path),
        "fingerprint_path": str(summary.fingerprint_path),
        "validation": validation,
    }
    print(json.dumps(report, indent=2))

    if not validation["passed"]:
        logger.error("Dataset validation failed: %s", validation)
        return 1

    logger.info("Dataset generation completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
