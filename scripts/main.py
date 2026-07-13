"""
Entry point for the research-grade Chart Understanding Framework.
"""

from __future__ import annotations

import argparse
import json
import logging

from src.pipeline.config import load_pipeline_config
from src.pipeline.logging_utils import setup_logging
from src.pipeline.research_pipeline import ResearchPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run chart understanding research pipeline.")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Path to YAML config file.")
    parser.add_argument("--invalidate-cache", action="store_true", help="Invalidate OCR cache before run.")
    parser.add_argument("--no-ablation", action="store_true", help="Disable ablation study for this run.")
    return parser.parse_args()

def run_pipeline(force_recompute: bool | None = None):
    """
    Compatibility wrapper so legacy scripts can still call run_pipeline().
    """

    cfg = load_pipeline_config("configs/pipeline.yaml")

    if force_recompute:
        cfg.cache.invalidate = True

    setup_logging(cfg.logging, cfg.outputs.logs_dir)

    logger = logging.getLogger(__name__)
    logger.info("Running pipeline through run_pipeline() compatibility wrapper")

    pipeline = ResearchPipeline(cfg)
    outputs = pipeline.run()

    return outputs

def main() -> int:
    args = parse_args()
    cfg = load_pipeline_config(args.config)
    if args.invalidate_cache:
        cfg.cache.invalidate = True
    if args.no_ablation:
        cfg.run_ablation = False
        cfg.ablation.enabled = False

    setup_logging(cfg.logging, cfg.outputs.logs_dir)
    logger = logging.getLogger(__name__)

    logger.info("Starting Chart Understanding research pipeline")
    pipeline = ResearchPipeline(cfg)
    outputs = pipeline.run()
    logger.info("Pipeline outputs:\n%s", json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
