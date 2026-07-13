"""Logging utilities for the research project."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.config.settings import LoggingConfig, ProjectConfig, resolve_path


def setup_logging(config: ProjectConfig) -> logging.Logger:
    """
    Configure application logging from project configuration.

    Args:
        config: Loaded project configuration.

    Returns:
        Root application logger.
    """
    logging_config = config.logging
    level = getattr(logging, logging_config.level.upper(), logging.INFO)

    root_logger = logging.getLogger("chart_research")
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    root_logger.propagate = False

    formatter = logging.Formatter(logging_config.format)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    if logging_config.file_name:
        log_path = resolve_path(config, logging_config.file_name)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the chart_research hierarchy."""
    return logging.getLogger(f"chart_research.{name}")
