"""
Structured logging helpers for experiment runs.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from src.pipeline.config import LoggingConfig


class JsonFormatter(logging.Formatter):
    """Minimal JSON lines formatter for reproducible logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("event", "image_name", "stage", "engine", "error"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=True)


def setup_logging(config: LoggingConfig, logs_dir: str | Path) -> Path:
    """Configure root logger for both console and file outputs."""
    log_dir = Path(logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / config.file_name
    json_log_path = log_dir / "pipeline.jsonl"

    level = getattr(logging, config.level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    text_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(text_formatter)
    stream_handler.setLevel(level)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(text_formatter)
    file_handler.setLevel(level)
    root.addHandler(file_handler)

    if config.structured_json:
        json_handler = logging.FileHandler(json_log_path, mode="w", encoding="utf-8")
        json_handler.setFormatter(JsonFormatter())
        json_handler.setLevel(level)
        root.addHandler(json_handler)

    return log_path
