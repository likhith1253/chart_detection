from __future__ import annotations

from pathlib import Path
from typing import Any


def save_result_text(output_path: str | Path, content: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path

