from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from PIL import Image


class ArtifactManager:
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save_json(self, name: str, payload: Any) -> Path:
        path = self.root_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self._serialize(payload), handle, indent=2, sort_keys=True)
        return path

    def save_text(self, name: str, content: str) -> Path:
        path = self.root_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def save_image(self, name: str, image: Image.Image) -> Path:
        path = self.root_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return path

    def _serialize(self, payload: Any) -> Any:
        if is_dataclass(payload):
            return asdict(payload)
        if isinstance(payload, Path):
            return str(payload)
        if isinstance(payload, Image.Image):
            return {
                "mode": payload.mode,
                "size": list(payload.size),
            }
        if isinstance(payload, dict):
            return {k: self._serialize(v) for k, v in payload.items()}
        if isinstance(payload, list):
            return [self._serialize(v) for v in payload]
        if isinstance(payload, tuple):
            return [self._serialize(v) for v in payload]
        return payload
