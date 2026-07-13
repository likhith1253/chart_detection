"""
Disk-backed OCR cache keyed by image hash.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class OCRCache:
    """Stores OCR outputs as JSON files under cache/ocr/."""

    def __init__(self, root_dir: str | Path = "cache", subdir: str = "ocr") -> None:
        self.root_dir = Path(root_dir)
        self.cache_dir = self.root_dir / subdir
        self.metadata_path = self.root_dir / "metadata.json"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        if self.metadata_path.exists():
            try:
                return json.loads(self.metadata_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hits": 0,
            "misses": 0,
            "entries": 0,
        }

    def _persist_metadata(self) -> None:
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(json.dumps(self._metadata, indent=2), encoding="utf-8")

    @staticmethod
    def image_hash(image_path: str | Path, salt: str = "") -> str:
        digest = hashlib.sha256()
        with Path(image_path).open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        if salt:
            digest.update(salt.encode("utf-8"))
        return digest.hexdigest()

    def _entry_path(self, image_hash: str) -> Path:
        return self.cache_dir / f"{image_hash}.json"

    def get(self, image_hash: str) -> Optional[Dict[str, Any]]:
        path = self._entry_path(image_hash)
        with self._lock:
            if not path.exists():
                self._metadata["misses"] = int(self._metadata.get("misses", 0)) + 1
                self._persist_metadata()
                return None
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                self._metadata["hits"] = int(self._metadata.get("hits", 0)) + 1
                self._persist_metadata()
                return payload
            except Exception:
                self._metadata["misses"] = int(self._metadata.get("misses", 0)) + 1
                self._persist_metadata()
                return None

    def set(self, image_hash: str, payload: Dict[str, Any]) -> Path:
        path = self._entry_path(image_hash)
        with self._lock:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._metadata["entries"] = len(list(self.cache_dir.glob("*.json")))
            self._metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._persist_metadata()
        return path

    def invalidate(self) -> None:
        with self._lock:
            for file_path in self.cache_dir.glob("*.json"):
                file_path.unlink(missing_ok=True)
            self._metadata = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "hits": 0,
                "misses": 0,
                "entries": 0,
                "invalidated_at": datetime.now(timezone.utc).isoformat(),
            }
            self._persist_metadata()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._metadata)
