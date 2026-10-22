from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def clamp(self, width: int, height: int) -> "BoundingBox":
        x1 = max(0, min(width - 1, int(self.x1)))
        y1 = max(0, min(height - 1, int(self.y1)))
        x2 = max(0, min(width, int(self.x2)))
        y2 = max(0, min(height, int(self.y2)))
        if x2 <= x1:
            x2 = min(width, x1 + 1)
        if y2 <= y1:
            y2 = min(height, y1 + 1)
        return BoundingBox(x1, y1, x2, y2)

    def as_list(self) -> list[int]:
        return [int(self.x1), int(self.y1), int(self.x2), int(self.y2)]


def load_image(path: str | Path) -> Image.Image:
    return Image.open(Path(path)).convert("RGB")


def image_size(path: str | Path) -> tuple[int, int]:
    with load_image(path) as img:
        return img.size


def crop_image(image: Image.Image, box: Sequence[int]) -> Image.Image:
    if len(box) != 4:
        raise ValueError("Bounding box must contain four coordinates")
    x1, y1, x2, y2 = map(int, box)
    width, height = image.size
    clipped = BoundingBox(x1, y1, x2, y2).clamp(width, height)
    return image.crop((clipped.x1, clipped.y1, clipped.x2, clipped.y2))


def iter_image_files(root: str | Path) -> Iterator[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return iter(())
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return (p for p in root_path.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def md5_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def ensure_numpy_image(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    return np.asarray(image)

