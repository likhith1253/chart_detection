from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

from PIL import Image

from .utils import BoundingBox, crop_image, load_image


@dataclass(frozen=True)
class DetectionSample:
    image_id: str
    image_path: Path
    bbox: BoundingBox
    class_name: str = "chart"
    split: str = "train"
    source: str = "benchmark"


class DetectionDataset:
    def __init__(self, samples: Sequence[DetectionSample], transforms: Callable[[Image.Image], Image.Image] | None = None):
        self.samples = list(samples)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[DetectionSample]:
        return iter(self.samples)

    def __getitem__(self, index: int) -> tuple[Image.Image, DetectionSample]:
        sample = self.samples[index]
        image = load_image(sample.image_path)
        if self.transforms is not None:
            image = self.transforms(image)
        return image, sample

    def crop(self, index: int) -> Image.Image:
        image, sample = self[index]
        return crop_image(image, sample.bbox.as_list())


def _parse_bbox(raw: str) -> BoundingBox:
    parts = [int(float(x)) for x in raw.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Expected bbox with four values, got {raw!r}")
    return BoundingBox(*parts)


def load_detection_dataset(path: str | Path, split: str | None = None) -> DetectionDataset:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    samples: list[DetectionSample] = []
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"image_id", "image_path", "bbox"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Detection dataset is missing required columns: {sorted(missing)}")
        for row in reader:
            row_split = row.get("split", "train") or "train"
            if split is not None and row_split != split:
                continue
            samples.append(
                DetectionSample(
                    image_id=row["image_id"],
                    image_path=Path(row["image_path"]),
                    bbox=_parse_bbox(row["bbox"]),
                    class_name=row.get("class", "chart") or "chart",
                    split=row_split,
                    source=row.get("source", "benchmark") or "benchmark",
                )
            )
    return DetectionDataset(samples=samples)
