from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from src.detection.utils import BoundingBox, crop_image, load_image
from .prediction_record import DetectionPrediction


@dataclass(frozen=True)
class ChartCrop:
    crop_image: Image.Image
    original_image_path: Path
    bbox: list[int]
    scale_information: dict[str, float] = field(default_factory=dict)
    transform_information: dict[str, Any] = field(default_factory=dict)


def crop_chart_regions(
    image: Image.Image | str | Path,
    predictions: Iterable[DetectionPrediction],
    save_dir: str | Path | None = None,
) -> list[ChartCrop]:
    img = load_image(image) if not isinstance(image, Image.Image) else image
    source_path = Path(image) if isinstance(image, (str, Path)) else Path("")
    crops: list[ChartCrop] = []
    save_root = Path(save_dir) if save_dir is not None else None
    if save_root is not None:
        save_root.mkdir(parents=True, exist_ok=True)

    for idx, pred in enumerate(predictions):
        crop = crop_image(img, pred.bbox)
        crop_path = None
        if save_root is not None:
            crop_path = save_root / f"{source_path.stem or 'image'}_{idx:04d}.png"
            crop.save(crop_path)
        crops.append(
            ChartCrop(
                crop_image=crop,
                original_image_path=source_path,
                bbox=list(pred.bbox),
                scale_information={"width": float(img.width), "height": float(img.height)},
                transform_information={"saved_path": str(crop_path) if crop_path else None},
            )
        )
    return crops

