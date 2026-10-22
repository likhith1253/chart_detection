from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

from .prediction_record import DetectionPrediction


def draw_predictions(image: Image.Image, predictions: Iterable[DetectionPrediction]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for pred in predictions:
        x1, y1, x2, y2 = pred.bbox
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        label = f"{pred.class_name} {pred.confidence:.2f}"
        draw.text((x1 + 2, max(0, y1 - 12)), label, fill="red")
    return annotated


class DetectionVisualizer:
    def save_annotated_page(self, image: Image.Image, predictions: Iterable[DetectionPrediction], output_path: str | Path) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        annotated = draw_predictions(image, predictions)
        annotated.save(out)
        return out

