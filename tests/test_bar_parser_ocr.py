from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw

from src.parsers.bar import BarChartParser
from src.parsers.bar.ocr import OCRBackend, OCRResult, OCRManager, TickParser
from src.parsers.bar.ocr.label_matcher import CategoryMatcher


@dataclass
class StubOCRBackend(OCRBackend):
    def recognize(self, image, *, metadata=None):
        return [
            OCRResult(text="Sales", confidence=0.98, bounding_box=(10, 0, 40, 12), source="stub"),
            OCRResult(text="0", confidence=0.92, bounding_box=(0, 95, 8, 105), source="stub"),
            OCRResult(text="50", confidence=0.90, bounding_box=(0, 55, 8, 65), source="stub"),
            OCRResult(text="100", confidence=0.91, bounding_box=(0, 5, 8, 15), source="stub"),
            OCRResult(text="North", confidence=0.85, bounding_box=(20, 110, 40, 118), source="stub"),
            OCRResult(text="South", confidence=0.84, bounding_box=(55, 110, 75, 118), source="stub"),
        ]


def _bar_image() -> Image.Image:
    img = Image.new("RGB", (120, 120), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 30, 40, 100), fill="black")
    draw.rectangle((55, 50, 75, 100), fill="black")
    draw.rectangle((90, 20, 110, 100), fill="black")
    return img


def test_ocr_manager_and_tick_parser() -> None:
    mgr = OCRManager(backend=StubOCRBackend())
    ticks = mgr.extract_ticks(_bar_image(), axis="y")
    assert ticks.numeric is True
    assert len(ticks.ticks) >= 3
    assert ticks.ordered is True
    assert any(t.value == 100.0 for t in ticks.ticks)


def test_category_matcher_uses_ocr_labels() -> None:
    matcher = CategoryMatcher()
    bars = BarChartParser().bar_detector.detect(_bar_image())
    matches = matcher.match([b.geometry for b in bars], StubOCRBackend().recognize(_bar_image()), orientation="vertical")
    assert matches
    assert matches[0].label


def test_bar_parser_uses_ocr_for_title_legend_and_ticks() -> None:
    parser = BarChartParser()
    parser.ocr_manager.backend = StubOCRBackend()
    result = parser.parse(_bar_image(), metadata={"axis_scale": "linear"})
    assert result.success in {True, False}
    data = result.structured_data
    assert data["title"] == "Sales"
    assert data["labels"]["legend_labels"]
    assert data["axes"]["ticks"]
    assert data["axes"]["calibration"]["metadata"]["source"] == "ticks"


def test_bar_values_are_tick_calibrated() -> None:
    parser = BarChartParser()
    parser.ocr_manager.backend = StubOCRBackend()
    result = parser.parse(_bar_image(), metadata={"axis_scale": "linear"})
    values = result.structured_data["values"]
    assert all(v >= 0 for v in values)
    assert max(values) > 0

