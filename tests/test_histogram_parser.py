from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from src.parser_engine import ParserResult
from src.parsers.histogram import HistogramChartParser, get_histogram_parser
from src.router import get_parser_registry


def _histogram_image() -> Image.Image:
    img = Image.new("RGB", (180, 120), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 75, 40, 110), fill="black")
    draw.rectangle((40, 60, 62, 110), fill="black")
    draw.rectangle((62, 45, 84, 110), fill="black")
    draw.rectangle((84, 55, 110, 110), fill="black")
    draw.rectangle((110, 70, 135, 110), fill="black")
    return img


def test_histogram_parser_auto_registers() -> None:
    registry = get_parser_registry()
    assert registry.has("histogram")
    assert registry.get("histogram").parser_name == "histogram"


def test_histogram_parser_returns_parser_result() -> None:
    parser = get_histogram_parser()
    result = parser.parse(_histogram_image(), metadata={"axis_scale": "linear"})
    assert isinstance(result, ParserResult)
    assert result.chart_type == "histogram"
    assert "bins" in result.structured_data
    assert "distribution" in result.structured_data
    assert "axes" in result.structured_data


def test_histogram_parser_validation_export_and_debug(tmp_path) -> None:
    parser = HistogramChartParser()
    result = parser.parse(_histogram_image())
    assert parser.validate(result)
    out = parser.export(result, tmp_path / "histogram_result.txt")
    assert out.exists()
    overlay = parser.debug_overlay(_histogram_image(), result, tmp_path / "overlay.png")
    assert Path(overlay).exists()


def test_histogram_parser_extension_hooks() -> None:
    parser = HistogramChartParser()
    result = parser.parse(_histogram_image())
    bins = result.structured_data["bins"]
    assert parser.reconstruct_distribution([], method="kde")["method"] == "kde"
    assert parser.fit_gaussian([])["fitted"] is False
    assert parser.fit_mixture([])["fitted"] is False

