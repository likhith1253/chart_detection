from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from src.parser_engine import ParserResult
from src.parsers.line import LineChartParser, get_line_parser
from src.router import get_parser_registry


def _line_image() -> Image.Image:
    img = Image.new("RGB", (160, 120), "white")
    draw = ImageDraw.Draw(img)
    draw.line((20, 90, 60, 70, 100, 50, 140, 30), fill="black", width=2)
    draw.ellipse((55, 65, 65, 75), fill="black")
    draw.ellipse((95, 45, 105, 55), fill="black")
    return img


def test_line_parser_auto_registers() -> None:
    registry = get_parser_registry()
    assert registry.has("line")
    assert registry.get("line").parser_name == "line"


def test_line_parser_returns_parser_result() -> None:
    parser = get_line_parser()
    result = parser.parse(_line_image(), metadata={"axis_scale": "linear"})
    assert isinstance(result, ParserResult)
    assert result.chart_type == "line"
    assert "curves" in result.structured_data
    assert "series" in result.structured_data
    assert "axes" in result.structured_data
    assert result.provenance["parser_name"] == "line"


def test_line_parser_validation_export_and_debug(tmp_path) -> None:
    parser = LineChartParser()
    result = parser.parse(_line_image())
    assert parser.validate(result)
    out = parser.export(result, tmp_path / "line_result.txt")
    assert out.exists()
    overlay = parser.debug_overlay(_line_image(), result, tmp_path / "overlay.png")
    assert Path(overlay).exists()


def test_line_parser_structured_output_contains_points() -> None:
    parser = LineChartParser()
    result = parser.parse(_line_image())
    data = result.structured_data
    assert isinstance(data["points"], list)
    assert isinstance(data["series"], list)
    assert "plot_area" in data

