from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from src.parser_engine import ParserResult
from src.parsers.scatter import ScatterChartParser, get_scatter_parser
from src.router import get_parser_registry


def _scatter_image() -> Image.Image:
    img = Image.new("RGB", (160, 120), "white")
    draw = ImageDraw.Draw(img)
    for pt in [(20, 95), (35, 80), (55, 70), (75, 55), (95, 45), (120, 30)]:
        draw.ellipse((pt[0] - 3, pt[1] - 3, pt[0] + 3, pt[1] + 3), fill="black")
    for pt in [(25, 90), (40, 76), (62, 63), (84, 52), (105, 38)]:
        draw.rectangle((pt[0] - 2, pt[1] - 2, pt[0] + 2, pt[1] + 2), fill="black")
    return img


def test_scatter_parser_auto_registers() -> None:
    registry = get_parser_registry()
    assert registry.has("scatter")
    assert registry.get("scatter").parser_name == "scatter"


def test_scatter_parser_returns_parser_result() -> None:
    parser = get_scatter_parser()
    result = parser.parse(_scatter_image(), metadata={"axis_scale": "linear"})
    assert isinstance(result, ParserResult)
    assert result.chart_type == "scatter"
    assert "markers" in result.structured_data
    assert "series" in result.structured_data
    assert "axes" in result.structured_data


def test_scatter_parser_validation_export_and_debug(tmp_path) -> None:
    parser = ScatterChartParser()
    result = parser.parse(_scatter_image())
    assert parser.validate(result)
    out = parser.export(result, tmp_path / "scatter_result.txt")
    assert out.exists()
    overlay = parser.debug_overlay(_scatter_image(), result, tmp_path / "overlay.png")
    assert Path(overlay).exists()


def test_scatter_parser_structured_output_contains_points() -> None:
    parser = ScatterChartParser()
    result = parser.parse(_scatter_image())
    data = result.structured_data
    assert isinstance(data["points"], list)
    assert isinstance(data["series"], list)
    assert "plot_area" in data

