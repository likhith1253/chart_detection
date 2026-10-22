from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from src.parser_engine import ParserResult
from src.parsers.pie import PieChartParser, get_pie_parser
from src.router import get_parser_registry


def _pie_image() -> Image.Image:
    img = Image.new("RGB", (160, 160), "white")
    draw = ImageDraw.Draw(img)
    draw.pieslice((30, 30, 130, 130), 0, 90, fill="red")
    draw.pieslice((30, 30, 130, 130), 90, 180, fill="green")
    draw.pieslice((30, 30, 130, 130), 180, 270, fill="blue")
    draw.pieslice((30, 30, 130, 130), 270, 360, fill="yellow")
    return img


def test_pie_parser_auto_registers() -> None:
    registry = get_parser_registry()
    assert registry.has("pie")
    assert registry.get("pie").parser_name == "pie"


def test_pie_parser_returns_parser_result() -> None:
    parser = get_pie_parser()
    result = parser.parse(_pie_image(), metadata={"title": "Share"})
    assert isinstance(result, ParserResult)
    assert result.chart_type == "pie"
    assert "slices" in result.structured_data
    assert "values" in result.structured_data
    assert "labels" in result.structured_data


def test_pie_parser_validation_export_and_debug(tmp_path) -> None:
    parser = PieChartParser()
    result = parser.parse(_pie_image())
    assert parser.validate(result)
    out = parser.export(result, tmp_path / "pie_result.txt")
    assert out.exists()
    overlay = parser.debug_overlay(_pie_image(), result, tmp_path / "overlay.png")
    assert Path(overlay).exists()


def test_pie_parser_extension_hooks() -> None:
    parser = PieChartParser()
    result = parser.parse(_pie_image())
    assert parser.fit_kde([])["fitted"] is False
    assert parser.fit_gaussian([])["fitted"] is False
    assert parser.reconstruct_nested([])["supported"] is False

