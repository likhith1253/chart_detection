from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from src.parser_engine import ParserResult
from src.parsers.bar import BarChartParser, get_bar_parser
from src.parsers.bar.axis_detector import AxisDetector
from src.parsers.bar.bar_detector import BarDetector
from src.parsers.bar.debug import create_debug_overlay
from src.parsers.bar.geometry import BarGeometry
from src.parsers.bar.validator import BarValidator
from src.parsers.bar.value_extractor import ValueExtractor
from src.router import get_parser_registry


def _bar_image() -> Image.Image:
    img = Image.new("RGB", (120, 120), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 30, 40, 100), fill="black")
    draw.rectangle((55, 50, 75, 100), fill="black")
    draw.rectangle((90, 20, 110, 100), fill="black")
    return img


def _horizontal_bar_image() -> Image.Image:
    img = Image.new("RGB", (160, 120), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((15, 20, 90, 35), fill="black")
    draw.rectangle((15, 50, 120, 65), fill="black")
    draw.rectangle((15, 80, 70, 95), fill="black")
    return img


def test_bar_parser_auto_registers() -> None:
    registry = get_parser_registry()
    assert registry.has("bar")
    assert registry.get("bar").parser_name == "bar"


def test_bar_parser_returns_parser_result() -> None:
    parser = get_bar_parser()
    result = parser.parse(_bar_image(), metadata={"axis_scale": "linear"})
    assert isinstance(result, ParserResult)
    assert result.success in {True, False}
    assert result.chart_type == "bar"
    assert "bars" in result.structured_data
    assert "orientation" in result.structured_data
    assert "axes" in result.structured_data
    assert result.provenance["parser_name"] == "bar"


def test_bar_parser_validation_and_export(tmp_path) -> None:
    parser = BarChartParser()
    result = parser.parse(_bar_image())
    assert parser.validate(result)
    out = parser.export(result, tmp_path / "bar_result.txt")
    assert out.exists()


def test_bar_parser_structured_output_has_labels_and_values() -> None:
    parser = BarChartParser()
    result = parser.parse(_bar_image())
    data = result.structured_data
    assert isinstance(data["labels"]["category_labels"], list)
    assert isinstance(data["values"], list)
    assert len(data["values"]) == len(data["bars"])


def test_detector_and_axis_calibration_produce_geometry() -> None:
    detector = BarDetector()
    bars = detector.detect(_bar_image())
    assert bars
    assert all(bar.geometry.width > 0 and bar.geometry.height > 0 for bar in bars)
    assert all(bar.geometry.area > 0 for bar in bars)

    axis = AxisDetector().detect(_bar_image(), bars=bars)
    assert axis.x_axis_present and axis.y_axis_present
    assert axis.confidence >= 0.0
    assert 0 <= axis.plot_area[0] <= axis.plot_area[2] <= 120


def test_orientation_and_value_extraction() -> None:
    vertical_bars = BarDetector().detect(_bar_image())
    horizontal_bars = BarDetector().detect(_horizontal_bar_image())
    vertical_geoms = [b.geometry for b in vertical_bars]
    horizontal_geoms = [b.geometry for b in horizontal_bars]

    values_v = ValueExtractor().estimate_values(vertical_geoms, orientation="vertical")
    values_h = ValueExtractor().estimate_values(horizontal_geoms, orientation="horizontal")
    assert len(values_v) == len(vertical_geoms)
    assert len(values_h) == len(horizontal_geoms)
    assert all(v >= 0 for v in values_v)
    assert all(v >= 0 for v in values_h)


def test_validator_catches_problems() -> None:
    bars = [BarGeometry(0, 0, 10, 50, "vertical"), BarGeometry(5, 5, 12, 40, "vertical")]
    result = BarValidator().validate(bars, ["a"], "horizontal", plot_area=(0, 0, 20, 20))
    assert not result.valid
    assert "number_of_bars_does_not_match_number_of_labels" in result.warnings
    assert "overlapping_bars_detected" in result.warnings or "out_of_plot_detection" in result.warnings


def test_debug_overlay_saves_output(tmp_path) -> None:
    parser = BarChartParser()
    result = parser.parse(_bar_image())
    overlay = parser.debug_overlay(_bar_image(), result, tmp_path / "overlay.png")
    assert Path(overlay).exists()
