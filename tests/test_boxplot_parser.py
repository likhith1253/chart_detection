from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.parser_engine import ParserResult
from src.parsers.boxplot.boxplot_parser import BoxPlotParser
from src.parsers.boxplot.geometry import BoxGeometry, DetectedBox, OutlierPoint, QuartileGeometry, WhiskerGeometry
from src.parsers.boxplot.validator import BoxPlotValidator
from src.router import get_parser_registry


def test_boxplot_registered():
    assert get_parser_registry().has("boxplot")


def test_boxplot_parse_export_and_debug(tmp_path: Path):
    parser = BoxPlotParser()
    image = Image.new("RGB", (300, 200), "white")
    result = parser.parse(image, metadata={"orientation": "vertical", "title": "Synthetic Box Plot"})
    assert isinstance(result, ParserResult)
    assert result.chart_type == "boxplot"
    assert "boxes" in result.structured_data
    out = parser.export(result, tmp_path / "result.txt")
    assert out.exists()
    debug = parser.debug_overlay(image, result, output_path=tmp_path / "debug.png")
    assert Path(debug).exists()


def test_boxplot_validation_flags_invalid_geometry():
    validator = BoxPlotValidator()
    box = DetectedBox(
        box_id=1,
        geometry=BoxGeometry(10, 10, 40, 60, "vertical"),
        quartiles=QuartileGeometry(50, 40, 30),
        whiskers=WhiskerGeometry(lower=20, upper=30),
        outliers=[OutlierPoint(999, 999)],
    )
    result = validator.validate([box], "vertical", plot_area=(0, 0, 100, 100))
    assert not result.valid
    assert "quartile_ordering_invalid" in result.warnings
