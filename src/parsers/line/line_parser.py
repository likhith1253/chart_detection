from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from src.parser_engine import ParserResult
from src.router import BaseChartParser, get_parser_registry
from src.parsers.bar.calibration import AxisCalibration
from src.parsers.bar.ocr.ocr_manager import OCRManager
from src.parsers.bar.ocr.tick_parser import TickParser
from src.parsers.bar.ocr.label_matcher import CategoryMatcher
from src.parsers.bar.ocr.title_text import TitleText

from .axis_detector import AxisDetector, AxisInfo
from .curve_detector import CurveDetector
from .debug import create_debug_overlay
from .geometry import CurveGeometry, DetectedPoint, DetectedSeries
from .point_detector import PointDetector
from .series_detector import SeriesDetector
from .utils import save_result_text
from .validator import LineValidator
from .value_extractor import ValueExtractor


@dataclass(frozen=True)
class LineParserOutput:
    chart_type: str
    orientation: str
    plot_area: dict[str, float]
    title: str
    axes: dict[str, Any]
    curves: list[dict[str, Any]]
    points: list[dict[str, Any]]
    series: list[dict[str, Any]]


class LineChartParser(BaseChartParser):
    parser_name = "line"

    def __init__(self) -> None:
        self.curve_detector = CurveDetector()
        self.point_detector = PointDetector()
        self.axis_detector = AxisDetector()
        self.series_detector = SeriesDetector()
        self.validator = LineValidator()
        self.value_extractor = ValueExtractor()
        self.ocr_manager = OCRManager()
        self.tick_parser = TickParser()
        self.category_matcher = CategoryMatcher()

    def parse(self, image: Image.Image, *, metadata: dict[str, Any] | None = None) -> ParserResult:
        start = perf_counter()
        metadata = dict(metadata or {})
        curves = self.curve_detector.detect(image)
        points = self.point_detector.detect(image)
        axis_info = self.axis_detector.detect(image, curves=curves, points=points)
        series = self.series_detector.detect(curves, points)
        ocr_results = self.ocr_manager.recognize(image, metadata=metadata)
        tick_result = self.ocr_manager.extract_ticks(image, axis="y" if axis_info.orientation != "horizontal" else "x", metadata=metadata)
        title = self.ocr_manager.extract_title(image, metadata=metadata)
        legend = self.ocr_manager.extract_legend(image, metadata=metadata)
        calibration = self._calibrate(axis_info, tick_result, series)
        numeric_points = self.value_extractor.estimate_points(series, calibration=calibration, axis_scale=metadata.get("axis_scale", "linear"), orientation=axis_info.orientation)
        validation = self.validator.validate(curves, points, series, axis_info.orientation, plot_area=axis_info.plot_area)
        output = LineParserOutput(
            chart_type="line",
            orientation=axis_info.orientation,
            plot_area=self._normalize_plot_area(axis_info.plot_area, image.size),
            title=title.text or metadata.get("title", ""),
            axes={
                "origin": axis_info.origin,
                "confidence": axis_info.confidence,
                "calibration": asdict(calibration),
                "ticks": [asdict(t) for t in tick_result.ticks],
                "tick_ordered": tick_result.ordered,
                "tick_numeric": tick_result.numeric,
            },
            curves=[self._serialize_curve(curve, idx) for idx, curve in enumerate(curves)],
            points=[self._serialize_point(point, idx) for idx, point in enumerate(points)],
            series=[self._serialize_series(item, numeric_points[idx] if idx < len(numeric_points) else [], idx) for idx, item in enumerate(series)],
        )
        return ParserResult(
            success=validation.valid,
            chart_type="line",
            confidence=self._confidence(curves, points, axis_info.confidence, validation.valid),
            structured_data=asdict(output),
            warnings=validation.warnings,
            execution_time=perf_counter() - start,
            provenance={
                "parser_name": self.parser_name,
                "ocr_count": len(ocr_results),
                "legend_count": len(legend),
                "metadata": metadata,
            },
        )

    def validate(self, parsed_output: Any) -> bool:
        return isinstance(parsed_output, ParserResult) and parsed_output.chart_type == "line"

    def export(self, parsed_output: Any, output_path: str | Path) -> Path:
        if not isinstance(parsed_output, ParserResult):
            raise TypeError("LineChartParser.export expects ParserResult")
        return save_result_text(output_path, str(parsed_output.structured_data))

    def debug_overlay(self, image: Image.Image, result: ParserResult, output_path: str | Path | None = None) -> Image.Image | Path:
        curves = [CurveGeometry(points=[(int(p["x"]), int(p["y"])) for p in item.get("points", [])], bbox=tuple(item.get("bbox", {}).get(k, 0) for k in ("x1", "y1", "x2", "y2")), curve_type=item.get("curve_type", "polyline")) for item in result.structured_data.get("curves", [])]
        points = [DetectedPoint(x=float(p["x"]), y=float(p["y"]), confidence=float(p.get("confidence", 0.0))) for p in result.structured_data.get("points", [])]
        series = []
        for item in result.structured_data.get("series", []):
            pts = [DetectedPoint(x=float(p["x"]), y=float(p["y"]), confidence=float(p.get("confidence", 0.0)), series_id=item.get("series_id")) for p in item.get("points", [])]
            series.append(DetectedSeries(series_id=int(item.get("series_id", 0)), color=None, line_type=item.get("line_type", "polyline"), points=pts))
        axes = result.structured_data.get("axes", {})
        plot_area = result.structured_data.get("plot_area", {})
        width, height = image.size
        axis_info = AxisInfo(True, True, tuple(axes.get("origin", (0, 0))), (
            int(plot_area.get("x1", 0) * width),
            int(plot_area.get("y1", 0) * height),
            int(plot_area.get("x2", width) * width),
            int(plot_area.get("y2", height) * height),
        ), result.structured_data.get("orientation", "vertical"), confidence=float(axes.get("confidence", 0.0)))
        return create_debug_overlay(image, curves, points, series, axis_info, output_path=output_path)

    def _calibrate(self, axis_info: AxisInfo, tick_result: Any | None, series: list[DetectedSeries]) -> AxisCalibration:
        if tick_result is not None and getattr(tick_result, "ticks", None):
            return AxisCalibration.from_ticks(axis_info.origin[0], axis_info.origin[1], tick_result, axis="y" if axis_info.orientation != "horizontal" else "x", fallback_scale=1.0, scale_type="linear")
        scale = 1.0 / max(1, len(series) or 1)
        return AxisCalibration(axis_info.origin[0], axis_info.origin[1], scale, scale, 1, 1, metadata={"source": "fallback"})

    def _serialize_curve(self, curve: CurveGeometry, idx: int) -> dict[str, Any]:
        return {"index": idx, "bbox": {"x1": curve.bbox[0], "y1": curve.bbox[1], "x2": curve.bbox[2], "y2": curve.bbox[3]}, "curve_type": curve.curve_type, "confidence": curve.confidence, "points": [{"x": x, "y": y} for x, y in curve.points], "series_id": curve.series_id}

    def _serialize_point(self, point: DetectedPoint, idx: int) -> dict[str, Any]:
        return {"index": idx, "x": point.x, "y": point.y, "confidence": point.confidence, "marker": point.marker, "series_id": point.series_id}

    def _serialize_series(self, series: DetectedSeries, values: list[dict[str, float]], idx: int) -> dict[str, Any]:
        return {"index": idx, "series_id": series.series_id, "line_type": series.line_type, "confidence": series.confidence, "points": values}

    def _confidence(self, curves: list[CurveGeometry], points: list[DetectedPoint], axis_conf: float, valid: bool) -> float:
        base = (sum(c.confidence for c in curves) / len(curves) if curves else 0.0) * 0.45 + (sum(p.confidence for p in points) / len(points) if points else 0.0) * 0.35 + axis_conf * 0.2
        return round(min(1.0, base + (0.1 if valid else 0.0)), 4)

    def _normalize_plot_area(self, plot_area: tuple[int, int, int, int], image_size: tuple[int, int]) -> dict[str, float]:
        width, height = image_size
        x1, y1, x2, y2 = plot_area
        return {"x1": round(x1 / width, 6) if width else 0.0, "y1": round(y1 / height, 6) if height else 0.0, "x2": round(x2 / width, 6) if width else 0.0, "y2": round(y2 / height, 6) if height else 0.0}


def get_line_parser() -> LineChartParser:
    return LineChartParser()


_REGISTRY = get_parser_registry()
if not _REGISTRY.has("line"):
    _REGISTRY.register(get_line_parser())

