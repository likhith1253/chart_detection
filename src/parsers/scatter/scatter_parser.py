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

from .axis_detector import AxisDetector, AxisInfo
from .cluster_detector import ClusterDetector
from .debug import create_debug_overlay
from .geometry import DetectedPoint, MarkerGeometry, ScatterSeries
from .point_detector import PointDetector
from .series_detector import SeriesDetector
from .utils import save_result_text
from .validator import ScatterValidator
from .value_extractor import ValueExtractor


@dataclass(frozen=True)
class ScatterParserOutput:
    chart_type: str
    orientation: str
    plot_area: dict[str, float]
    title: str
    axes: dict[str, Any]
    markers: list[dict[str, Any]]
    points: list[dict[str, Any]]
    series: list[dict[str, Any]]


class ScatterChartParser(BaseChartParser):
    parser_name = "scatter"

    def __init__(self) -> None:
        self.point_detector = PointDetector()
        self.cluster_detector = ClusterDetector()
        self.series_detector = SeriesDetector()
        self.axis_detector = AxisDetector()
        self.validator = ScatterValidator()
        self.value_extractor = ValueExtractor()
        self.ocr_manager = OCRManager()
        self.tick_parser = TickParser()
        self.category_matcher = CategoryMatcher()

    def parse(self, image: Image.Image, *, metadata: dict[str, Any] | None = None) -> ParserResult:
        start = perf_counter()
        metadata = dict(metadata or {})
        markers = self.point_detector.detect(image)
        clusters = self.cluster_detector.detect(markers)
        series = self.series_detector.detect(clusters)
        flat_points = [pt for item in series for pt in item.points]
        axis_info = self.axis_detector.detect(image, points=markers)
        ocr_results = self.ocr_manager.recognize(image, metadata=metadata)
        tick_result = self.ocr_manager.extract_ticks(image, axis="y", metadata=metadata)
        title = self.ocr_manager.extract_title(image, metadata=metadata)
        legend = self.ocr_manager.extract_legend(image, metadata=metadata)
        calibration = self._calibrate(axis_info, tick_result)
        numeric_points = self.value_extractor.estimate_points(series, calibration=calibration, axis_scale=metadata.get("axis_scale", "linear"))
        validation = self.validator.validate(markers, flat_points, series, plot_area=axis_info.plot_area)
        output = ScatterParserOutput(
            chart_type="scatter",
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
            markers=[self._serialize_marker(marker, idx) for idx, marker in enumerate(markers)],
            points=[self._serialize_point(point, idx) for idx, point in enumerate(flat_points)],
            series=[self._serialize_series(item, numeric_points[idx] if idx < len(numeric_points) else [], idx) for idx, item in enumerate(series)],
        )
        return ParserResult(
            success=validation.valid,
            chart_type="scatter",
            confidence=self._confidence(markers, axis_info.confidence, validation.valid),
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
        return isinstance(parsed_output, ParserResult) and parsed_output.chart_type == "scatter"

    def export(self, parsed_output: Any, output_path: str | Path) -> Path:
        if not isinstance(parsed_output, ParserResult):
            raise TypeError("ScatterChartParser.export expects ParserResult")
        return save_result_text(output_path, str(parsed_output.structured_data))

    def debug_overlay(self, image: Image.Image, result: ParserResult, output_path: str | Path | None = None) -> Image.Image | Path:
        markers = [MarkerGeometry(bbox=tuple(m["bbox"][k] for k in ("x1", "y1", "x2", "y2")), centroid=(float(m["centroid"]["x"]), float(m["centroid"]["y"])), shape=m.get("shape", "unknown"), confidence=float(m.get("confidence", 0.0)), series_id=m.get("series_id")) for m in result.structured_data.get("markers", [])]
        points = [DetectedPoint(x=float(p["x"]), y=float(p["y"]), confidence=float(p.get("confidence", 0.0)), marker_shape=p.get("marker_shape", "unknown"), series_id=p.get("series_id")) for p in result.structured_data.get("points", [])]
        series = []
        for item in result.structured_data.get("series", []):
            pts = [DetectedPoint(x=float(p["x"]), y=float(p["y"]), confidence=float(p.get("confidence", 0.0)), series_id=item.get("series_id")) for p in item.get("points", [])]
            series.append(ScatterSeries(series_id=int(item.get("series_id", 0)), markers=markers, points=pts, confidence=float(item.get("confidence", 0.0)), label=item.get("label", "")))
        axes = result.structured_data.get("axes", {})
        plot_area = result.structured_data.get("plot_area", {})
        width, height = image.size
        axis_info = AxisInfo(True, True, tuple(axes.get("origin", (0, 0))), (
            int(plot_area.get("x1", 0) * width),
            int(plot_area.get("y1", 0) * height),
            int(plot_area.get("x2", width) * width),
            int(plot_area.get("y2", height) * height),
        ), result.structured_data.get("orientation", "mixed"), confidence=float(axes.get("confidence", 0.0)))
        return create_debug_overlay(image, markers, points, series, axis_info, output_path=output_path)

    def _calibrate(self, axis_info: AxisInfo, tick_result: Any | None) -> AxisCalibration:
        if tick_result is not None and getattr(tick_result, "ticks", None):
            return AxisCalibration.from_ticks(axis_info.origin[0], axis_info.origin[1], tick_result, axis="y", fallback_scale=1.0, scale_type="linear")
        return AxisCalibration(axis_info.origin[0], axis_info.origin[1], 1.0, 1.0, 1, 1, metadata={"source": "fallback"})

    def _serialize_marker(self, marker: MarkerGeometry, idx: int) -> dict[str, Any]:
        return {
            "index": idx,
            "bbox": {"x1": marker.bbox[0], "y1": marker.bbox[1], "x2": marker.bbox[2], "y2": marker.bbox[3]},
            "centroid": {"x": marker.centroid[0], "y": marker.centroid[1]},
            "shape": marker.shape,
            "color": marker.color,
            "confidence": marker.confidence,
            "series_id": marker.series_id,
        }

    def _serialize_point(self, point: DetectedPoint, idx: int) -> dict[str, Any]:
        return {"index": idx, "x": point.x, "y": point.y, "confidence": point.confidence, "marker_shape": point.marker_shape, "series_id": point.series_id}

    def _serialize_series(self, series: ScatterSeries, values: list[dict[str, float]], idx: int) -> dict[str, Any]:
        return {"index": idx, "series_id": series.series_id, "label": series.label, "confidence": series.confidence, "points": values}

    def _confidence(self, markers: list[MarkerGeometry], axis_conf: float, valid: bool) -> float:
        base = (sum(m.confidence for m in markers) / len(markers) if markers else 0.0) * 0.8 + axis_conf * 0.2
        return round(min(1.0, base + (0.1 if valid else 0.0)), 4)

    def _normalize_plot_area(self, plot_area: tuple[int, int, int, int], image_size: tuple[int, int]) -> dict[str, float]:
        width, height = image_size
        x1, y1, x2, y2 = plot_area
        return {"x1": round(x1 / width, 6) if width else 0.0, "y1": round(y1 / height, 6) if height else 0.0, "x2": round(x2 / width, 6) if width else 0.0, "y2": round(y2 / height, 6) if height else 0.0}


def get_scatter_parser() -> ScatterChartParser:
    return ScatterChartParser()


_REGISTRY = get_parser_registry()
if not _REGISTRY.has("scatter"):
    _REGISTRY.register(get_scatter_parser())

