from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from src.parser_engine import ParserResult
from src.router import BaseChartParser, get_parser_registry

from .axis_detector import AxisDetector, AxisInfo
from .bar_detector import BarDetector, DetectedBar
from .calibration import AxisCalibration
from .debug import create_debug_overlay
from .geometry import BarGeometry
from .interfaces import OCRHook
from .label_detector import LabelDetector
from .utils import save_result_text
from .validator import BarValidator
from .value_extractor import ValueExtractor
from .ocr.ocr_manager import OCRManager
from .ocr.tick_parser import TickParser
from .ocr.label_matcher import CategoryMatcher


@dataclass(frozen=True)
class BarParserOutput:
    chart_type: str
    bars: list[dict[str, Any]]
    axes: dict[str, Any]
    labels: dict[str, Any]
    values: list[float]
    title: str
    orientation: str


class BarChartParser(BaseChartParser):
    parser_name = "bar"

    def __init__(self) -> None:
        self.bar_detector = BarDetector()
        self.axis_detector = AxisDetector()
        self.label_detector = LabelDetector()
        self.value_extractor = ValueExtractor()
        self.validator = BarValidator()
        self.ocr_hook = OCRHook()
        self.ocr_manager = OCRManager()
        self.tick_parser = TickParser()
        self.category_matcher = CategoryMatcher()

    def parse(self, image: Image.Image, *, metadata: dict[str, Any] | None = None) -> ParserResult:
        start = perf_counter()
        metadata = dict(metadata or {})
        detected_bars = self.bar_detector.detect(image)
        axis_info = self.axis_detector.detect(image, bars=detected_bars)
        bar_geometries = [bar.geometry for bar in detected_bars]
        ocr_results = self.ocr_manager.recognize(image, metadata=metadata)
        tick_result = self.ocr_manager.extract_ticks(image, axis="y" if axis_info.orientation != "horizontal" else "x", metadata=metadata)
        labels = self.label_detector.detect_categories(image, len(bar_geometries))
        axis_labels = self.label_detector.detect_axis_labels(image)
        title_text = self.ocr_manager.extract_title(image, metadata=metadata)
        legend_texts = self.ocr_manager.extract_legend(image, metadata=metadata)
        category_matches = self.category_matcher.match(bar_geometries, ocr_results, orientation=axis_info.orientation)
        orientation = self._infer_orientation(bar_geometries, axis_info)
        calibration = self._calibrate_axes(bar_geometries, axis_info, tick_result)
        values = self.value_extractor.estimate_values(
            bar_geometries,
            calibration=calibration,
            axis_scale=metadata.get("axis_scale", "linear"),
            orientation=orientation,
        )
        validation = self.validator.validate(bar_geometries, [label.text for label in labels], orientation, plot_area=axis_info.plot_area)
        output = BarParserOutput(
            chart_type="bar",
            bars=[self._serialize_bar(bar, value, idx, labels[idx].text if idx < len(labels) else "") for idx, (bar, value) in enumerate(zip(bar_geometries, values))],
            axes={
                "x": axis_labels.get("x_axis_label", ""),
                "y": axis_labels.get("y_axis_label", ""),
                "plot_area": self._normalize_plot_area(axis_info.plot_area, image.size),
                "origin": axis_info.origin,
                "confidence": axis_info.confidence,
                "calibration": asdict(calibration),
                "ticks": [asdict(t) for t in tick_result.ticks],
                "tick_ordered": tick_result.ordered,
                "tick_numeric": tick_result.numeric,
            },
            labels={
                "category_labels": [asdict(label) for label in labels],
                "legend_labels": [asdict(item) for item in legend_texts],
                "ocr_category_matches": [asdict(match) for match in category_matches],
                **axis_labels,
            },
            values=values,
            title=title_text.text or axis_labels.get("title", ""),
            orientation=orientation,
        )
        return ParserResult(
            success=validation.valid,
            chart_type="bar",
            confidence=self._confidence(detected_bars, validation.valid, axis_info.confidence),
            structured_data=asdict(output),
            warnings=validation.warnings,
            execution_time=perf_counter() - start,
            provenance={
                "parser_name": self.parser_name,
                "detected_bars": len(detected_bars),
                "metadata": metadata,
                "validation_warnings": validation.warnings,
                "ocr_count": len(ocr_results),
            },
        )

    def validate(self, parsed_output: Any) -> bool:
        return isinstance(parsed_output, ParserResult) and parsed_output.chart_type == "bar"

    def export(self, parsed_output: Any, output_path: str | Path) -> Path:
        payload = parsed_output if isinstance(parsed_output, ParserResult) else None
        if payload is None:
            raise TypeError("BarChartParser.export expects ParserResult")
        return save_result_text(output_path, str(payload.structured_data))

    def debug_overlay(self, image: Image.Image, result: ParserResult, output_path: str | Path | None = None) -> Image.Image | Path:
        bars = []
        for item in result.structured_data.get("bars", []):
            bbox = item["bbox"]
            bars.append(BarGeometry(bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"], result.structured_data.get("orientation", "vertical")))
        axes = result.structured_data.get("axes", {})
        plot_area = axes.get("plot_area", {})
        width, height = image.size
        axis_info = AxisInfo(
            True,
            True,
            tuple(axes.get("origin", (0, 0))),
            (
                int(plot_area.get("x1", 0) * width),
                int(plot_area.get("y1", 0) * height),
                int(plot_area.get("x2", width) * width),
                int(plot_area.get("y2", height) * height),
            ),
            result.structured_data.get("orientation", "vertical"),
            confidence=float(axes.get("confidence", 0.0)),
        )
        return create_debug_overlay(image, bars, axis_info, output_path=output_path)

    def _calibrate_axes(self, bars: list[BarGeometry], axis_info: AxisInfo, tick_result: Any | None = None) -> AxisCalibration:
        if not bars:
            return AxisCalibration(axis_info.origin[0], axis_info.origin[1], 1.0, 1.0, 1, 1, metadata={"source": "fallback"})
        axis = "x" if axis_info.orientation == "horizontal" else "y"
        if tick_result is not None and getattr(tick_result, "ticks", None):
            return AxisCalibration.from_ticks(
                axis_info.origin[0],
                axis_info.origin[1],
                tick_result,
                axis=axis,
                fallback_scale=1.0 / max((bar.width if axis == "x" else bar.height for bar in bars), default=1) or 1.0,
                scale_type="linear",
            )
        if axis == "x":
            span = max((bar.width for bar in bars), default=1) or 1
            scale = 1.0 / span
            return AxisCalibration(axis_info.origin[0], axis_info.origin[1], scale, 1.0, 1, 1, metadata={"axis": "horizontal"})
        span = max((bar.height for bar in bars), default=1) or 1
        scale = 1.0 / span
        return AxisCalibration(axis_info.origin[0], axis_info.origin[1], 1.0, scale, 1, -1, metadata={"axis": "vertical"})

    def _serialize_bar(self, bar: BarGeometry, value: float, index: int, label: str) -> dict[str, Any]:
        cx, cy = bar.centroid
        return {
            "index": index,
            "label": label,
            "bbox": {"x1": bar.x1, "y1": bar.y1, "x2": bar.x2, "y2": bar.y2},
            "centroid": {"x": cx, "y": cy},
            "width": bar.width,
            "height": bar.height,
            "area": bar.area,
            "orientation": bar.orientation,
            "series_id": bar.series_id,
            "stack_id": bar.stack_id,
            "value": value,
            "confidence": bar.confidence,
        }

    def _confidence(self, detected_bars: list[DetectedBar], valid: bool, axis_confidence: float) -> float:
        if not detected_bars:
            return 0.0
        base = sum(bar.score for bar in detected_bars) / len(detected_bars)
        return round(min(1.0, base * 0.6 + axis_confidence * 0.4 + (0.1 if valid else 0.0)), 4)

    def _infer_orientation(self, bars: list[BarGeometry], axis_info: AxisInfo) -> str:
        if not bars:
            return axis_info.orientation
        vertical = sum(bar.height >= bar.width for bar in bars)
        horizontal = len(bars) - vertical
        if vertical > horizontal:
            return "vertical"
        if horizontal > vertical:
            return "horizontal"
        return axis_info.orientation

    def _normalize_plot_area(self, plot_area: tuple[int, int, int, int], image_size: tuple[int, int]) -> dict[str, float]:
        width, height = image_size
        x1, y1, x2, y2 = plot_area
        return {
            "x1": round(x1 / width, 6) if width else 0.0,
            "y1": round(y1 / height, 6) if height else 0.0,
            "x2": round(x2 / width, 6) if width else 0.0,
            "y2": round(y2 / height, 6) if height else 0.0,
        }


def get_bar_parser() -> BarChartParser:
    return BarChartParser()


_REGISTRY = get_parser_registry()
if not _REGISTRY.has("bar"):
    _REGISTRY.register(get_bar_parser())
