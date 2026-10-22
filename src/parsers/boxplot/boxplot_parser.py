from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from src.parser_engine import ParserResult
from src.parsers.bar.calibration import AxisCalibration
from src.parsers.bar.ocr.ocr_manager import OCRManager
from src.parsers.bar.ocr.tick_parser import TickParser
from src.router import BaseChartParser, get_parser_registry

from .box_detector import BoxDetector
from .debug import create_debug_overlay
from .geometry import BoxStatistics
from .median_detector import MedianDetector
from .outlier_detector import OutlierDetector
from .quartile_estimator import QuartileEstimator
from .utils import save_result_text
from .validator import BoxPlotValidator
from .value_extractor import ValueExtractor
from .whisker_detector import WhiskerDetector


@dataclass(frozen=True)
class BoxPlotParserOutput:
    chart_type: str
    orientation: str
    title: str
    axes: dict[str, Any]
    boxes: list[dict[str, Any]]
    statistics: list[dict[str, Any]]
    outliers: list[dict[str, Any]]
    hooks: dict[str, list[str]]


class BoxPlotParser(BaseChartParser):
    parser_name = "boxplot"

    def __init__(self) -> None:
        self.box_detector = BoxDetector()
        self.whisker_detector = WhiskerDetector()
        self.median_detector = MedianDetector()
        self.quartile_estimator = QuartileEstimator()
        self.outlier_detector = OutlierDetector()
        self.value_extractor = ValueExtractor()
        self.validator = BoxPlotValidator()
        self.ocr_manager = OCRManager()
        self.tick_parser = TickParser()

    def parse(self, image: Image.Image, *, metadata: dict[str, Any] | None = None) -> ParserResult:
        start = perf_counter()
        metadata = dict(metadata or {})
        detection = self.box_detector.detect(image, metadata=metadata)
        ocr_results = self.ocr_manager.recognize(image, metadata=metadata)
        title = self.ocr_manager.extract_title(image, metadata=metadata)
        legend = self.ocr_manager.extract_legend(image, metadata=metadata)
        axis_ticks = self.ocr_manager.extract_ticks(image, axis="y" if detection.orientation != "horizontal" else "x", metadata=metadata)
        calibration = self._calibrate(image, detection.orientation, axis_ticks)
        boxes = []
        statistics = []
        outliers = []
        for box in detection.boxes:
            detected_outliers = self.outlier_detector.detect(image, box)
            box = type(box)(box.box_id, box.geometry, box.quartiles, self.whisker_detector.detect(box), detected_outliers, box.confidence)
            values = self.value_extractor.estimate(box, calibration=calibration, orientation=detection.orientation, axis_scale=metadata.get("axis_scale", "linear"))
            boxes.append(self._serialize_box(box, values))
            statistics.append(asdict(BoxStatistics(box.box_id, values.lower_whisker, values.lower_quartile, values.median, values.upper_quartile, values.upper_whisker, len(detected_outliers), sum(o.extreme for o in detected_outliers))))
            outliers.extend(self._serialize_outlier(o) for o in detected_outliers)
        validation = self.validator.validate(detection.boxes, detection.orientation, plot_area=(0, 0, image.size[0], image.size[1]))
        output = BoxPlotParserOutput(
            chart_type="boxplot",
            orientation=detection.orientation,
            title=title.text or metadata.get("title", ""),
            axes={"calibration": asdict(calibration), "ticks": [asdict(t) for t in getattr(axis_ticks, "ticks", [])], "tick_ordered": getattr(axis_ticks, "ordered", False), "tick_numeric": getattr(axis_ticks, "numeric", False)},
            boxes=boxes,
            statistics=statistics,
            outliers=outliers,
            hooks={"extension_points": ["violin_plots", "candlestick_charts", "boxen_plots"]},
        )
        return ParserResult(success=validation.valid, chart_type="boxplot", confidence=self._confidence(detection.boxes, validation.valid), structured_data=asdict(output), warnings=validation.warnings, execution_time=perf_counter() - start, provenance={"parser_name": self.parser_name, "ocr_count": len(ocr_results), "legend_count": len(legend), "metadata": metadata})

    def validate(self, parsed_output: Any) -> bool:
        return isinstance(parsed_output, ParserResult) and parsed_output.chart_type == "boxplot"

    def export(self, parsed_output: Any, output_path: str | Path) -> Path:
        if not isinstance(parsed_output, ParserResult):
            raise TypeError("BoxPlotParser.export expects ParserResult")
        return save_result_text(output_path, str(parsed_output.structured_data))

    def debug_overlay(self, image: Image.Image, result: ParserResult, output_path: str | Path | None = None) -> Image.Image | Path:
        from .geometry import BoxGeometry, DetectedBox, QuartileGeometry, WhiskerGeometry, OutlierPoint
        boxes = []
        outliers = []
        for item in result.structured_data.get("boxes", []):
            g = item["geometry"]
            q = item["quartiles"]
            w = item["whiskers"]
            box = DetectedBox(item["box_id"], BoxGeometry(g["x1"], g["y1"], g["x2"], g["y2"], g["orientation"], confidence=float(g.get("confidence", 0.0))), QuartileGeometry(q["lower_quartile"], q["median"], q["upper_quartile"]), WhiskerGeometry(w["lower"], w["upper"], w.get("lower_cap"), w.get("upper_cap")), confidence=float(item.get("confidence", 0.0)))
            boxes.append(box)
        for item in result.structured_data.get("outliers", []):
            outliers.append(OutlierPoint(item["x"], item["y"], item.get("value"), bool(item.get("extreme", False)), item.get("box_id"), float(item.get("confidence", 0.0))))
        return create_debug_overlay(image, boxes, outliers, output_path=output_path)

    def fit_violin(self, boxes: list[Any]) -> dict[str, Any]:
        return {"supported": False, "reason": "hook_only", "box_count": len(boxes)}

    def fit_candlestick(self, boxes: list[Any]) -> dict[str, Any]:
        return {"supported": False, "reason": "hook_only", "box_count": len(boxes)}

    def fit_boxen(self, boxes: list[Any]) -> dict[str, Any]:
        return {"supported": False, "reason": "hook_only", "box_count": len(boxes)}

    def _calibrate(self, image: Image.Image, orientation: str, tick_result: Any | None) -> AxisCalibration:
        if tick_result is not None and getattr(tick_result, "ticks", None):
            axis = "x" if orientation == "horizontal" else "y"
            return AxisCalibration.from_ticks(0.0, 0.0, tick_result, axis=axis, fallback_scale=1.0, scale_type="linear")
        return AxisCalibration(0.0, 0.0, 1.0, 1.0, 1, 1, metadata={"source": "fallback", "image_size": image.size})

    def _serialize_box(self, box: Any, values: Any) -> dict[str, Any]:
        g = box.geometry
        q = box.quartiles
        w = box.whiskers
        return {"box_id": box.box_id, "geometry": {"x1": g.x1, "y1": g.y1, "x2": g.x2, "y2": g.y2, "orientation": g.orientation, "confidence": g.confidence}, "quartiles": asdict(q), "whiskers": asdict(w), "values": asdict(values), "outlier_count": len(box.outliers), "confidence": box.confidence}

    def _serialize_outlier(self, outlier: Any) -> dict[str, Any]:
        return {"x": outlier.x, "y": outlier.y, "value": outlier.value, "extreme": outlier.extreme, "box_id": outlier.box_id, "confidence": outlier.confidence}

    def _confidence(self, boxes: list[Any], valid: bool) -> float:
        base = sum(box.confidence for box in boxes) / len(boxes) if boxes else 0.0
        return round(min(1.0, base + (0.1 if valid else 0.0)), 4)


def get_boxplot_parser() -> BoxPlotParser:
    return BoxPlotParser()


_REGISTRY = get_parser_registry()
if not _REGISTRY.has("boxplot"):
    _REGISTRY.register(get_boxplot_parser())
