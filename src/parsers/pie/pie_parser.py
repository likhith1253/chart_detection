from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from src.parser_engine import ParserResult
from src.router import BaseChartParser, get_parser_registry
from src.parsers.bar.ocr.ocr_manager import OCRManager
from src.parsers.bar.ocr.tick_parser import TickParser
from src.parsers.bar.ocr.label_matcher import CategoryMatcher

from .circle_detector import CircleDetector, CircleDetectionResult
from .debug import create_debug_overlay
from .geometry import DetectedSlice, PieSeries, SliceGeometry
from .label_mapper import LabelMapper
from .slice_detector import SliceDetector
from .utils import save_result_text
from .validator import PieValidator
from .value_extractor import ValueExtractor


@dataclass(frozen=True)
class PieParserOutput:
    chart_type: str
    pie_type: str
    center: dict[str, int]
    radius: float
    inner_radius: float
    title: str
    slices: list[dict[str, Any]]
    values: list[dict[str, float]]
    labels: list[dict[str, Any]]


class PieChartParser(BaseChartParser):
    parser_name = "pie"

    def __init__(self) -> None:
        self.circle_detector = CircleDetector()
        self.slice_detector = SliceDetector()
        self.label_mapper = LabelMapper()
        self.value_extractor = ValueExtractor()
        self.validator = PieValidator()
        self.ocr_manager = OCRManager()
        self.tick_parser = TickParser()
        self.category_matcher = CategoryMatcher()

    def parse(self, image: Image.Image, *, metadata: dict[str, Any] | None = None) -> ParserResult:
        start = perf_counter()
        metadata = dict(metadata or {})
        circle = self.circle_detector.detect(image)
        slices = self.slice_detector.detect(image, circle)
        ocr_results = self.ocr_manager.recognize(image, metadata=metadata)
        title = self.ocr_manager.extract_title(image, metadata=metadata)
        labels = self.label_mapper.map(slices, ocr_results)
        values = self.value_extractor.estimate(slices, absolute_total=metadata.get("absolute_total"))
        validation = self.validator.validate(slices, center=circle.center)
        pie_type = self._pie_type(circle, slices, metadata)
        output = PieParserOutput(
            chart_type="pie",
            pie_type=pie_type,
            center={"x": circle.center[0], "y": circle.center[1]},
            radius=float(circle.radius),
            inner_radius=float(circle.inner_radius),
            title=title.text or metadata.get("title", ""),
            slices=[self._serialize_slice(s, idx) for idx, s in enumerate(slices)],
            values=values,
            labels=[asdict(label) for label in labels],
        )
        return ParserResult(
            success=validation.valid,
            chart_type="pie",
            confidence=self._confidence(circle, slices, validation.valid),
            structured_data=asdict(output),
            warnings=validation.warnings,
            execution_time=perf_counter() - start,
            provenance={
                "parser_name": self.parser_name,
                "ocr_count": len(ocr_results),
                "metadata": metadata,
                "pie_type": pie_type,
            },
        )

    def validate(self, parsed_output: Any) -> bool:
        return isinstance(parsed_output, ParserResult) and parsed_output.chart_type == "pie"

    def export(self, parsed_output: Any, output_path: str | Path) -> Path:
        if not isinstance(parsed_output, ParserResult):
            raise TypeError("PieChartParser.export expects ParserResult")
        return save_result_text(output_path, str(parsed_output.structured_data))

    def debug_overlay(self, image: Image.Image, result: ParserResult, output_path: str | Path | None = None) -> Image.Image | Path:
        slices = [
            DetectedSlice(
                geometry=SliceGeometry(
                    start_angle=s["start_angle"],
                    end_angle=s["end_angle"],
                    sweep_angle=s["sweep_angle"],
                    radius=s["radius"],
                    inner_radius=s["inner_radius"],
                    centroid=(float(s["centroid"]["x"]), float(s["centroid"]["y"])),
                    confidence=float(s.get("confidence", 0.0)),
                    label=s.get("label", ""),
                    slice_id=s.get("slice_id"),
                ),
                score=float(s.get("score", 0.0)),
                exploded=bool(s.get("exploded", False)),
            )
            for s in result.structured_data.get("slices", [])
        ]
        center = result.structured_data.get("center", {"x": image.size[0] // 2, "y": image.size[1] // 2})
        circle = CircleDetectionResult(center=(int(center["x"]), int(center["y"])), radius=int(result.structured_data.get("radius", min(image.size) * 0.35)), inner_radius=int(result.structured_data.get("inner_radius", 0)), confidence=float(result.confidence))
        return create_debug_overlay(image, slices, circle, labels=result.structured_data.get("labels", []), output_path=output_path)

    def fit_kde(self, slices: list[DetectedSlice]) -> dict[str, Any]:
        return {"fitted": False, "reason": "hook_only", "slice_count": len(slices)}

    def fit_gaussian(self, slices: list[DetectedSlice]) -> dict[str, Any]:
        return {"fitted": False, "reason": "hook_only", "slice_count": len(slices)}

    def reconstruct_nested(self, slices: list[DetectedSlice]) -> dict[str, Any]:
        return {"supported": False, "reason": "hook_only", "slice_count": len(slices)}

    def _pie_type(self, circle: CircleDetectionResult, slices: list[DetectedSlice], metadata: dict[str, Any]) -> str:
        if circle.inner_radius > 0:
            return "donut"
        if metadata.get("pie_type"):
            return str(metadata["pie_type"])
        if any(s.exploded for s in slices):
            return "exploded"
        if sum(s.geometry.sweep_angle for s in slices) < 330:
            return "partial"
        return "full"

    def _serialize_slice(self, item: DetectedSlice, idx: int) -> dict[str, Any]:
        g = item.geometry
        return {
            "index": idx,
            "slice_id": g.slice_id,
            "start_angle": g.start_angle,
            "end_angle": g.end_angle,
            "sweep_angle": g.sweep_angle,
            "radius": g.radius,
            "inner_radius": g.inner_radius,
            "centroid": {"x": g.centroid[0], "y": g.centroid[1]},
            "confidence": g.confidence,
            "label": g.label,
            "score": item.score,
            "exploded": item.exploded,
        }

    def _confidence(self, circle: CircleDetectionResult, slices: list[DetectedSlice], valid: bool) -> float:
        base = circle.confidence * 0.5 + (sum(s.score for s in slices) / len(slices) if slices else 0.0) * 0.4
        return round(min(1.0, base + (0.1 if valid else 0.0)), 4)


def get_pie_parser() -> PieChartParser:
    return PieChartParser()


_REGISTRY = get_parser_registry()
if not _REGISTRY.has("pie"):
    _REGISTRY.register(get_pie_parser())

