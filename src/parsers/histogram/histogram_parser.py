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
from .bin_detector import BinDetector
from .debug import create_debug_overlay
from .distribution_analyzer import DistributionAnalyzer
from .geometry import BinGeometry, DetectedBin, DistributionStatistics, HistogramDistribution
from .utils import save_result_text
from .validator import HistogramValidator
from .value_extractor import ValueExtractor


@dataclass(frozen=True)
class HistogramParserOutput:
    chart_type: str
    plot_area: dict[str, float]
    title: str
    axes: dict[str, Any]
    bins: list[dict[str, Any]]
    distribution: dict[str, Any]
    values: list[dict[str, float]]


class HistogramChartParser(BaseChartParser):
    parser_name = "histogram"

    def __init__(self) -> None:
        self.bin_detector = BinDetector()
        self.distribution_analyzer = DistributionAnalyzer()
        self.axis_detector = AxisDetector()
        self.validator = HistogramValidator()
        self.value_extractor = ValueExtractor()
        self.ocr_manager = OCRManager()
        self.tick_parser = TickParser()
        self.category_matcher = CategoryMatcher()

    def parse(self, image: Image.Image, *, metadata: dict[str, Any] | None = None) -> ParserResult:
        start = perf_counter()
        metadata = dict(metadata or {})
        bins = self.bin_detector.detect(image)
        distribution = self.distribution_analyzer.analyze(bins)
        axis_info = self.axis_detector.detect(image, bins=bins)
        ocr_results = self.ocr_manager.recognize(image, metadata=metadata)
        tick_result = self.ocr_manager.extract_ticks(image, axis="y", metadata=metadata)
        title = self.ocr_manager.extract_title(image, metadata=metadata)
        legend = self.ocr_manager.extract_legend(image, metadata=metadata)
        calibration = self._calibrate(axis_info, tick_result, bins)
        values = self.value_extractor.estimate_bins(bins, calibration=calibration, axis_scale=metadata.get("axis_scale", "linear"))
        validation = self.validator.validate(bins, distribution, axis_info.plot_area)
        output = HistogramParserOutput(
            chart_type="histogram",
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
            bins=[self._serialize_bin(bin_, idx) for idx, bin_ in enumerate(bins)],
            distribution=self._serialize_distribution(distribution),
            values=values,
        )
        return ParserResult(
            success=validation.valid,
            chart_type="histogram",
            confidence=self._confidence(bins, distribution, axis_info.confidence, validation.valid),
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
        return isinstance(parsed_output, ParserResult) and parsed_output.chart_type == "histogram"

    def export(self, parsed_output: Any, output_path: str | Path) -> Path:
        if not isinstance(parsed_output, ParserResult):
            raise TypeError("HistogramChartParser.export expects ParserResult")
        return save_result_text(output_path, str(parsed_output.structured_data))

    def debug_overlay(self, image: Image.Image, result: ParserResult, output_path: str | Path | None = None) -> Image.Image | Path:
        bins = [DetectedBin(geometry=BinGeometry(b["bbox"]["x1"], b["bbox"]["y1"], b["bbox"]["x2"], b["bbox"]["y2"], b["width"], b["height"], b["area"], (b["centroid"]["x"], b["centroid"]["y"]), confidence=float(b.get("confidence", 0.0))), score=float(b.get("score", 0.0))) for b in result.structured_data.get("bins", [])]
        axes = result.structured_data.get("axes", {})
        plot_area = result.structured_data.get("plot_area", {})
        width, height = image.size
        axis_info = AxisInfo(True, True, tuple(axes.get("origin", (0, 0))), (
            int(plot_area.get("x1", 0) * width),
            int(plot_area.get("y1", 0) * height),
            int(plot_area.get("x2", width) * width),
            int(plot_area.get("y2", height) * height),
        ), "continuous", confidence=float(axes.get("confidence", 0.0)))
        stats = DistributionStatistics(**result.structured_data.get("distribution", {}).get("metadata", {})) if result.structured_data.get("distribution", {}).get("metadata") else None
        return create_debug_overlay(image, bins, stats, axis_info, output_path=output_path)

    def reconstruct_distribution(self, bins: list[DetectedBin], method: str = "histogram") -> dict[str, Any]:
        return {"method": method, "bin_count": len(bins), "extension_points": ["kde", "gaussian_fit", "mixture_modeling"]}

    def fit_gaussian(self, bins: list[DetectedBin]) -> dict[str, Any]:
        return {"fitted": False, "reason": "hook_only", "bin_count": len(bins)}

    def fit_mixture(self, bins: list[DetectedBin]) -> dict[str, Any]:
        return {"fitted": False, "reason": "hook_only", "bin_count": len(bins)}

    def _calibrate(self, axis_info: AxisInfo, tick_result: Any | None, bins: list[DetectedBin]) -> AxisCalibration:
        if tick_result is not None and getattr(tick_result, "ticks", None):
            return AxisCalibration.from_ticks(axis_info.origin[0], axis_info.origin[1], tick_result, axis="x", fallback_scale=1.0, scale_type="linear")
        scale = 1.0 / max(1, len(bins))
        return AxisCalibration(axis_info.origin[0], axis_info.origin[1], scale, scale, 1, 1, metadata={"source": "fallback"})

    def _serialize_bin(self, bin_: DetectedBin, idx: int) -> dict[str, Any]:
        g = bin_.geometry
        return {
            "index": idx,
            "bbox": {"x1": g.x1, "y1": g.y1, "x2": g.x2, "y2": g.y2},
            "width": g.width,
            "height": g.height,
            "area": g.area,
            "centroid": {"x": g.centroid[0], "y": g.centroid[1]},
            "confidence": g.confidence,
            "score": bin_.score,
            "merged_from": list(bin_.merged_from),
        }

    def _serialize_distribution(self, distribution: HistogramDistribution) -> dict[str, Any]:
        stats = self.distribution_analyzer.statistics(distribution.bins)
        return {
            "continuity": distribution.continuity,
            "symmetry": distribution.symmetry,
            "skewness": distribution.skewness,
            "density": distribution.density,
            "statistics": asdict(stats),
            "metadata": distribution.metadata,
        }

    def _confidence(self, bins: list[DetectedBin], distribution: HistogramDistribution, axis_conf: float, valid: bool) -> float:
        bin_score = sum(b.score for b in bins) / len(bins) if bins else 0.0
        dist_score = (distribution.continuity + distribution.symmetry + min(1.0, abs(distribution.skewness))) / 3.0
        return round(min(1.0, bin_score * 0.45 + dist_score * 0.35 + axis_conf * 0.2 + (0.1 if valid else 0.0)), 4)

    def _normalize_plot_area(self, plot_area: tuple[int, int, int, int], image_size: tuple[int, int]) -> dict[str, float]:
        width, height = image_size
        x1, y1, x2, y2 = plot_area
        return {"x1": round(x1 / width, 6) if width else 0.0, "y1": round(y1 / height, 6) if height else 0.0, "x2": round(x2 / width, 6) if width else 0.0, "y2": round(y2 / height, 6) if height else 0.0}


def get_histogram_parser() -> HistogramChartParser:
    return HistogramChartParser()


_REGISTRY = get_parser_registry()
if not _REGISTRY.has("histogram"):
    _REGISTRY.register(get_histogram_parser())

