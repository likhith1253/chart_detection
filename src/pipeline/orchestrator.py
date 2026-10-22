from __future__ import annotations

import time
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from src.detection.runtime.cropper import ChartCrop, crop_chart_regions
from src.detection.runtime.runner import DetectionRunner
from src.detection.runtime.prediction_record import DetectionPrediction
from src.parser_engine.engine import ParserEngine
from src.parser_engine.parser_result import ParserResult
from src.router.router import ParserRouter

from .artifacts import ArtifactManager
from .configuration import PipelineConfiguration
from .context import PipelineContext
from .exceptions import PipelineConfigurationError, StageExecutionError
from .hooks import PipelineHooks
from .metrics import PipelineMetrics
from .result import PipelineResult


class ClassifierProtocol(Protocol):
    def classify_chart(self, image_path: str) -> dict[str, Any]: ...


class PipelineClassifierAdapter:
    def __init__(self, classifier: Any):
        self.classifier = classifier

    def classify(self, crop: Image.Image, image_path: str | None = None) -> dict[str, Any]:
        if hasattr(self.classifier, "classify_chart"):
            if image_path is None:
                raise PipelineConfigurationError("classifier requires an image path")
            return dict(self.classifier.classify_chart(str(image_path)))
        if callable(self.classifier):
            return dict(self.classifier(crop))
        raise PipelineConfigurationError("Unsupported classifier interface")


@dataclass
class ResearchPipeline:
    detector_runner: DetectionRunner
    classifier: Any
    router: ParserRouter
    parser_engine: ParserEngine
    config: PipelineConfiguration = field(default_factory=PipelineConfiguration)
    hooks: PipelineHooks = field(default_factory=PipelineHooks)
    metrics: PipelineMetrics = field(default_factory=PipelineMetrics)
    artifacts: ArtifactManager | None = None

    def __post_init__(self) -> None:
        self.config.validate()
        self.artifacts = self.artifacts or ArtifactManager(self.config.output_dir)
        self._classifier_adapter = PipelineClassifierAdapter(self.classifier)

    def _load_page(self, document: str | Path | Image.Image) -> tuple[Image.Image, Path | None]:
        if isinstance(document, Image.Image):
            return document, None
        path = Path(document)
        return Image.open(path).convert("RGB"), path

    def _module_versions(self) -> dict[str, str]:
        versions: dict[str, str] = {}
        for name in ["src", "torch", "numpy", "pandas", "Pillow"]:
            try:
                versions[name] = importlib_metadata.version(name)
            except Exception:
                versions[name] = "unknown"
        return versions

    def _validate_detection_outputs(self, detections: list[DetectionPrediction], crops: list[ChartCrop]) -> None:
        if len(detections) != len(crops):
            raise StageExecutionError("Detection outputs are inconsistent")
        for det in detections:
            if not isinstance(det.bbox, list) or len(det.bbox) != 4:
                raise StageExecutionError("Invalid detection bbox")
            if det.confidence < 0 or det.confidence > 1:
                raise StageExecutionError("Invalid detection confidence")

    def _validate_parser_outputs(self, outputs: list[ParserResult]) -> None:
        for parsed in outputs:
            if not isinstance(parsed.success, bool):
                raise StageExecutionError("Invalid parser output")
            if not isinstance(parsed.chart_type, str):
                raise StageExecutionError("Invalid parser output chart_type")

    def _build_integration_report(self, context: PipelineContext, result: PipelineResult, started_at: float) -> dict[str, Any]:
        return {
            "pipeline": "ResearchPipeline",
            "pipeline_stages": list(self.metrics.stage_timings),
            "stage_timings": dict(self.metrics.stage_timings),
            "module_versions": self._module_versions(),
            "detected_charts": len(context.detections),
            "parsed_charts": len(context.parser_outputs),
            "success_rate": self.metrics.snapshot()["success_rate"],
            "runtime_sec": time.perf_counter() - started_at,
            "warnings": list(result.warnings),
            "errors": list(result.errors),
        }

    def run(
        self,
        document: str | Path | Image.Image,
        *,
        metadata: dict[str, Any] | None = None,
        mock_detections: list[DetectionPrediction] | None = None,
    ) -> PipelineResult:
        context = PipelineContext(document=document, metadata=dict(metadata or {}))
        result = PipelineResult()
        self.metrics.total_runs += 1
        started_at = time.perf_counter()

        try:
            start = time.perf_counter()
            page, page_path = self._load_page(document)
            context.page = page
            context.page_path = page_path
            self.metrics.record_stage("load_document", time.perf_counter() - start)

            self.hooks.run(self.hooks.before_detection, context)

            start = time.perf_counter()
            if mock_detections is not None:
                det_preds = list(mock_detections)
                crops = crop_chart_regions(page, det_preds, save_dir=(self.config.output_dir / "crops") if self.config.save_crops else None)
            else:
                det_preds, crops = self.detector_runner.run_on_image(page, image_id=(page_path.stem if page_path else "page"), save_crops_dir=(self.config.output_dir / "crops") if self.config.save_crops else None)
            context.detections = list(det_preds)
            context.crops = list(crops)
            self._validate_detection_outputs(context.detections, context.crops)
            self.metrics.detected_charts += len(det_preds)
            self.metrics.record_stage("detection", time.perf_counter() - start)
            self.hooks.run(self.hooks.after_detection, context)

            if not crops:
                context.classifier_outputs = []
                context.routing_outputs = []
                context.parser_outputs = []

            for idx, crop in enumerate(crops):
                start = time.perf_counter()
                crop_image = crop.crop_image
                crop_path = None
                if self.config.save_crops or hasattr(self.classifier, "classify_chart"):
                    stem = page_path.stem if page_path is not None else "page"
                    crop_path = self.artifacts.root_dir / "classifier_inputs" / f"{stem}_{idx:04d}.png"
                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                    crop_image.save(crop_path)

                classifier_output = self._classifier_adapter.classify(crop_image, str(crop_path) if crop_path else None)
                context.classifier_outputs.append(classifier_output)
                self.metrics.record_stage("classification", time.perf_counter() - start)
            self.hooks.run(self.hooks.after_classification, context)

            for idx, (crop, det, classifier_output) in enumerate(zip(crops, det_preds, context.classifier_outputs)):
                start = time.perf_counter()
                confidence_scores = dict(classifier_output.get("confidence_scores", {}))
                selected = classifier_output.get("chart_type", classifier_output.get("prediction", "unknown"))
                if selected not in confidence_scores:
                    confidence_scores[selected] = float(classifier_output.get("confidence", 1.0))
                routing_result = self.router.route(crop.crop_image, selected, confidence_scores, metadata={**context.metadata, **classifier_output})
                context.routing_outputs.append(routing_result)
                self.metrics.record_stage("routing", time.perf_counter() - start)
            self.hooks.run(self.hooks.after_routing, context)

            for idx, (crop, det, classifier_output, routing_result) in enumerate(zip(crops, det_preds, context.classifier_outputs, context.routing_outputs)):
                start = time.perf_counter()
                parser_result = self.parser_engine.run(
                    crop.crop_image,
                    routing_result,
                    metadata={**context.metadata, "detection": det.metadata, "crop": crop.transform_information},
                    classifier_output=classifier_output,
                )
                context.parser_outputs.append(parser_result)
                self.metrics.parsed_charts += int(parser_result.success)
                self.metrics.record_stage("parsing", time.perf_counter() - start)
            self._validate_parser_outputs(context.parser_outputs)
            self.hooks.run(self.hooks.after_parsing, context)

            result.detected_charts = context.detections
            result.classified_charts = context.classifier_outputs
            result.parser_outputs = context.parser_outputs
            result.structured_json = {
                "document": str(document),
                "charts": [self._parser_to_dict(i, det, cls, route, parsed) for i, (det, cls, route, parsed) in enumerate(zip(context.detections, context.classifier_outputs, context.routing_outputs, context.parser_outputs))],
            }
            result.runtime_metrics = self.metrics.snapshot()
            result.integration_report = self._build_integration_report(context, result, started_at)

            self.hooks.run_export(self.hooks.before_export, context, result)
            if self.config.save_json:
                self.artifacts.save_json("structured_output.json", result.structured_json)
            if self.config.save_metrics:
                self.artifacts.save_json("metrics.json", result.runtime_metrics)
            self.artifacts.save_json("integration_report.json", result.integration_report)
            self.metrics.successful_runs += 1
            return result
        except Exception as exc:
            result.errors.append(str(exc))
            raise StageExecutionError(str(exc)) from exc

    def run_batch(
        self,
        documents: list[str | Path | Image.Image],
        *,
        metadata_list: list[dict[str, Any] | None] | None = None,
        mock_detections_list: list[list[DetectionPrediction] | None] | None = None,
    ) -> list[PipelineResult]:
        results: list[PipelineResult] = []
        for index, document in enumerate(documents):
            metadata = metadata_list[index] if metadata_list and index < len(metadata_list) else None
            mock_detections = mock_detections_list[index] if mock_detections_list and index < len(mock_detections_list) else None
            results.append(self.run(document, metadata=metadata, mock_detections=mock_detections))
        return results

    def _parser_to_dict(self, index: int, det: DetectionPrediction, cls: dict[str, Any], route: Any, parsed: ParserResult) -> dict[str, Any]:
        return {
            "index": index,
            "detection": {
                "bbox": list(det.bbox),
                "confidence": det.confidence,
                "class_name": det.class_name,
                "backend_name": det.backend_name,
                "image_id": det.image_id,
                "metadata": dict(det.metadata),
            },
            "classification": dict(cls),
            "routing": {
                "selected_parser": route.selected_parser,
                "candidate_parsers": list(route.candidate_parsers),
                "confidence": route.confidence,
                "routing_reason": route.routing_reason,
                "fallback_used": route.fallback_used,
                "metadata": dict(route.metadata),
            },
            "parser_output": {
                "success": parsed.success,
                "chart_type": parsed.chart_type,
                "confidence": parsed.confidence,
                "structured_data": parsed.structured_data,
                "warnings": list(parsed.warnings),
                "execution_time": parsed.execution_time,
                "provenance": dict(parsed.provenance),
            },
        }
