from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from src.detection.runtime.cropper import ChartCrop
from src.detection.runtime.prediction_record import DetectionPrediction
from src.detection.runtime.runner import DetectionRunner
from src.parser_engine.parser_factory import ParserFactory
from src.parser_engine.engine import ParserEngine
from src.parser_engine.parser_result import ParserResult
from src.router.result import RoutingResult
from src.router.registry import ParserRegistry
from src.router.router import ParserRouter
from src.pipeline import ArtifactManager, PipelineConfiguration, PipelineHooks, PipelineMetrics, ResearchPipeline


class StubDetector:
    def predict(self, image):
        return [
            DetectionPrediction(bbox=[0, 0, 20, 20], confidence=0.9, class_name="chart", backend_name="stub", image_id="page1")
        ]


class StubClassifier:
    def classify_chart(self, image_path: str):
        return {"chart_type": "bar", "confidence": 0.87, "confidence_scores": {"bar": 0.87, "line": 0.1}}


class UnknownClassifier:
    def classify_chart(self, image_path: str):
        return {"chart_type": "unknown", "confidence": 0.05, "confidence_scores": {"unknown": 0.05}}


class StubParser:
    parser_name = "bar"

    def parse(self, image, *, metadata=None):
        return ParserResult(success=True, chart_type="bar", confidence=0.93, structured_data={"ok": True})

    def validate(self, parsed_output):
        return True

    def export(self, parsed_output, output_path):
        return Path(output_path)


class UnknownParser(StubParser):
    parser_name = "unknown"

    def parse(self, image, *, metadata=None):
        return ParserResult(success=False, chart_type="unknown", confidence=0.1, structured_data={"reason": "unknown"})


def test_pipeline_runs_end_to_end(tmp_path):
    image = Image.new("RGB", (64, 64), "white")
    registry = ParserRegistry()
    registry.register(StubParser())
    detector_runner = DetectionRunner(detector=StubDetector(), backend_name="stub")
    router = ParserRouter(registry=registry)
    parser_engine = ParserEngine(factory=ParserFactory(registry=registry))
    pipeline = ResearchPipeline(
        detector_runner=detector_runner,
        classifier=StubClassifier(),
        router=router,
        parser_engine=parser_engine,
        config=PipelineConfiguration(output_dir=tmp_path),
        artifacts=ArtifactManager(tmp_path),
        hooks=PipelineHooks(),
        metrics=PipelineMetrics(),
    )

    result = pipeline.run(image, metadata={"doc_id": "doc-1"})

    assert result.detected_charts
    assert result.classified_charts[0]["chart_type"] == "bar"
    assert result.parser_outputs[0].success is True
    assert result.runtime_metrics["detected_charts"] == 1
    assert (tmp_path / "structured_output.json").exists()
    assert (tmp_path / "metrics.json").exists()


def test_pipeline_hooks_fire_in_order(tmp_path):
    events = []

    def mark(name):
        def _hook(context, result=None):
            events.append(name)
        return _hook

    image = Image.new("RGB", (32, 32), "white")
    registry = ParserRegistry()
    registry.register(StubParser())
    pipeline = ResearchPipeline(
        detector_runner=DetectionRunner(detector=StubDetector(), backend_name="stub"),
        classifier=StubClassifier(),
        router=ParserRouter(registry=registry),
        parser_engine=ParserEngine(factory=ParserFactory(registry=registry)),
        config=PipelineConfiguration(output_dir=tmp_path, save_json=False, save_metrics=False),
        hooks=PipelineHooks(
            before_detection=[mark("before_detection")],
            after_detection=[mark("after_detection")],
            after_classification=[mark("after_classification")],
            after_routing=[mark("after_routing")],
            after_parsing=[mark("after_parsing")],
            before_export=[lambda context, result: events.append("before_export")],
        ),
    )

    pipeline.run(image)

    assert events == [
        "before_detection",
        "after_detection",
        "after_classification",
        "after_routing",
        "after_parsing",
        "before_export",
    ]


def test_pipeline_supports_mock_detections_and_empty_results(tmp_path):
    image = Image.new("RGB", (48, 48), "white")
    registry = ParserRegistry()
    registry.register(StubParser())
    pipeline = ResearchPipeline(
        detector_runner=DetectionRunner(detector=StubDetector(), backend_name="stub"),
        classifier=StubClassifier(),
        router=ParserRouter(registry=registry),
        parser_engine=ParserEngine(factory=ParserFactory(registry=registry)),
        config=PipelineConfiguration(output_dir=tmp_path / "mock", save_json=False, save_metrics=False),
    )

    mock = [
        DetectionPrediction(bbox=[1, 1, 12, 12], confidence=0.99, class_name="chart", backend_name="mock", image_id="x")
    ]
    result = pipeline.run(image, mock_detections=mock)
    assert len(result.detected_charts) == 1
    assert result.integration_report["detected_charts"] == 1

    empty = ResearchPipeline(
        detector_runner=DetectionRunner(detector=StubDetector(), backend_name="stub"),
        classifier=StubClassifier(),
        router=ParserRouter(registry=registry),
        parser_engine=ParserEngine(factory=ParserFactory(registry=registry)),
        config=PipelineConfiguration(output_dir=tmp_path / "empty", save_json=False, save_metrics=False),
    )
    empty_result = empty.run(
        image,
        mock_detections=[],
    )
    assert empty_result.detected_charts == []
    assert empty_result.parser_outputs == []
    assert empty_result.integration_report["parsed_charts"] == 0


def test_pipeline_batch_and_unknown_chart_type(tmp_path):
    registry = ParserRegistry()
    registry.register(UnknownParser())
    pipeline = ResearchPipeline(
        detector_runner=DetectionRunner(detector=StubDetector(), backend_name="stub"),
        classifier=UnknownClassifier(),
        router=ParserRouter(registry=registry),
        parser_engine=ParserEngine(factory=ParserFactory(registry=registry)),
        config=PipelineConfiguration(output_dir=tmp_path / "batch", save_json=False, save_metrics=False),
    )

    images = [Image.new("RGB", (24, 24), "white"), Image.new("RGB", (32, 32), "white")]
    results = pipeline.run_batch(
        images,
        mock_detections_list=[
            [DetectionPrediction(bbox=[0, 0, 8, 8], confidence=0.99, class_name="chart", backend_name="mock", image_id="a")],
            [DetectionPrediction(bbox=[0, 0, 10, 10], confidence=0.99, class_name="chart", backend_name="mock", image_id="b")],
        ],
    )

    assert len(results) == 2
    assert all(r.integration_report["detected_charts"] == 1 for r in results)


def test_pipeline_failure_is_wrapped(tmp_path):
    registry = ParserRegistry()
    registry.register(StubParser())
    pipeline = ResearchPipeline(
        detector_runner=DetectionRunner(detector=StubDetector(), backend_name="stub"),
        classifier=StubClassifier(),
        router=ParserRouter(registry=registry),
        parser_engine=ParserEngine(factory=ParserFactory(registry=registry)),
        config=PipelineConfiguration(output_dir=tmp_path / "failure", save_json=False, save_metrics=False),
    )

    image = Image.new("RGB", (32, 32), "white")
    import pytest

    with pytest.raises(Exception) as excinfo:
        pipeline.run(
            image,
            mock_detections=[DetectionPrediction(bbox=[0, 0, 10], confidence=0.99, class_name="chart", backend_name="mock", image_id="c")],
        )

    assert "StageExecutionError" in type(excinfo.value).__name__
