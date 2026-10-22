from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from src.parser_engine import (
    ParserContext,
    ParserEngine,
    ParserExecutionError,
    ParserFactory,
    ParserInitializationError,
    ParserLoader,
    ParserResult,
    ParserValidationError,
)
from src.router import BaseChartParser, RoutingResult, get_parser_registry


@dataclass
class StubParser(BaseChartParser):
    parser_name: str = "stub"

    def parse(self, image, *, metadata=None):
        return {"parsed": True, "metadata": metadata or {}}

    def validate(self, parsed_output) -> bool:
        return True

    def export(self, parsed_output, output_path):
        path = Path(output_path)
        path.write_text("ok", encoding="utf-8")
        return path


@dataclass
class BadParser(BaseChartParser):
    parser_name: str = "bad"

    def parse(self, image, *, metadata=None):
        raise ValueError("boom")

    def validate(self, parsed_output) -> bool:
        return False

    def export(self, parsed_output, output_path):
        return Path(output_path)


def _image() -> Image.Image:
    return Image.new("RGB", (32, 32), "white")


def test_factory_loader_and_context() -> None:
    registry = get_parser_registry()
    registry.register(StubParser())
    loader = ParserLoader()
    assert "stub" in loader.available()
    factory = ParserFactory()
    parser = factory.create("stub")
    assert parser.parser_name == "stub"
    ctx = ParserContext(image=_image(), metadata={"a": 1}, classifier_output={"chart_type": "stub"})
    assert ctx.metadata["a"] == 1


def test_engine_executes_and_validates() -> None:
    registry = get_parser_registry()
    registry.register(StubParser())
    engine = ParserEngine()
    routing = RoutingResult(selected_parser="stub", candidate_parsers=["stub"], confidence=0.9, routing_reason="test")
    result = engine.run(_image(), routing, metadata={"page": 1}, classifier_output={"chart_type": "stub", "confidence": 0.8})
    assert result.success is True
    assert result.chart_type == "stub"
    assert result.structured_data["parsed"] is True
    assert result.provenance["parser_name"] == "stub"


def test_engine_raises_for_missing_parser() -> None:
    engine = ParserEngine()
    routing = RoutingResult(selected_parser="missing", candidate_parsers=[], confidence=0.0, routing_reason="test")
    try:
        engine.run(_image(), routing)
        assert False, "expected ParserInitializationError"
    except ParserInitializationError:
        pass


def test_executor_wraps_failures() -> None:
    registry = get_parser_registry()
    registry.register(BadParser())
    engine = ParserEngine()
    routing = RoutingResult(selected_parser="bad", candidate_parsers=["bad"], confidence=0.1, routing_reason="test")
    try:
        engine.run(_image(), routing)
        assert False, "expected ParserExecutionError"
    except ParserExecutionError:
        pass


def test_validation_rejects_bad_result() -> None:
    engine = ParserEngine()
    bad = ParserResult(success=True, chart_type="", confidence=2.0, structured_data={})
    try:
        engine.validate(object(), bad)
        assert False, "expected ParserValidationError"
    except ParserValidationError:
        pass

