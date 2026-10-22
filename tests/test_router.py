from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from src.router import (
    BaseChartParser,
    FallbackRouter,
    MultipleParserExecution,
    ParserRouter,
    RoutingStrategy,
    RoutingStrategyType,
    UnknownParser,
    get_parser_registry,
)


@dataclass
class StubParser(BaseChartParser):
    parser_name: str = "bar"

    def parse(self, image, *, metadata=None):
        return {"ok": True}

    def validate(self, parsed_output) -> bool:
        return True

    def export(self, parsed_output, output_path):
        path = Path(output_path)
        path.write_text("ok", encoding="utf-8")
        return path


def _image() -> Image.Image:
    return Image.new("RGB", (32, 32), "white")


def test_registry_and_base_parser() -> None:
    registry = get_parser_registry()
    registry.register(StubParser())
    assert registry.has("bar")
    assert registry.get("bar").parser_name == "bar"


def test_highest_confidence_routing_and_fallback() -> None:
    registry = get_parser_registry()
    registry.register(StubParser())
    router = ParserRouter()
    result = router.route(_image(), "bar", {"bar": 0.9}, metadata={"source": "test"})
    assert result.selected_parser == "bar"
    assert result.fallback_used is False
    assert result.routing_reason in {"highest_confidence_ranking", "rule_based_classifier_match"}


def test_low_confidence_uses_fallback() -> None:
    registry = get_parser_registry()
    registry.register(StubParser())
    router = ParserRouter(fallback_router=FallbackRouter(confidence_threshold=0.95))
    result = router.route(_image(), "bar", {"bar": 0.1}, metadata={})
    assert result.fallback_used is True
    assert result.selected_parser in {"unknown", "multiple"}


def test_routing_strategy_modes() -> None:
    strategy = RoutingStrategy(strategy_type=RoutingStrategyType.ORACLE)
    candidates, reason = strategy.select("bar", {"bar": 0.5}, ["bar", "line"], metadata={"oracle_parser": "line"})
    assert candidates == ["line"]
    assert reason == "oracle_routing"

    strategy = RoutingStrategy(strategy_type=RoutingStrategyType.RULE_BASED)
    candidates, reason = strategy.select("bar", {"bar": 0.5}, ["bar", "line"], metadata={})
    assert candidates == ["bar"]
    assert reason == "rule_based_classifier_match"


def test_fallback_router_types() -> None:
    fb = FallbackRouter(confidence_threshold=0.7)
    assert isinstance(fb.choose("bar", 0.1, ["bar", "line"]), MultipleParserExecution)
    assert isinstance(fb.choose("bar", 0.1, ["bar"]), UnknownParser)

