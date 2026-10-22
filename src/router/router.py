from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .base_parser import BaseChartParser
from .config import RouterConfig
from .fallback import FallbackRouter, MultipleParserExecution, RejectPrediction, UnknownParser
from .registry import ParserRegistry, get_parser_registry
from .result import RoutingResult
from .routing_strategy import RoutingStrategy, RoutingStrategyType


@dataclass
class ParserRouter:
    registry: ParserRegistry = field(default_factory=get_parser_registry)
    strategy: RoutingStrategy = field(default_factory=RoutingStrategy)
    fallback_router: FallbackRouter = field(default_factory=FallbackRouter)

    def route(
        self,
        image: Image.Image,
        classifier_prediction: str | None,
        confidence_scores: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> RoutingResult:
        available = list(self.registry.names())
        candidates, reason = self.strategy.select(
            classifier_name=classifier_prediction,
            confidence_scores=confidence_scores,
            available_parsers=available,
            metadata=metadata,
        )
        if not candidates:
            return RoutingResult(
                selected_parser="unknown",
                candidate_parsers=[],
                confidence=0.0,
                routing_reason="no_available_parsers",
                fallback_used=True,
                metadata=dict(metadata or {}),
            )

        selected = candidates[0]
        confidence = float(confidence_scores.get(selected, 0.0))
        fallback = self.fallback_router.choose(selected, confidence, candidates)
        if isinstance(fallback, MultipleParserExecution):
            return RoutingResult(
                selected_parser="multiple",
                candidate_parsers=fallback.parsers,
                confidence=confidence,
                routing_reason="fallback_multiple_parser_execution",
                fallback_used=True,
                metadata=dict(metadata or {}),
            )
        if isinstance(fallback, UnknownParser):
            return RoutingResult(
                selected_parser="unknown",
                candidate_parsers=candidates,
                confidence=confidence,
                routing_reason="fallback_unknown_parser",
                fallback_used=True,
                metadata={"reason": fallback.reason, **dict(metadata or {})},
            )
        if isinstance(fallback, RejectPrediction):
            return RoutingResult(
                selected_parser="reject",
                candidate_parsers=candidates,
                confidence=confidence,
                routing_reason=fallback.reason,
                fallback_used=True,
                metadata=dict(metadata or {}),
            )
        return RoutingResult(
            selected_parser=str(fallback),
            candidate_parsers=candidates,
            confidence=confidence,
            routing_reason=reason,
            fallback_used=False,
            metadata=dict(metadata or {}),
        )

    def get_parser(self, name: str) -> BaseChartParser:
        return self.registry.get(name)

