from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from src.router.result import RoutingResult

from .exceptions import ParserValidationError
from .executor import ParserExecutor
from .parser_context import ParserContext
from .parser_factory import ParserFactory
from .parser_result import ParserResult


@dataclass
class ParserEngine:
    factory: ParserFactory = field(default_factory=ParserFactory)
    executor: ParserExecutor = field(default_factory=ParserExecutor)

    def _build_context(
        self,
        image: Image.Image,
        routing_result: RoutingResult,
        metadata: dict[str, Any] | None = None,
        classifier_output: dict[str, Any] | None = None,
    ) -> ParserContext:
        return ParserContext(
            image=image,
            metadata=dict(metadata or {}),
            classifier_output=dict(classifier_output or {}),
            routing_information={
                "selected_parser": routing_result.selected_parser,
                "candidate_parsers": list(routing_result.candidate_parsers),
                "confidence": routing_result.confidence,
                "routing_reason": routing_result.routing_reason,
                "fallback_used": routing_result.fallback_used,
                **dict(routing_result.metadata),
            },
        )

    def validate(self, parser: Any, result: ParserResult) -> bool:
        if not isinstance(result.success, bool):
            raise ParserValidationError("ParserResult.success must be boolean")
        if not isinstance(result.chart_type, str) or not result.chart_type:
            raise ParserValidationError("ParserResult.chart_type must be a non-empty string")
        if not isinstance(result.confidence, (int, float)):
            raise ParserValidationError("ParserResult.confidence must be numeric")
        if result.confidence < 0 or result.confidence > 1:
            raise ParserValidationError("ParserResult.confidence must be within [0, 1]")
        return True

    def run(
        self,
        image: Image.Image,
        routing_result: RoutingResult,
        metadata: dict[str, Any] | None = None,
        classifier_output: dict[str, Any] | None = None,
    ) -> ParserResult:
        parser = self.factory.create(routing_result.selected_parser)
        context = self._build_context(image, routing_result, metadata=metadata, classifier_output=classifier_output)
        result = self.executor.execute(parser, context)
        self.validate(parser, result)
        return result

