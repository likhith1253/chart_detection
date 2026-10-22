from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from src.router.base_parser import BaseChartParser

from .exceptions import ParserExecutionError
from .parser_context import ParserContext
from .parser_result import ParserResult


@dataclass
class ParserExecutor:
    def execute(self, parser: BaseChartParser, context: ParserContext) -> ParserResult:
        start = perf_counter()
        try:
            output = parser.parse(context.image, metadata={**context.metadata, **context.routing_information, "classifier_output": context.classifier_output})
            elapsed = perf_counter() - start
            confidence = float(context.routing_information.get("confidence", context.classifier_output.get("confidence", 0.0)))
            chart_type = str(context.routing_information.get("selected_parser", context.classifier_output.get("chart_type", parser.parser_name)))
            return ParserResult(
                success=True,
                chart_type=chart_type,
                confidence=confidence,
                structured_data=output,
                warnings=[],
                execution_time=elapsed,
                provenance={
                    "parser_name": parser.parser_name,
                    "routing_information": dict(context.routing_information),
                    "classifier_output": dict(context.classifier_output),
                },
            )
        except Exception as exc:
            raise ParserExecutionError(f"Parser execution failed for {parser.parser_name}") from exc

