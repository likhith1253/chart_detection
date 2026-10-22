from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from .context import PipelineContext
from .result import PipelineResult


Hook = Callable[[PipelineContext], None]
ExportHook = Callable[[PipelineContext, PipelineResult], None]


@dataclass
class PipelineHooks:
    before_detection: list[Hook] = field(default_factory=list)
    after_detection: list[Hook] = field(default_factory=list)
    after_classification: list[Hook] = field(default_factory=list)
    after_routing: list[Hook] = field(default_factory=list)
    after_parsing: list[Hook] = field(default_factory=list)
    before_export: list[ExportHook] = field(default_factory=list)

    def run(self, hooks: Iterable[Hook], context: PipelineContext) -> None:
        for hook in hooks:
            hook(context)

    def run_export(self, hooks: Iterable[ExportHook], context: PipelineContext, result: PipelineResult) -> None:
        for hook in hooks:
            hook(context, result)

