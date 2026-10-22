from __future__ import annotations


class PipelineError(RuntimeError):
    """Base class for pipeline failures."""


class StageExecutionError(PipelineError):
    """Raised when a pipeline stage fails during execution."""


class PipelineConfigurationError(PipelineError):
    """Raised when pipeline configuration is invalid."""

