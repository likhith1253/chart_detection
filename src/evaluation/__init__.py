"""Evaluation package."""

from .comparison import ComparisonResult, GroundTruthComparator
from .evaluation_engine import EvaluationEngine
from .ground_truth import GroundTruthInstance
from .leaderboard import Leaderboard
from .metrics import (
    MetricResult,
    category_accuracy,
    geometry_iou,
    memory_metric,
    ocr_accuracy,
    parsing_success_rate,
    point_accuracy,
    runtime_metric,
    series_accuracy,
    value_mae,
    value_rmse,
)
from .parser_evaluator import ParserEvaluationRecord, ParserEvaluator
from .report_generator import ReportGenerator
from .statistics import StatisticalAnalyzer
from .visualization import EvaluationVisualizer

