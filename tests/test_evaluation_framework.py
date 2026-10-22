from __future__ import annotations

from pathlib import Path

from src.evaluation import EvaluationEngine, GroundTruthInstance, ParserEvaluator, StatisticalAnalyzer


def test_metrics_and_aggregation(tmp_path: Path):
    engine = EvaluationEngine(tmp_path)
    gt = GroundTruthInstance(chart_type="bar", chart_id="c1", boxes=[{"x1": 0, "y1": 0, "x2": 10, "y2": 10}], values=[1.0], categories=["a"], series=[{"series_id": 1, "label": "s"}], ocr_text=["A"])
    pred = {"success": True, "boxes": [{"x1": 0, "y1": 0, "x2": 10, "y2": 10}], "values": [1.2], "categories": ["a"], "series": [{"series_id": 1, "label": "s"}], "ocr_text": ["A"]}
    record = engine.evaluate("bar", "bar", pred, gt, runtime_seconds=0.5, memory_megabytes=10.0)
    assert record.metrics["geometry_iou"] == 1.0
    agg = engine.aggregate()
    assert agg["overall"]["parsing_success_rate"] == 1.0
    out = engine.save("dataset-x", 123, {"python": "3.11"}, {"seed": 123})
    assert out["json"].exists()
    assert out["csv"].exists()
    assert out["leaderboard"].exists()
    assert out["pdf"].exists()


def test_statistics_and_evaluator():
    stats = StatisticalAnalyzer()
    summary = stats.summary([1.0, 2.0, 3.0])
    assert summary.mean == 2.0
    evaluator = ParserEvaluator()
    gt = GroundTruthInstance(chart_type="pie", chart_id="c2")
    record = evaluator.evaluate("pie", "pie", {"success": False}, gt)
    assert record.parser_name == "pie"

