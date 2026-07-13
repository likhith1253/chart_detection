"""
LLM-assisted chart reasoning module.

The module prefers an API-backed LLM when configured, and falls back to a
deterministic reasoning engine to keep the pipeline fully runnable offline.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ChartReasoningEngine:
    """Generate chart-level QA and natural-language explanations."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.backend = "heuristic"
        self._client = None

        key = os.getenv("OPENAI_API_KEY", "").strip()
        if key:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=key)
                self.backend = "openai"
            except Exception as exc:
                logger.warning("OpenAI client unavailable, fallback to heuristics: %s", exc)

    @staticmethod
    def _coerce_numeric_series(extracted_values) -> Tuple[List[str], List[float]]:
        if isinstance(extracted_values, dict):
            labels = list(extracted_values.keys())
            vals = []
            for key in labels:
                try:
                    vals.append(float(extracted_values[key]))
                except Exception:
                    vals.append(0.0)
            return labels, vals

        if isinstance(extracted_values, list):
            vals = []
            labels = []
            for i, item in enumerate(extracted_values):
                if isinstance(item, dict):
                    label = str(item.get("label", f"item_{i+1}"))
                    raw = item.get("value", item.get("normalized_value", item.get("estimated_percentage", 0.0)))
                else:
                    label = f"item_{i+1}"
                    raw = item
                try:
                    vals.append(float(raw))
                except Exception:
                    vals.append(0.0)
                labels.append(label)
            return labels, vals

        return [], []

    @staticmethod
    def _heuristic_qa(structured: Dict) -> List[Dict]:
        chart_type = structured.get("chart_type", "unknown")
        labels, values = ChartReasoningEngine._coerce_numeric_series(structured.get("values", {}))
        qa = []

        if not values:
            qa.append(
                {
                    "question": "What trend does the chart show?",
                    "answer": "Insufficient numeric extraction to infer a reliable trend.",
                }
            )
            qa.append(
                {
                    "question": "What is the highest value?",
                    "answer": "Unable to determine from extracted values.",
                }
            )
            qa.append(
                {
                    "question": "What category has the largest share?",
                    "answer": "Unable to determine from extracted values.",
                }
            )
            return qa

        arr = np.asarray(values, dtype=float)
        max_idx = int(np.argmax(arr))
        min_idx = int(np.argmin(arr))
        slope = float(arr[-1] - arr[0]) if len(arr) > 1 else 0.0

        if slope > 0:
            trend = "an increasing trend"
        elif slope < 0:
            trend = "a decreasing trend"
        else:
            trend = "a mostly stable trend"

        top_label = labels[max_idx] if labels else f"item_{max_idx+1}"
        low_label = labels[min_idx] if labels else f"item_{min_idx+1}"

        qa.append({"question": "What trend does the chart show?", "answer": f"The {chart_type} indicates {trend}."})
        qa.append(
            {
                "question": "What is the highest value?",
                "answer": f"{top_label} has the highest value at {arr[max_idx]:.2f}.",
            }
        )
        qa.append(
            {
                "question": "What category has the largest share?",
                "answer": f"{top_label} has the largest share, while {low_label} is the smallest.",
            }
        )
        return qa

    def _openai_qa(self, structured: Dict) -> List[Dict]:
        if self._client is None:
            return self._heuristic_qa(structured)

        prompt = (
            "You are a chart reasoning assistant.\n"
            "Given structured chart JSON, answer exactly three questions:\n"
            "1) What trend does the chart show?\n"
            "2) What is the highest value?\n"
            "3) What category has the largest share?\n"
            "Return strict JSON list: "
            '[{"question":"...","answer":"..."}].\n'
            f"Input JSON:\n{json.dumps(structured, ensure_ascii=True)}"
        )
        try:
            response = self._client.responses.create(
                model=self.model_name,
                input=prompt,
                max_output_tokens=350,
            )
            text = response.output_text.strip()
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except Exception as exc:
            logger.warning("OpenAI reasoning failed, using heuristic fallback: %s", exc)
        return self._heuristic_qa(structured)

    def reason_one(self, structured: Dict) -> Dict:
        qa = self._openai_qa(structured) if self.backend == "openai" else self._heuristic_qa(structured)
        summary = " ".join([str(item.get("answer", "")).strip() for item in qa]).strip()
        return {
            "image_name": structured.get("image_name", ""),
            "chart_type": structured.get("chart_type", "unknown"),
            "backend": self.backend,
            "questions": qa,
            "narrative": summary,
        }

    def reason_batch(self, structured_rows: List[Dict], max_items: int = 250) -> Dict:
        payload = []
        for row in structured_rows[:max_items]:
            payload.append(self.reason_one(row))
        return {
            "backend": self.backend,
            "num_items": len(payload),
            "items": payload,
        }
