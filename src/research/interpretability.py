"""
Model interpretability utilities with SHAP fallback support.

If SHAP is unavailable, this module falls back to model-native importances or
permutation importance while preserving a consistent output schema.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)


class InterpretabilityAnalyzer:
    """Generate feature attribution artifacts for classical ML models."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _top_features(importances: np.ndarray, feature_names: List[str], top_k: int = 20) -> List[Dict]:
        pairs = sorted(
            [(feature_names[i], float(importances[i])) for i in range(len(feature_names))],
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]
        return [{"feature": p[0], "importance": round(p[1], 8)} for p in pairs]

    def _plot_importance(self, top_features: List[Dict], path: Path):
        if not top_features:
            return
        names = [t["feature"] for t in top_features][::-1]
        values = [t["importance"] for t in top_features][::-1]

        plt.figure(figsize=(10, 7))
        plt.barh(range(len(names)), values, color="#2a9d8f")
        plt.yticks(range(len(names)), names, fontsize=9)
        plt.xlabel("Importance")
        plt.title("Feature Importance / Attribution")
        plt.tight_layout()
        plt.savefig(path, dpi=220)
        plt.close()

    def _try_shap_tree(
        self,
        model,
        x_frame: pd.DataFrame,
        feature_names: List[str],
    ) -> Dict | None:
        try:
            import shap
        except Exception:
            return None

        try:
            sample = x_frame.sample(min(500, len(x_frame)), random_state=42) if len(x_frame) > 500 else x_frame
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample)

            if isinstance(shap_values, list):
                sv = np.mean([np.abs(s) for s in shap_values], axis=0)
            else:
                sv = np.abs(shap_values)

            if sv.ndim == 3:
                sv = sv.mean(axis=0)
            if sv.ndim == 2:
                mean_abs = sv.mean(axis=0)
            else:
                mean_abs = np.asarray(sv).reshape(-1)

            top = self._top_features(mean_abs, feature_names, top_k=20)
            return {
                "method": "shap_tree_explainer",
                "top_features": top,
                "sample_size": int(len(sample)),
            }
        except Exception as exc:
            logger.warning("SHAP attribution failed, using fallback: %s", exc)
            return None

    def analyze(
        self,
        model,
        x_frame: pd.DataFrame,
        y_true: np.ndarray,
        feature_names: List[str],
    ) -> Dict:
        """
        Produce interpretability outputs and return metadata.
        """
        if x_frame.empty or not feature_names:
            return {"method": "none", "top_features": []}

        result = self._try_shap_tree(model, x_frame, feature_names)

        if result is None:
            if hasattr(model, "feature_importances_"):
                importances = np.asarray(model.feature_importances_, dtype=float)
                method = "model_feature_importances"
            else:
                # Permutation fallback works for any sklearn-compatible model.
                perm = permutation_importance(
                    model,
                    x_frame,
                    y_true,
                    n_repeats=4,
                    random_state=42,
                    n_jobs=-1,
                    scoring="f1_weighted",
                )
                importances = np.asarray(perm.importances_mean, dtype=float)
                method = "permutation_importance"

            result = {
                "method": method,
                "top_features": self._top_features(importances, feature_names, top_k=20),
                "sample_size": int(len(x_frame)),
            }

        out_json = self.output_dir / "feature_attribution.json"
        out_png = self.output_dir / "feature_importance.png"
        out_json.write_text(json.dumps(result, indent=2))
        self._plot_importance(result.get("top_features", []), out_png)

        # Decision-level visualization surrogate:
        # show contribution by feature magnitude for a representative sample.
        try:
            sample_idx = 0
            row = x_frame.iloc[sample_idx]
            vals = np.abs(row.to_numpy(dtype=float))
            top_idx = np.argsort(vals)[::-1][:10]
            names = [feature_names[i] for i in top_idx][::-1]
            v = [float(vals[i]) for i in top_idx][::-1]

            plt.figure(figsize=(8, 5))
            plt.barh(range(len(names)), v, color="#e76f51")
            plt.yticks(range(len(names)), names, fontsize=9)
            plt.xlabel("|Feature value|")
            plt.title("Sample Decision Attribution (Proxy)")
            plt.tight_layout()
            plt.savefig(self.output_dir / "decision_visualization.png", dpi=220)
            plt.close()
        except Exception as exc:
            logger.warning("Decision visualization skipped: %s", exc)

        return result
