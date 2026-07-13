"""
ML Chart Classifier — Research-Grade.
Trains RandomForest, SVM, GradientBoosting on extracted features.
Provides cross-validation, confusion matrix, accuracy/precision/recall/F1.
Saves best model via joblib.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_predict
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

logger = logging.getLogger(__name__)


class MLChartClassifier:
    """
    ML-based chart type classifier.
    Trains on feature vectors extracted from the pipeline.
    """

    FEATURE_COLUMNS = [
        # From geometric_features
        "feature_bar_count", "feature_bar_width_variance",
        "feature_bar_spacing_variance", "feature_line_count",
        "feature_point_density", "feature_circle_count",
        "feature_edge_density", "feature_axis_orientation",
        "feature_contour_count", "feature_texture_entropy",
        "feature_bar_height_variance", "feature_bar_alignment_score",
        "feature_bin_adjacency_ratio", "feature_width_uniformity",
        "feature_polyline_score", "feature_slope_variance",
        "feature_scatter_spatial_var", "feature_nn_density",
        "feature_hough_circle_count", "feature_circular_symmetry",
        "feature_radial_density", "feature_tick_density",
        "feature_axis_confidence", "feature_fill_ratio_mean",
        "feature_aspect_ratio_var", "feature_color_diversity",
        "feature_saturation_mean", "feature_brightness_variance",
        "feature_line_continuity_score", "feature_scatter_cluster_count",
        "feature_polyline_vertex_count", "feature_bar_gap_ratio",
        "feature_histogram_uniformity_score", "feature_axis_numeric_density",
        "feature_edge_orientation_entropy", "feature_line_connectivity_score",
        "feature_slope_continuity_score", "feature_cluster_compactness",
    ]

    def __init__(self, models_dir: Path = None):
        import config
        self.models_dir = models_dir or (config.RESULT_DIR / "models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None

    def train_and_evaluate(self, csv_path: Path) -> Dict:
        """
        Train RF, SVM, GB on the features CSV.
        Returns dict of {model_name: {accuracy, precision, recall, f1, confusion_matrix}}.
        """
        df = pd.read_csv(csv_path)

        # Filter rows with known labels
        df = df[df["true_chart_type"] != "unknown"].copy()
        if len(df) < 20:
            logger.warning(f"Only {len(df)} labelled samples — need at least 20")
            return {}

        # Get available feature columns
        available = [c for c in self.FEATURE_COLUMNS if c in df.columns]
        if len(available) < 5:
            logger.error(f"Only {len(available)} feature columns found")
            return {}

        X = df[available].fillna(0).values
        y_raw = df["true_chart_type"].values

        # Encode labels
        y = self.label_encoder.fit_transform(y_raw)
        class_names = self.label_encoder.classes_.tolist()

        # Scale features
        X = self.scaler.fit_transform(X)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Define models
        models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=200, max_depth=15, random_state=42,
                class_weight="balanced", n_jobs=-1
            ),
            "SVM": SVC(
                kernel="rbf", C=10.0, gamma="scale",
                class_weight="balanced", random_state=42
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.1,
                random_state=42
            ),
        }

        results = {}
        best_acc = 0.0

        for name, model in models.items():
            logger.info(f"Training {name}...")

            # Fit
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

            # Cross-validation
            cv = StratifiedKFold(n_splits=min(5, len(np.unique(y_train))), shuffle=True, random_state=42)
            cv_preds = cross_val_predict(model, X, y, cv=cv)
            cv_acc = accuracy_score(y, cv_preds)

            results[name] = {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "cv_accuracy": round(cv_acc, 4),
                "confusion_matrix": cm.tolist(),
                "class_names": class_names,
                "report": classification_report(
                    y_test, y_pred,
                    target_names=class_names,
                    zero_division=0,
                    output_dict=True
                ),
            }

            logger.info(
                f"  {name}: acc={acc:.4f}, prec={prec:.4f}, "
                f"rec={rec:.4f}, f1={f1:.4f}, cv_acc={cv_acc:.4f}"
            )

            # Save model
            model_path = self.models_dir / f"{name.lower()}_model.joblib"
            joblib.dump(model, model_path)

            if acc > best_acc:
                best_acc = acc
                self.best_model = model
                self.best_model_name = name

        # Save scaler and encoder
        joblib.dump(self.scaler, self.models_dir / "scaler.joblib")
        joblib.dump(self.label_encoder, self.models_dir / "label_encoder.joblib")

        # Save feature importance (RF only)
        if "RandomForest" in models:
            rf = models["RandomForest"]
            importances = rf.feature_importances_
            feat_imp = sorted(
                zip(available, importances),
                key=lambda x: x[1], reverse=True
            )
            results["feature_importance"] = [
                {"feature": f.replace("feature_", ""), "importance": round(float(v), 4)}
                for f, v in feat_imp
            ]

        # Save results
        results_path = self.models_dir / "ml_training_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"ML training results saved to {results_path}")

        logger.info(f"\nBest model: {self.best_model_name} (accuracy={best_acc:.4f})")
        return results

    def predict(self, features: Dict[str, float]) -> str:
        """Predict chart type from feature dict using the best trained model."""
        if self.best_model is None:
            return "unknown"

        available = [c for c in self.FEATURE_COLUMNS
                     if c.replace("feature_", "") in features or c in features]
        vec = []
        for c in self.FEATURE_COLUMNS:
            key = c.replace("feature_", "")
            vec.append(features.get(key, features.get(c, 0.0)))

        X = np.array([vec])
        X = self.scaler.transform(X)
        pred = self.best_model.predict(X)
        return self.label_encoder.inverse_transform(pred)[0]

    def load_model(self, model_name: str = "randomforest") -> bool:
        """Load a previously trained model."""
        model_path = self.models_dir / f"{model_name}_model.joblib"
        scaler_path = self.models_dir / "scaler.joblib"
        encoder_path = self.models_dir / "label_encoder.joblib"

        if not all(p.exists() for p in [model_path, scaler_path, encoder_path]):
            logger.warning(f"Model files not found for {model_name}")
            return False

        self.best_model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)
        self.best_model_name = model_name
        logger.info(f"Loaded model: {model_name}")
        return True
