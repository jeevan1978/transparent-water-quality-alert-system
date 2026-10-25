"""
current_classifier.py
=====================
Trains and runs multi-class machine learning models to classify current
water telemetry into 'Good', 'Needs Treatment', or 'Poor'.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from src.labeling import determine_quality_label
from src.preprocessing import FEATURE_COLUMNS


DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "current_classifier.joblib"
)


class WaterQualityClassifier:
    """
    Random Forest Classifier for real-time water quality categorization.
    Includes seamless fallback to rule-based classification if model weights
    have not yet been trained.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.model: Optional[RandomForestClassifier] = None
        self.classes_: List[str] = ["Good", "Needs Treatment", "Poor"]
        self.feature_names: List[str] = FEATURE_COLUMNS

        # Attempt to load trained model if available on disk
        if os.path.exists(self.model_path):
            self.load()

    def train(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> Dict[str, any]:
        """
        Fits a Random Forest model on training features and quality labels.

        Args:
            X: Matrix of telemetry features.
            y: Vector of quality target strings ('Good', 'Needs Treatment', 'Poor').

        Returns:
            Dict[str, any]: Training summary metrics.
        """
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].values

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            class_weight="balanced"
        )
        self.model.fit(X, y)
        self.classes_ = list(self.model.classes_)

        y_pred = self.model.predict(X)
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        return report

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> List[str]:
        """
        Predicts quality category for one or more telemetry samples.

        Args:
            X: Telemetry features as numpy array or DataFrame.

        Returns:
            List[str]: List of predicted category strings.
        """
        # If model is loaded, use scikit-learn model
        if self.model is not None:
            if isinstance(X, pd.DataFrame):
                X_vals = X[self.feature_names].values
            else:
                X_vals = X
            return list(self.model.predict(X_vals))

        # Fallback to rule-based labeling if model is not yet trained
        predictions = []
        if isinstance(X, pd.DataFrame):
            for _, row in X.iterrows():
                cat, _ = determine_quality_label(row.to_dict())
                predictions.append(cat)
        else:
            for row in X:
                sample_dict = {
                    name: row[i] for i, name in enumerate(self.feature_names) if i < len(row)
                }
                cat, _ = determine_quality_label(sample_dict)
                predictions.append(cat)

        return predictions

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predicts class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            np.ndarray: Predicted probability distribution across classes.
        """
        if self.model is not None:
            if isinstance(X, pd.DataFrame):
                X_vals = X[self.feature_names].values
            else:
                X_vals = X
            return self.model.predict_proba(X_vals)

        # Fallback deterministic pseudo-probabilities
        preds = self.predict(X)
        probs = []
        for p in preds:
            row_prob = [0.0] * len(self.classes_)
            if p in self.classes_:
                idx = self.classes_.index(p)
                row_prob[idx] = 1.0
            else:
                row_prob[0] = 1.0
            probs.append(row_prob)
        return np.array(probs)

    def save(self, filepath: Optional[str] = None) -> str:
        """Saves fitted classifier to disk using joblib."""
        save_path = filepath or self.model_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(self.model, save_path)
        return save_path

    def load(self, filepath: Optional[str] = None) -> None:
        """Loads fitted classifier from disk."""
        load_path = filepath or self.model_path
        if os.path.exists(load_path):
            self.model = joblib.load(load_path)
            if hasattr(self.model, "classes_"):
                self.classes_ = list(self.model.classes_)


# TODO: Team Member 3 - Experiment with GradientBoosting or XGBoost classifiers
# TODO: Team Member 3 - Add hyperparameter tuning with GridSearchCV and cross-validation
