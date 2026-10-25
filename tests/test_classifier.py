"""
test_classifier.py
==================
Unit tests for water quality classification logic and model wrapper.
"""

import numpy as np
import pandas as pd
import pytest

from src.current_classifier import WaterQualityClassifier
from src.labeling import determine_quality_label, label_dataframe
from src.preprocessing import FEATURE_COLUMNS


def test_rule_based_quality_labeling():
    """Verify deterministic domain rules classify known good, treatment, and poor samples."""
    # Pristine sample -> Good
    good_sample = {
        "ph": 7.2,
        "turbidity": 1.5,
        "dissolved_oxygen": 7.8,
        "conductivity": 320.0,
        "total_dissolved_solids": 210.0,
    }
    cat_good, _ = determine_quality_label(good_sample)
    assert cat_good == "Good"

    # Moderate turbidity -> Needs Treatment
    treatment_sample = {
        "ph": 7.0,
        "turbidity": 7.5,
        "dissolved_oxygen": 6.5,
        "conductivity": 350.0,
        "total_dissolved_solids": 230.0,
    }
    cat_treatment, _ = determine_quality_label(treatment_sample)
    assert cat_treatment == "Needs Treatment"

    # Extreme hypoxia -> Poor
    poor_sample = {
        "ph": 7.1,
        "turbidity": 2.0,
        "dissolved_oxygen": 2.2,
        "conductivity": 400.0,
        "total_dissolved_solids": 260.0,
    }
    cat_poor, _ = determine_quality_label(poor_sample)
    assert cat_poor == "Poor"


def test_classifier_fallback_and_prediction():
    """Verify classifier predicts valid category even if no model file is loaded."""
    # Non-existent path forces rule fallback
    classifier = WaterQualityClassifier(model_path="/tmp/non_existent_model.joblib")

    test_df = pd.DataFrame([{
        "ph": 7.3,
        "turbidity": 1.2,
        "dissolved_oxygen": 8.0,
        "temperature": 21.0,
        "conductivity": 300.0,
        "total_dissolved_solids": 200.0
    }])

    preds = classifier.predict(test_df)
    assert len(preds) == 1
    assert preds[0] in ["Good", "Needs Treatment", "Poor"]
    assert preds[0] == "Good"


def test_classifier_fit_and_inference():
    """Verify classifier training and prediction pipeline on synthetic batch."""
    # Synthetic dataset
    np.random.seed(42)
    n_samples = 60
    X = np.random.uniform(low=[6.0, 0.5, 3.0, 15.0, 150.0, 100.0],
                          high=[8.5, 18.0, 10.0, 28.0, 1100.0, 700.0],
                          size=(n_samples, 6))
    y = np.random.choice(["Good", "Needs Treatment", "Poor"], size=n_samples)

    classifier = WaterQualityClassifier(model_path="/tmp/test_clf.joblib")
    metrics = classifier.train(X, y)
    assert "accuracy" in metrics

    preds = classifier.predict(X[:5])
    assert len(preds) == 5
    for p in preds:
        assert p in ["Good", "Needs Treatment", "Poor"]
