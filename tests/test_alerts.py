"""
test_alerts.py
==============
Unit tests for real-time and predictive water quality alert generation.
"""

import pytest
from src.alert_engine import WaterAlertEngine, Alert


def test_alert_threshold_breach():
    """Verify severe parameter deviations trigger CRITICAL threshold breach alerts."""
    engine = WaterAlertEngine()

    critical_reading = {
        "ph": 5.2,  # Acidic breach
        "turbidity": 18.5,  # Excessive turbidity
        "dissolved_oxygen": 3.1,  # Hypoxic breach
        "conductivity": 950.0
    }

    alerts = engine.evaluate_current_reading("Station-Test", critical_reading)
    assert len(alerts) >= 3

    severities = [a.severity for a in alerts]
    assert "CRITICAL" in severities

    for a in alerts:
        assert isinstance(a, Alert)
        assert a.station_id == "Station-Test"
        assert a.alert_type == "THRESHOLD_BREACH"
        assert len(a.recommended_action) > 0


def test_predictive_degradation_alert():
    """Verify predictive alerts trigger when future category degrades."""
    engine = WaterAlertEngine()

    future_params = {"dissolved_oxygen": 3.5, "ph": 6.8, "turbidity": 12.0}
    alerts = engine.evaluate_predictive_forecast(
        station_id="Station-Forecast",
        horizon_hours=6,
        predicted_category="Poor",
        predicted_params=future_params
    )

    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.severity == "CRITICAL"
    assert alert.alert_type == "PREDICTIVE_DEGRADATION"
    assert "+6h" in alert.title
