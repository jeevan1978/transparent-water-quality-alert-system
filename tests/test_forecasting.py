"""
test_forecasting.py
===================
Unit tests for 6-hour and 24-hour water quality forecasting modules.
"""

from datetime import datetime, timedelta
import pandas as pd
import pytest

from src.data_simulator import WaterDataSimulator
from src.forecasting import WaterParameterForecaster
from src.preprocessing import FEATURE_COLUMNS


@pytest.fixture
def sample_timeseries():
    """Generates 7 days of simulated telemetry for forecasting tests."""
    sim = WaterDataSimulator(station_id="Station-Test-Forecast", seed=42)
    df = sim.generate_timeseries(days=7, interval_minutes=60)
    return df


def test_forecast_parameters_fallback(sample_timeseries):
    """Verify fallback forecaster outputs valid numeric dictionary for all features."""
    forecaster = WaterParameterForecaster(horizon_hours=6)
    # Ensure test runs even without trained weights
    forecaster.model = None

    forecast = forecaster.forecast_parameters(sample_timeseries)
    assert isinstance(forecast, dict)
    for col in FEATURE_COLUMNS:
        assert col in forecast
        assert isinstance(forecast[col], (int, float))
        assert forecast[col] >= 0.0


def test_predict_future_category(sample_timeseries):
    """Verify future category projection outputs valid status and reason."""
    forecaster_6h = WaterParameterForecaster(horizon_hours=6)
    forecaster_6h.model = None

    cat_6h, reason_6h, params_6h = forecaster_6h.predict_future_category(sample_timeseries)
    assert cat_6h in ["Good", "Needs Treatment", "Poor"]
    assert "6h" in reason_6h
    assert isinstance(params_6h, dict)

    forecaster_24h = WaterParameterForecaster(horizon_hours=24)
    forecaster_24h.model = None
    cat_24h, reason_24h, params_24h = forecaster_24h.predict_future_category(sample_timeseries)
    assert cat_24h in ["Good", "Needs Treatment", "Poor"]
    assert "24h" in reason_24h
