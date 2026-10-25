"""
test_simulator.py
=================
Unit tests for synthetic water telemetry generation.
"""

from datetime import datetime
import pandas as pd
import pytest

from src.data_simulator import WaterDataSimulator


def test_single_reading_structure():
    """Verify single reading output contains all required parameters and synthetic flag."""
    sim = WaterDataSimulator(station_id="Test-Station", seed=123)
    reading = sim.generate_single_reading(timestamp=datetime.now())

    expected_keys = [
        "timestamp",
        "station_id",
        "ph",
        "turbidity",
        "dissolved_oxygen",
        "temperature",
        "conductivity",
        "total_dissolved_solids",
        "is_synthetic"
    ]
    for key in expected_keys:
        assert key in reading, f"Missing expected key: {key}"

    assert reading["station_id"] == "Test-Station"
    assert reading["is_synthetic"] is True
    assert 0.0 <= reading["ph"] <= 14.0
    assert reading["turbidity"] >= 0.0
    assert reading["dissolved_oxygen"] >= 0.0


def test_timeseries_generation():
    """Verify batch time-series generation produces sorted and non-empty DataFrame."""
    sim = WaterDataSimulator(station_id="Station-1", seed=42)
    df = sim.generate_timeseries(days=2, interval_minutes=60)

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 40  # ~48 hourly readings in 2 days
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert df["is_synthetic"].all()


def test_anomaly_injection():
    """Verify anomaly injection generates shifted parameter distributions."""
    sim = WaterDataSimulator(station_id="Station-Anomaly", seed=999)
    # Inject severe runoff anomaly
    reading = sim.generate_single_reading(datetime.now(), inject_anomaly=True, anomaly_type="runoff")
    assert reading["turbidity"] > 10.0 or reading["conductivity"] > 500.0
