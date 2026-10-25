"""
test_persistence.py
===================
Unit and integration tests for database persistence, forecast storage,
schema migration, alert deduplication, and source distinction.
All tests run against isolated temporary SQLite databases.
"""

from datetime import datetime, timedelta
import os
import sqlite3
import pytest

from src.database import WaterDatabase, SQLITE_PATH
from src.alert_engine import WaterAlertEngine, Alert


@pytest.fixture
def temp_db(tmp_path):
    """Provides an isolated temporary database for testing."""
    test_db_path = str(tmp_path / "test_persistence.db")
    db = WaterDatabase(db_path=test_db_path)
    return db, test_db_path


def test_forecast_insertion_and_retrieval(temp_db):
    """
    Verify forecasts (6h and 24h) are persisted with station ID, creation timestamp,
    target timestamp, horizon, predicted category, and numeric parameters.
    """
    db, db_path = temp_db
    created_at = "2026-10-25 10:00:00"
    target_6h = "2026-10-25 16:00:00"
    target_24h = "2026-10-26 10:00:00"

    p6 = {
        "station_id": "Station-Alpha",
        "created_at": created_at,
        "target_timestamp": target_6h,
        "horizon_hours": 6,
        "predicted_category": "Good",
        "forecasted_params": {"dissolved_oxygen": 7.2, "ph": 7.4, "turbidity": 1.5}
    }

    p24 = {
        "station_id": "Station-Alpha",
        "created_at": created_at,
        "target_timestamp": target_24h,
        "horizon_hours": 24,
        "predicted_category": "Needs Treatment",
        "forecasted_params": {"dissolved_oxygen": 5.1, "ph": 7.1, "turbidity": 6.8}
    }

    assert db.insert_prediction(p6) is True
    assert db.insert_prediction(p24) is True

    # Retrieve forecasts for Station-Alpha
    stored = db.get_recent_predictions(station_id="Station-Alpha", limit=10)
    assert len(stored) == 2

    # Verify fields of retrieved records
    stored_24 = [p for p in stored if p["horizon_hours"] == 24][0]
    assert stored_24["station_id"] == "Station-Alpha"
    assert stored_24["created_at"] == created_at
    assert stored_24["target_timestamp"] == target_24h
    assert stored_24["predicted_category"] == "Needs Treatment"
    assert isinstance(stored_24["parsed_params"], dict)
    assert stored_24["parsed_params"]["dissolved_oxygen"] == 5.1
    assert stored_24["parsed_params"]["turbidity"] == 6.8

    # Deduplication test: re-inserting identical target forecast returns False
    assert db.insert_prediction(p6) is False
    assert len(db.get_recent_predictions(station_id="Station-Alpha")) == 2


def test_forecast_schema_migration(tmp_path):
    """
    Verify safe schema migration when the predictions table was created
    by an earlier schema version without target_timestamp.
    """
    legacy_db_path = str(tmp_path / "legacy.db")
    # Manually create old schema without target_timestamp
    with sqlite3.connect(legacy_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                station_id TEXT,
                horizon_hours INTEGER,
                predicted_category TEXT,
                forecasted_params TEXT
            )
        """)
        conn.commit()

    # Initializing WaterDatabase should safely migrate and add target_timestamp
    db = WaterDatabase(db_path=legacy_db_path)

    with sqlite3.connect(legacy_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(predictions)")
        cols = [col[1] for col in cursor.fetchall()]
        assert "target_timestamp" in cols

    # Verify insert works on migrated schema
    res = db.insert_prediction({
        "station_id": "Station-Migrated",
        "created_at": "2026-10-25 12:00:00",
        "target_timestamp": "2026-10-25 18:00:00",
        "horizon_hours": 6,
        "predicted_category": "Good",
        "forecasted_params": {"ph": 7.5}
    })
    assert res is True


def test_alert_deduplication(temp_db):
    """
    Verify alert deduplication prevents repeated button clicks or reruns
    from storing duplicate alerts for the same underlying event.
    """
    db, _ = temp_db
    engine = WaterAlertEngine()

    reading = {
        "timestamp": "2026-10-25 12:00:00",
        "station_id": "Station-Beta",
        "dissolved_oxygen": 3.2,  # Critical hypoxia
        "ph": 7.2,
        "turbidity": 1.0,
        "conductivity": 300.0
    }

    # Evaluate alerts
    alerts_1 = engine.evaluate_current_reading("Station-Beta", reading)
    assert len(alerts_1) == 1
    alert_obj = alerts_1[0]

    # First insertion must succeed
    assert db.insert_alert(alert_obj.to_dict()) is True
    assert len(db.get_active_alerts("Station-Beta")) == 1

    # Simulate immediate second click (rerun / duplicate injection in same event window)
    # Even with a freshly generated random UUID, dedup_key must catch it
    alerts_2 = engine.evaluate_current_reading("Station-Beta", reading)
    alert_duplicate = alerts_2[0]
    assert alert_duplicate.alert_id != alert_obj.alert_id  # Unique random alert_id preserved
    assert alert_duplicate.dedup_key == alert_obj.dedup_key  # Same dedup_key

    # Second insertion must be suppressed
    assert db.insert_alert(alert_duplicate.to_dict()) is False
    assert len(db.get_active_alerts("Station-Beta")) == 1

    # Distinct legitimate event: different parameter breach (e.g. acid spike)
    acid_reading = {
        "timestamp": "2026-10-25 12:00:00",
        "station_id": "Station-Beta",
        "dissolved_oxygen": 7.0,
        "ph": 4.8,  # Critical acid breach
        "turbidity": 1.0,
        "conductivity": 300.0
    }
    acid_alerts = engine.evaluate_current_reading("Station-Beta", acid_reading)
    assert len(acid_alerts) == 1
    assert db.insert_alert(acid_alerts[0].to_dict()) is True
    assert len(db.get_active_alerts("Station-Beta")) == 2

    # Distinct legitimate event: same parameter but in a different event window (next day)
    later_reading = dict(reading)
    later_reading["timestamp"] = "2026-10-26 12:00:00"
    later_alerts = engine.evaluate_current_reading("Station-Beta", later_reading)
    assert len(later_alerts) == 1
    assert db.insert_alert(later_alerts[0].to_dict()) is True
    assert len(db.get_active_alerts("Station-Beta")) == 3


def test_alert_source_distinction(temp_db):
    """
    Verify CURRENT threshold alerts and PREDICTIVE alerts are explicitly
    distinguished in stored records and retrieval queries.
    """
    db, _ = temp_db
    engine = WaterAlertEngine()

    # Current threshold alert
    current_reading = {
        "timestamp": "2026-10-25 09:00:00",
        "dissolved_oxygen": 2.5,
        "ph": 7.0,
        "turbidity": 2.0,
        "conductivity": 300.0
    }
    cur_alerts = engine.evaluate_current_reading("Station-Gamma", current_reading)
    assert len(cur_alerts) >= 1
    cur_alert = cur_alerts[0]
    assert cur_alert.alert_source == "CURRENT"
    assert db.insert_alert(cur_alert.to_dict()) is True

    # Predictive alert
    pred_alerts = engine.evaluate_predictive_forecast(
        station_id="Station-Gamma",
        horizon_hours=24,
        predicted_category="Poor",
        predicted_params={"dissolved_oxygen": 3.8, "turbidity": 18.0},
        forecast_timestamp=datetime(2026, 10, 25, 9, 0, 0)
    )
    assert len(pred_alerts) == 1
    pred_alert = pred_alerts[0]
    assert pred_alert.alert_source == "PREDICTIVE"
    assert db.insert_alert(pred_alert.to_dict()) is True

    # Check total stored
    all_stored = db.get_active_alerts("Station-Gamma")
    assert len(all_stored) == 2

    # Filter by source
    current_only = db.get_active_alerts("Station-Gamma", alert_source="CURRENT")
    assert len(current_only) == 1
    assert current_only[0]["alert_source"] == "CURRENT"
    assert current_only[0]["alert_type"] == "THRESHOLD_BREACH"

    predictive_only = db.get_active_alerts("Station-Gamma", alert_source="PREDICTIVE")
    assert len(predictive_only) == 1
    assert predictive_only[0]["alert_source"] == "PREDICTIVE"
    assert predictive_only[0]["alert_type"] == "PREDICTIVE_DEGRADATION"


def test_real_database_safety(tmp_path):
    """
    Verify tests execute against temporary paths and do not touch the real local SQLite database.
    """
    real_db_mtime_before = None
    if os.path.exists(SQLITE_PATH):
        real_db_mtime_before = os.path.getmtime(SQLITE_PATH)

    # Perform operations on isolated temp database
    test_db = WaterDatabase(db_path=str(tmp_path / "safety_check.db"))
    test_db.insert_prediction({
        "station_id": "Safety-Station",
        "created_at": "2026-10-25 00:00:00",
        "horizon_hours": 6,
        "predicted_category": "Good",
        "forecasted_params": {}
    })

    if real_db_mtime_before is not None:
        real_db_mtime_after = os.path.getmtime(SQLITE_PATH)
        assert real_db_mtime_before == real_db_mtime_after, "Real database was modified during test!"
