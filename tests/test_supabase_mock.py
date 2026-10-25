"""
test_supabase_mock.py
=====================
Unit tests for Supabase cloud database operations using mock clients.
Validates telemetry, prediction, and alert CRUD, deduplication, error handling,
and ensures no silent fallback to SQLite occurs when cloud operations fail.
Runs offline with zero live credentials required.
"""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from src.database import WaterDatabase


class MockSupabaseTable:
    """Mock builder for Supabase / PostgREST table queries."""

    def __init__(self, data=None, error=None):
        self._data = data or []
        self._error = error
        self.inserted_payloads = []
        self.last_query = {}

    def insert(self, payload):
        self.inserted_payloads.append(payload)
        return self

    def select(self, *columns):
        self.last_query["columns"] = columns
        return self

    def eq(self, column, value):
        self.last_query.setdefault("eq", {})[column] = value
        return self

    def order(self, column, desc=False):
        self.last_query["order"] = (column, desc)
        return self

    def limit(self, count):
        self.last_query["limit"] = count
        return self

    def execute(self):
        if self._error:
            raise self._error
        mock_response = MagicMock()
        mock_response.data = self._data
        return mock_response


class MockSupabaseClient:
    """Mock Supabase client managing mock tables."""

    def __init__(self, table_data=None, table_errors=None):
        self.table_data = table_data or {}
        self.table_errors = table_errors or {}
        self.tables = {}

    def table(self, table_name):
        if table_name not in self.tables:
            data = self.table_data.get(table_name, [])
            error = self.table_errors.get(table_name, None)
            self.tables[table_name] = MockSupabaseTable(data=data, error=error)
        return self.tables[table_name]


def test_supabase_initialization_with_credentials():
    """Verify WaterDatabase switches to supabase mode when credentials exist."""
    mock_client = MockSupabaseClient()
    db = WaterDatabase(
        mode="supabase",
        supabase_url="https://test-ref.supabase.co",
        supabase_key="test-key",
        supabase_client=mock_client
    )
    assert db.mode == "supabase"
    assert db.client is mock_client


def test_supabase_telemetry_reads_and_writes():
    """Verify telemetry readings are written and read through Supabase."""
    mock_data = [
        {
            "id": 1,
            "timestamp": "2026-10-25 12:00:00",
            "station_id": "Station-Test",
            "ph": 7.3,
            "turbidity": 1.1,
            "dissolved_oxygen": 6.9,
            "temperature": 21.0,
            "conductivity": 305.0,
            "total_dissolved_solids": 190.0,
            "quality_label": "Good",
            "is_synthetic": 1
        }
    ]
    mock_client = MockSupabaseClient(table_data={"telemetry_readings": mock_data})
    db = WaterDatabase(mode="supabase", supabase_client=mock_client)

    # Insert test
    reading = {
        "timestamp": "2026-10-25 12:00:00",
        "station_id": "Station-Test",
        "ph": 7.3,
        "turbidity": 1.1,
        "dissolved_oxygen": 6.9,
        "temperature": 21.0,
        "conductivity": 305.0,
        "total_dissolved_solids": 190.0,
        "quality_label": "Good"
    }
    assert db.insert_reading(reading) is True
    assert len(mock_client.table("telemetry_readings").inserted_payloads) == 1

    # Query test
    df = db.get_recent_readings("Station-Test", limit=10)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["ph"] == 7.3


def test_supabase_prediction_reads_and_writes():
    """Verify forecasts are serialized with JSONB and retrieved with parsing."""
    mock_data = [
        {
            "id": 10,
            "created_at": "2026-10-25 12:00:00",
            "target_timestamp": "2026-10-25 18:00:00",
            "station_id": "Station-Test",
            "horizon_hours": 6,
            "predicted_category": "Good",
            "forecasted_params": {"dissolved_oxygen": 7.1, "ph": 7.4}
        }
    ]
    mock_client = MockSupabaseClient(table_data={"predictions": []})
    db = WaterDatabase(mode="supabase", supabase_client=mock_client)

    # Insert test
    pred = {
        "station_id": "Station-Test",
        "created_at": "2026-10-25 12:00:00",
        "target_timestamp": "2026-10-25 18:00:00",
        "horizon_hours": 6,
        "predicted_category": "Good",
        "forecasted_params": {"dissolved_oxygen": 7.1, "ph": 7.4}
    }
    assert db.insert_prediction(pred) is True
    inserted = mock_client.table("predictions").inserted_payloads[0]
    assert inserted["station_id"] == "Station-Test"
    assert isinstance(inserted["forecasted_params"], dict)

    # Retrieval test
    mock_client.tables["predictions"]._data = mock_data
    preds = db.get_recent_predictions("Station-Test", limit=5)
    assert len(preds) == 1
    assert preds[0]["parsed_params"]["dissolved_oxygen"] == 7.1


def test_supabase_alert_deduplication_and_filtering():
    """Verify alert deduplication and filtering by station & source on Supabase."""
    mock_client = MockSupabaseClient(table_data={"alerts": []})
    db = WaterDatabase(mode="supabase", supabase_client=mock_client)

    alert_1 = {
        "alert_id": "ALT-12345678",
        "dedup_key": "Station-Test:THRESHOLD_BREACH:WARNING:do:window1",
        "alert_source": "CURRENT",
        "station_id": "Station-Test",
        "severity": "WARNING",
        "alert_type": "THRESHOLD_BREACH",
        "title": "Low DO",
        "parameters": {"dissolved_oxygen": 5.5}
    }

    # Initial insert succeeds
    assert db.insert_alert(alert_1) is True

    # Simulate existing row found in pre-check
    mock_client.tables["alerts"]._data = [{"id": 1, "dedup_key": alert_1["dedup_key"]}]
    assert db.insert_alert(alert_1) is False

    # Simulate duplicate key unique constraint error from Postgres (409 / 23505)
    mock_client.tables["alerts"]._data = []
    mock_client.tables["alerts"]._error = RuntimeError("duplicate key value violates unique constraint 23505")
    assert db.insert_alert(alert_1) is False


def test_no_silent_fallback_to_sqlite_on_cloud_failure(tmp_path):
    """
    CRITICAL: Verify that when Supabase fails, errors are reported and the system
    does NOT silently fall back or write records to SQLite.
    """
    isolated_sqlite = str(tmp_path / "should_stay_empty.db")
    mock_client = MockSupabaseClient(
        table_errors={"telemetry_readings": RuntimeError("Cloud connection timed out")}
    )
    db = WaterDatabase(
        db_path=isolated_sqlite,
        mode="supabase",
        supabase_client=mock_client
    )

    reading = {
        "timestamp": "2026-10-25 12:00:00",
        "station_id": "Station-CloudFail",
        "ph": 7.0
    }

    # Insert must fail gracefully without writing to SQLite
    res = db.insert_reading(reading)
    assert res is False

    # Verify SQLite was not silently written to
    db_sqlite = WaterDatabase(db_path=isolated_sqlite, mode="sqlite")
    sqlite_df = db_sqlite.get_recent_readings("Station-CloudFail")
    assert sqlite_df.empty, "Record was silently inserted into SQLite during cloud failure!"


def test_sqlite_fallback_when_unconfigured(tmp_path):
    """Verify that if Supabase is unconfigured, system operates normally on SQLite."""
    test_db_path = str(tmp_path / "local_test.db")
    with patch.dict("os.environ", {"SUPABASE_URL": "", "SUPABASE_KEY": ""}):
        db = WaterDatabase(db_path=test_db_path)
        assert db.mode == "sqlite"

        # Local insert and read
        assert db.insert_reading({"timestamp": "2026-10-25 10:00:00", "station_id": "Station-Local", "ph": 7.2}) is True
        df = db.get_recent_readings("Station-Local")
        assert len(df) == 1
