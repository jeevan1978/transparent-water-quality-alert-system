"""
database.py
===========
Data access layer for water quality telemetry, predictions, and alerts.
Supports Supabase PostgreSQL when credentials exist, and falls back to
a local SQLite database so beginners can run and test offline with zero setup.

CRITICAL DISCLAIMER:
--------------------
ALL DATA STORED HERE REPRESENTS SYNTHETIC / SIMULATED BENCHMARK DATA.
"""

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional
import uuid
import pandas as pd
from dotenv import load_dotenv

# Ensure .env is loaded reliably regardless of working directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

SQLITE_PATH = os.path.join(PROJECT_ROOT, "data", "local_water_quality.db")


class WaterDatabase:
    """
    Unified database client that interfaces with Supabase PostgreSQL if
    configured, or defaults to an offline local SQLite database.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        mode: Optional[str] = None,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        supabase_client: Optional[Any] = None
    ):
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        self.client = supabase_client
        self.db_path = db_path or SQLITE_PATH

        # Explicit mode override or explicit db_path passed (e.g. in tests)
        if mode:
            self.mode = mode
            if self.mode == "supabase" and not self.client and self.supabase_url and self.supabase_key:
                try:
                    from supabase import create_client
                    self.client = create_client(self.supabase_url, self.supabase_key)
                except Exception as e:
                    print(f"[Database] Could not connect to Supabase ({e}).")
        elif supabase_client:
            self.client = supabase_client
            self.mode = "supabase"
        elif db_path:
            # An explicit SQLite file path was provided (e.g. pytest isolated fixture)
            self.mode = "sqlite"
        elif (
            self.supabase_url
            and self.supabase_key
            and "your-project" not in self.supabase_url
        ):
            try:
                from supabase import create_client
                self.client = create_client(self.supabase_url, self.supabase_key)
                self.mode = "supabase"
            except Exception as e:
                print(f"[Database] Could not connect to Supabase ({e}). Falling back to local SQLite.")
                self.mode = "sqlite"
        else:
            self.mode = "sqlite"

        # Initialize SQLite tables only when SQLite is the chosen backend
        if self.mode == "sqlite":
            self._init_sqlite()

    def _get_sqlite_conn(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        return sqlite3.connect(self.db_path)

    def _safe_migrate(self, conn: sqlite3.Connection):
        """Safely adds missing columns and indices to existing SQLite tables."""
        cursor = conn.cursor()

        # Migrate predictions table
        cursor.execute("PRAGMA table_info(predictions)")
        pred_cols = [col[1] for col in cursor.fetchall()]
        if "target_timestamp" not in pred_cols and len(pred_cols) > 0:
            cursor.execute("ALTER TABLE predictions ADD COLUMN target_timestamp TEXT")

        # Migrate alerts table
        cursor.execute("PRAGMA table_info(alerts)")
        alert_cols = [col[1] for col in cursor.fetchall()]
        if "alert_source" not in alert_cols and len(alert_cols) > 0:
            cursor.execute("ALTER TABLE alerts ADD COLUMN alert_source TEXT DEFAULT 'CURRENT'")
        if "dedup_key" not in alert_cols and len(alert_cols) > 0:
            cursor.execute("ALTER TABLE alerts ADD COLUMN dedup_key TEXT")

        # Ensure index on alerts.dedup_key exists
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_alerts_dedup_key ON alerts(dedup_key)")
        conn.commit()

    def _init_sqlite(self):
        """Initializes tables in local SQLite database if not present."""
        with self._get_sqlite_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    station_id TEXT,
                    ph REAL,
                    turbidity REAL,
                    dissolved_oxygen REAL,
                    temperature REAL,
                    conductivity REAL,
                    total_dissolved_solids REAL,
                    quality_label TEXT,
                    is_synthetic INTEGER DEFAULT 1
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE,
                    dedup_key TEXT UNIQUE,
                    alert_source TEXT DEFAULT 'CURRENT',
                    timestamp TEXT,
                    station_id TEXT,
                    severity TEXT,
                    alert_type TEXT,
                    title TEXT,
                    description TEXT,
                    recommended_action TEXT,
                    parameters TEXT,
                    is_resolved INTEGER DEFAULT 0
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT,
                    target_timestamp TEXT,
                    station_id TEXT,
                    horizon_hours INTEGER,
                    predicted_category TEXT,
                    forecasted_params TEXT
                )
            """)
            conn.commit()
            self._safe_migrate(conn)

    def insert_reading(self, reading: Dict[str, Any]) -> bool:
        """Saves a telemetry reading into the active database backend."""
        if self.mode == "supabase":
            if not self.client:
                print("[Database] Supabase client is not connected.")
                return False
            try:
                payload = {
                    "timestamp": str(reading.get("timestamp")),
                    "station_id": str(reading.get("station_id")),
                    "ph": float(reading.get("ph", 7.0)),
                    "turbidity": float(reading.get("turbidity", 1.0)),
                    "dissolved_oxygen": float(reading.get("dissolved_oxygen", 7.0)),
                    "temperature": float(reading.get("temperature", 20.0)),
                    "conductivity": float(reading.get("conductivity", 300.0)),
                    "total_dissolved_solids": float(reading.get("total_dissolved_solids", 200.0)),
                    "quality_label": str(reading.get("quality_label", "Good")),
                    "is_synthetic": 1 if reading.get("is_synthetic", True) else 0
                }
                self.client.table("telemetry_readings").insert(payload).execute()
                return True
            except Exception as e:
                print(f"[Supabase Insert Error]: {e}")
                return False

        # SQLite backend
        with self._get_sqlite_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO telemetry_readings (
                    timestamp, station_id, ph, turbidity, dissolved_oxygen,
                    temperature, conductivity, total_dissolved_solids,
                    quality_label, is_synthetic
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(reading.get("timestamp")),
                str(reading.get("station_id")),
                float(reading.get("ph", 7.0)),
                float(reading.get("turbidity", 1.0)),
                float(reading.get("dissolved_oxygen", 7.0)),
                float(reading.get("temperature", 20.0)),
                float(reading.get("conductivity", 300.0)),
                float(reading.get("total_dissolved_solids", 200.0)),
                str(reading.get("quality_label", "Good")),
                1 if reading.get("is_synthetic", True) else 0
            ))
            conn.commit()
            return True

    def get_recent_readings(self, station_id: str, limit: int = 100) -> pd.DataFrame:
        """Fetches the latest N readings for a monitoring station."""
        if self.mode == "supabase":
            if not self.client:
                print("[Database] Supabase client is not connected.")
                return pd.DataFrame()
            try:
                res = self.client.table("telemetry_readings") \
                    .select("*") \
                    .eq("station_id", station_id) \
                    .order("timestamp", desc=True) \
                    .limit(limit) \
                    .execute()
                if res.data:
                    df = pd.DataFrame(res.data)
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    return df.sort_values("timestamp").reset_index(drop=True)
                return pd.DataFrame()
            except Exception as e:
                print(f"[Supabase Query Error]: {e}")
                return pd.DataFrame()

        # SQLite backend
        with self._get_sqlite_conn() as conn:
            query = """
                SELECT * FROM telemetry_readings
                WHERE station_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(station_id, limit))
            if not df.empty and "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp").reset_index(drop=True)
            return df

    def insert_prediction(self, prediction: Dict[str, Any]) -> bool:
        """
        Saves a 6-hour or 24-hour prediction into the active database backend.
        Returns True if inserted, False if duplicate or failed.
        """
        station_id = str(prediction.get("station_id", "Station-Alpha"))
        horizon_hours = int(prediction.get("horizon_hours", 6))
        predicted_category = str(prediction.get("predicted_category", "Good"))
        raw_params = prediction.get("forecasted_params", {})

        if isinstance(raw_params, str):
            try:
                parsed_params = json.loads(raw_params)
            except Exception:
                parsed_params = {}
        elif isinstance(raw_params, dict):
            parsed_params = raw_params
        else:
            parsed_params = {}

        created_at = prediction.get("created_at")
        if created_at is None:
            created_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        elif hasattr(created_at, "strftime"):
            created_at = created_at.strftime("%Y-%m-%d %H:%M:%S")
        else:
            created_at = str(created_at)

        target_timestamp = prediction.get("target_timestamp")
        if target_timestamp is None:
            try:
                base_dt = pd.to_datetime(created_at)
                target_dt = base_dt + pd.Timedelta(hours=horizon_hours)
                target_timestamp = target_dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                target_timestamp = created_at
        elif hasattr(target_timestamp, "strftime"):
            target_timestamp = target_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        else:
            target_timestamp = str(target_timestamp)

        if self.mode == "supabase":
            if not self.client:
                print("[Database] Supabase client is not connected.")
                return False
            try:
                # Deduplication check for target forecast
                res = self.client.table("predictions") \
                    .select("id") \
                    .eq("station_id", station_id) \
                    .eq("horizon_hours", horizon_hours) \
                    .eq("target_timestamp", target_timestamp) \
                    .execute()
                if res.data and len(res.data) > 0:
                    return False

                payload = {
                    "created_at": created_at,
                    "target_timestamp": target_timestamp,
                    "station_id": station_id,
                    "horizon_hours": horizon_hours,
                    "predicted_category": predicted_category,
                    "forecasted_params": parsed_params
                }
                self.client.table("predictions").insert(payload).execute()
                return True
            except Exception as e:
                err_str = str(e).lower()
                if "duplicate" in err_str or "unique" in err_str or "23505" in err_str or "409" in err_str:
                    return False
                print(f"[Supabase Prediction Insert Error]: {e}")
                return False

        # SQLite backend
        with self._get_sqlite_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id FROM predictions
                WHERE station_id = ? AND horizon_hours = ? AND target_timestamp = ?
            """, (station_id, horizon_hours, target_timestamp))
            if cursor.fetchone():
                return False

            cursor.execute("""
                INSERT INTO predictions (
                    created_at, target_timestamp, station_id,
                    horizon_hours, predicted_category, forecasted_params
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                created_at,
                target_timestamp,
                station_id,
                horizon_hours,
                predicted_category,
                json.dumps(parsed_params)
            ))
            conn.commit()
            return True

    def get_recent_predictions(self, station_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetches the latest stored forecasts/predictions from the active backend."""
        if self.mode == "supabase":
            if not self.client:
                print("[Database] Supabase client is not connected.")
                return []
            try:
                query = self.client.table("predictions").select("*")
                if station_id:
                    query = query.eq("station_id", station_id)
                res = query.order("id", desc=True).limit(limit).execute()
                results = []
                for r in (res.data or []):
                    item = dict(r)
                    params = item.get("forecasted_params")
                    if isinstance(params, dict):
                        item["parsed_params"] = params
                    elif isinstance(params, str):
                        try:
                            item["parsed_params"] = json.loads(params)
                        except Exception:
                            item["parsed_params"] = {}
                    else:
                        item["parsed_params"] = {}
                    results.append(item)
                return results
            except Exception as e:
                print(f"[Supabase Prediction Retrieval Error]: {e}")
                return []

        # SQLite backend
        with self._get_sqlite_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if station_id:
                cursor.execute("""
                    SELECT * FROM predictions
                    WHERE station_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                """, (station_id, limit))
            else:
                cursor.execute("""
                    SELECT * FROM predictions
                    ORDER BY id DESC
                    LIMIT ?
                """, (limit,))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                if "forecasted_params" in item and isinstance(item["forecasted_params"], str):
                    try:
                        item["parsed_params"] = json.loads(item["forecasted_params"])
                    except Exception:
                        item["parsed_params"] = {}
                results.append(item)
            return results

    def insert_alert(self, alert_dict: Dict[str, Any]) -> bool:
        """
        Saves an alert record into the active database backend with deduplication.
        Returns True if a new alert was inserted, False if duplicate or failed.
        """
        st_id = str(alert_dict.get("station_id", "unknown"))
        a_type = str(alert_dict.get("alert_type", "THRESHOLD_BREACH"))
        severity = str(alert_dict.get("severity", "WARNING"))
        title = str(alert_dict.get("title", "Alert"))
        description = str(alert_dict.get("description", ""))
        rec_action = str(alert_dict.get("recommended_action", ""))

        alert_source = alert_dict.get("alert_source")
        if not alert_source:
            if a_type == "PREDICTIVE_DEGRADATION":
                alert_source = "PREDICTIVE"
            else:
                alert_source = "CURRENT"

        raw_ts = alert_dict.get("timestamp") or pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        ts_str = raw_ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(raw_ts, "strftime") else str(raw_ts)

        raw_params = alert_dict.get("parameters", {})
        if isinstance(raw_params, str):
            try:
                parsed_params = json.loads(raw_params)
            except Exception:
                parsed_params = {}
        elif isinstance(raw_params, dict):
            parsed_params = raw_params
        else:
            parsed_params = {}

        dedup_key = alert_dict.get("dedup_key")
        if not dedup_key:
            param_key = "_".join(sorted(parsed_params.keys())) if parsed_params else "general"
            window = ts_str[:13] + ":00" if len(ts_str) >= 13 else ts_str
            dedup_key = f"{st_id}:{a_type}:{severity}:{param_key}:{window}"

        alert_id = alert_dict.get("alert_id") or f"ALT-{uuid.uuid4().hex[:8].upper()}"

        if self.mode == "supabase":
            if not self.client:
                print("[Database] Supabase client is not connected.")
                return False
            try:
                # 1. Pre-insert deduplication query
                res = self.client.table("alerts").select("id").eq("dedup_key", dedup_key).execute()
                if res.data and len(res.data) > 0:
                    return False

                payload = {
                    "alert_id": alert_id,
                    "dedup_key": dedup_key,
                    "alert_source": alert_source,
                    "timestamp": ts_str,
                    "station_id": st_id,
                    "severity": severity,
                    "alert_type": a_type,
                    "title": title,
                    "description": description,
                    "recommended_action": rec_action,
                    "parameters": parsed_params,
                    "is_resolved": int(alert_dict.get("is_resolved", 0))
                }
                self.client.table("alerts").insert(payload).execute()
                return True
            except Exception as e:
                # 2. Handle DB unique constraint violation safely
                err_str = str(e).lower()
                if "duplicate" in err_str or "unique" in err_str or "23505" in err_str or "409" in err_str:
                    return False
                print(f"[Supabase Alert Insert Error]: {e}")
                return False

        # SQLite backend
        with self._get_sqlite_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM alerts WHERE dedup_key = ?", (dedup_key,))
            if cursor.fetchone():
                return False

            cursor.execute("""
                INSERT OR IGNORE INTO alerts (
                    alert_id, dedup_key, alert_source, timestamp, station_id, severity, alert_type,
                    title, description, recommended_action, parameters
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert_id,
                dedup_key,
                alert_source,
                ts_str,
                st_id,
                severity,
                a_type,
                title,
                description,
                rec_action,
                json.dumps(parsed_params)
            ))
            conn.commit()
            return cursor.rowcount > 0

    def get_active_alerts(
        self,
        station_id: Optional[str] = None,
        alert_source: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Returns recent active alerts with optional filtering by station and alert source."""
        if self.mode == "supabase":
            if not self.client:
                print("[Database] Supabase client is not connected.")
                return []
            try:
                query = self.client.table("alerts").select("*")
                if station_id:
                    query = query.eq("station_id", station_id)
                if alert_source:
                    query = query.eq("alert_source", alert_source)
                res = query.order("id", desc=True).limit(limit).execute()
                return res.data or []
            except Exception as e:
                print(f"[Supabase Alerts Retrieval Error]: {e}")
                return []

        # SQLite backend
        with self._get_sqlite_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            query = "SELECT * FROM alerts WHERE 1=1"
            params: List[Any] = []

            if station_id:
                query += " AND station_id = ?"
                params.append(station_id)
            if alert_source:
                query += " AND alert_source = ?"
                params.append(alert_source)

            query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
