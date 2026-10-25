"""
verify_supabase.py
==================
Diagnostic utility to safely verify Supabase cloud database connectivity,
table schemas, and insert/read/delete operations without affecting production or local data.
Cleans up only its own temporary diagnostic records.
"""

import sys
import os
import uuid
from datetime import datetime, timedelta

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import WaterDatabase


def run_verification():
    print("[1/5] Initializing database client...")
    db = WaterDatabase()

    if db.mode != "supabase" or not db.client:
        print(f"[ERROR] Database client is in '{db.mode}' mode. Supabase credentials not active.")
        print("Please ensure SUPABASE_URL and SUPABASE_KEY are properly configured in .env.")
        sys.exit(1)

    print("[SUCCESS] Connected to Supabase client successfully.")
    diag_station = f"Station-Diag-{uuid.uuid4().hex[:6]}"
    diag_records = {"telemetry": 0, "predictions": 0, "alerts": 0}

    try:
        # 1. Telemetry Test
        print(f"[2/5] Testing telemetry insert & query for station '{diag_station}'...")
        telemetry_payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "station_id": diag_station,
            "ph": 7.4,
            "turbidity": 1.2,
            "dissolved_oxygen": 6.8,
            "temperature": 21.5,
            "conductivity": 310.0,
            "total_dissolved_solids": 195.0,
            "quality_label": "Good",
            "is_synthetic": 1
        }
        if not db.insert_reading(telemetry_payload):
            raise RuntimeError("Failed to insert diagnostic telemetry reading into Supabase.")
        diag_records["telemetry"] += 1

        readings_df = db.get_recent_readings(diag_station, limit=5)
        if readings_df.empty or len(readings_df) < 1:
            raise RuntimeError("Failed to retrieve diagnostic telemetry reading from Supabase.")
        print("      -> Telemetry insert & retrieval confirmed.")

        # 2. Prediction Test
        print(f"[3/5] Testing prediction insert & query for station '{diag_station}'...")
        pred_payload = {
            "station_id": diag_station,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "target_timestamp": (datetime.now() + timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
            "horizon_hours": 6,
            "predicted_category": "Good",
            "forecasted_params": {"dissolved_oxygen": 6.9, "ph": 7.3}
        }
        if not db.insert_prediction(pred_payload):
            raise RuntimeError("Failed to insert diagnostic prediction into Supabase.")
        diag_records["predictions"] += 1

        # Duplicate prediction check
        if db.insert_prediction(pred_payload):
            raise RuntimeError("Deduplication failed: Duplicate prediction was not suppressed.")

        preds = db.get_recent_predictions(diag_station, limit=5)
        if not preds or len(preds) < 1:
            raise RuntimeError("Failed to retrieve diagnostic prediction from Supabase.")
        print("      -> Prediction insert, retrieval & deduplication confirmed.")

        # 3. Alert Test
        print(f"[4/5] Testing alert insert, retrieval & deduplication for station '{diag_station}'...")
        alert_payload = {
            "alert_id": f"ALT-{uuid.uuid4().hex[:8].upper()}",
            "station_id": diag_station,
            "severity": "WARNING",
            "alert_type": "THRESHOLD_BREACH",
            "alert_source": "CURRENT",
            "title": "Diagnostic Test Alert",
            "description": "Automated verification test alert.",
            "recommended_action": "Safe to delete.",
            "parameters": {"turbidity": 6.5},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dedup_key": f"{diag_station}:THRESHOLD_BREACH:WARNING:turbidity:diag"
        }
        if not db.insert_alert(alert_payload):
            raise RuntimeError("Failed to insert diagnostic alert into Supabase.")
        diag_records["alerts"] += 1

        # Test duplicate alert suppression
        duplicate_payload = dict(alert_payload)
        duplicate_payload["alert_id"] = f"ALT-{uuid.uuid4().hex[:8].upper()}"
        if db.insert_alert(duplicate_payload):
            raise RuntimeError("Deduplication failed: Duplicate alert with same dedup_key was not suppressed.")

        alerts = db.get_active_alerts(diag_station, alert_source="CURRENT", limit=5)
        if not alerts or len(alerts) < 1:
            raise RuntimeError("Failed to retrieve diagnostic alert from Supabase.")
        print("      -> Alert insert, retrieval & deduplication confirmed.")

        print("[5/5] Cleaning up diagnostic test records from Supabase...")
    finally:
        # Cleanup: only delete records with this specific diagnostic station_id
        if db.client:
            try:
                db.client.table("telemetry_readings").delete().eq("station_id", diag_station).execute()
                db.client.table("predictions").delete().eq("station_id", diag_station).execute()
                db.client.table("alerts").delete().eq("station_id", diag_station).execute()
                print("      -> Successfully cleaned up all diagnostic test records.")
            except Exception as cleanup_err:
                print(f"[WARNING] Cleanup error: {cleanup_err}")

    print("\n[SUCCESS] Supabase verification complete! All cloud operations verified.")


if __name__ == "__main__":
    run_verification()
