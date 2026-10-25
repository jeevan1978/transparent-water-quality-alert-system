"""
generate_data.py
================
CLI utility to generate synthetic water quality telemetry datasets for
training, testing, and populating the project's databases.

CRITICAL DISCLAIMER:
--------------------
ALL GENERATED DATA IS FULLY SYNTHETIC / SIMULATED.
IT DOES NOT REPRESENT REAL WATER QUALITY MEASUREMENTS.
"""

import argparse
import os
import sys

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_simulator import WaterDataSimulator
from src.labeling import label_dataframe
from src.database import WaterDatabase


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic water quality telemetry.")
    parser.add_argument("--days", type=int, default=30, help="Number of historical days to simulate (default: 30)")
    parser.add_argument("--interval", type=int, default=60, help="Interval between readings in minutes (default: 60)")
    parser.add_argument("--stations", type=str, default="Station-Alpha,Station-Beta,Station-Gamma",
                        help="Comma-separated list of station IDs")
    parser.add_argument("--output", type=str, default="data/sample/simulated_water_quality_sample.csv",
                        help="Output CSV filepath")
    parser.add_argument("--save-to-db", action="store_true", help="Also insert generated records into database")
    args = parser.parse_args()

    station_list = [s.strip() for s in args.stations.split(",") if s.strip()]
    all_dfs = []

    print(f"[Simulator] Generating {args.days} days of simulated telemetry for {len(station_list)} stations...")

    for station in station_list:
        sim = WaterDataSimulator(station_id=station)
        df_station = sim.generate_timeseries(days=args.days, interval_minutes=args.interval)
        df_station = label_dataframe(df_station)
        all_dfs.append(df_station)
        print(f"  -> Generated {len(df_station)} records for {station}")

    import pandas as pd
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Save to CSV
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    print(f"[Success] Saved {len(combined_df)} simulated records to {output_path}")

    # Optionally populate database
    if args.save_to_db:
        print("[Database] Inserting records into database...")
        db = WaterDatabase()
        count = 0
        for _, row in combined_df.iterrows():
            db.insert_reading(row.to_dict())
            count += 1
            if count % 200 == 0:
                print(f"  -> Inserted {count}/{len(combined_df)} records...")
        print(f"[Success] Completed database ingestion for {count} simulated records.")


if __name__ == "__main__":
    main()
