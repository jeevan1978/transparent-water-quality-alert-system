"""
train_models.py
===============
End-to-end model training script:
1. Loads or generates synthetic telemetry data
2. Trains the Current Water Quality Classifier (Good / Needs Treatment / Poor)
3. Trains the 6-Hour and 24-Hour Parameter Forecasters
4. Evaluates performance metrics and serializes models to models/ directory
"""

import os
import sys
import pandas as pd

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.current_classifier import WaterQualityClassifier
from src.data_simulator import WaterDataSimulator
from src.forecasting import WaterParameterForecaster
from src.labeling import label_dataframe
from src.preprocessing import FEATURE_COLUMNS


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_csv = os.path.join(base_dir, "data", "sample", "simulated_water_quality_sample.csv")

    if os.path.exists(sample_csv):
        print(f"[Training] Loading telemetry data from {sample_csv}...")
        df = pd.read_csv(sample_csv)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        print("[Training] Sample CSV not found. Generating 30 days of simulated training data...")
        sim = WaterDataSimulator(station_id="Station-Training")
        df = sim.generate_timeseries(days=30, interval_minutes=60, anomaly_probability=0.08)
        df = label_dataframe(df)
        os.makedirs(os.path.dirname(sample_csv), exist_ok=True)
        df.to_csv(sample_csv, index=False)
        print(f"  -> Generated {len(df)} records and saved to {sample_csv}")

    print("\n--- 1. Training Current Water Quality Classifier ---")
    classifier = WaterQualityClassifier()
    X = df[FEATURE_COLUMNS].values
    y = df["quality_label"].values

    report = classifier.train(X, y)
    saved_clf_path = classifier.save()
    print(f"[Success] Classifier trained and saved to: {saved_clf_path}")
    print(f"  -> Accuracy: {report.get('accuracy', 0.0):.4f}")
    for cat in ["Good", "Needs Treatment", "Poor"]:
        if cat in report:
            print(f"  -> Class '{cat}': Precision={report[cat]['precision']:.2f}, Recall={report[cat]['recall']:.2f}")

    print("\n--- 2. Training 6-Hour Numerical Parameter Forecaster ---")
    forecaster_6h = WaterParameterForecaster(horizon_hours=6)
    metrics_6h = forecaster_6h.train(df)
    saved_6h_path = forecaster_6h.save()
    print(f"[Success] 6h Forecaster trained and saved to: {saved_6h_path}")
    print(f"  -> MAE: {metrics_6h['mae']} | RMSE: {metrics_6h['rmse']}")

    print("\n--- 3. Training 24-Hour Numerical Parameter Forecaster ---")
    forecaster_24h = WaterParameterForecaster(horizon_hours=24)
    metrics_24h = forecaster_24h.train(df)
    saved_24h_path = forecaster_24h.save()
    print(f"[Success] 24h Forecaster trained and saved to: {saved_24h_path}")
    print(f"  -> MAE: {metrics_24h['mae']} | RMSE: {metrics_24h['rmse']}")

    print("\n[Complete] All models successfully trained on simulated data.")


if __name__ == "__main__":
    main()
