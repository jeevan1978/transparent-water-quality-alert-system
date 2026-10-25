"""
data_simulator.py
=================
Generates synthetic water quality telemetry data for testing, training,
and dashboard visualization.

CRITICAL DISCLAIMER:
--------------------
ALL DATA PRODUCED BY THIS MODULE IS SYNTHETIC / SIMULATED.
IT DOES NOT REPRESENT REAL PHYSICAL MEASUREMENTS AND CANNOT BE USED
TO CERTIFY DRINKING WATER SAFETY OR COMPLIANCE WITH STATUTORY HEALTH CODES.
"""

from datetime import datetime, timedelta
import math
import random
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class WaterDataSimulator:
    """
    Simulates time-series telemetry data for water quality monitoring stations.

    Attributes:
        station_id (str): Unique station name or identifier (e.g. "Station-Alpha").
        seed (int, optional): Random seed for reproducibility.
    """

    def __init__(self, station_id: str = "Station-Alpha", seed: Optional[int] = 42):
        self.station_id = station_id
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def generate_single_reading(
        self,
        timestamp: datetime,
        inject_anomaly: bool = False,
        anomaly_type: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Generate a single simulated sensor reading for a given timestamp.

        Args:
            timestamp (datetime): Timestamp of the simulated measurement.
            inject_anomaly (bool): Whether to inject an anomalous event.
            anomaly_type (str, optional): 'runoff', 'acid_spike', 'hypoxia', or None.

        Returns:
            Dict[str, any]: Dictionary of simulated sensor parameters.
        """
        # Diurnal (day-night) cycle modeling based on hour of day
        hour = timestamp.hour + timestamp.minute / 60.0

        # Baseline parameters with diurnal fluctuation
        # Temperature peaks around 15:00 (3 PM)
        temp_diurnal = 3.0 * math.sin(2 * math.pi * (hour - 9) / 24.0)
        temperature = 22.0 + temp_diurnal + np.random.normal(0, 0.5)

        # Dissolved oxygen inversely correlates with temperature and daylight photosynthesis
        # Higher during peak daylight, lower just before dawn
        do_diurnal = 1.5 * math.sin(2 * math.pi * (hour - 8) / 24.0)
        dissolved_oxygen = 7.5 + do_diurnal + np.random.normal(0, 0.3)

        # pH typically has slight diurnal fluctuation (algal carbon dioxide consumption)
        ph_diurnal = 0.2 * math.sin(2 * math.pi * (hour - 10) / 24.0)
        ph = 7.3 + ph_diurnal + np.random.normal(0, 0.1)

        # Turbidity baseline (NTU)
        turbidity = max(0.5, np.random.gamma(shape=2.0, scale=1.2))

        # Conductivity baseline (µS/cm)
        conductivity = max(100.0, 350.0 + np.random.normal(0, 25.0))

        # Total Dissolved Solids roughly proportional to conductivity
        tds = conductivity * 0.65 + np.random.normal(0, 10.0)

        # Anomaly Injection (Simulating sudden pollution, storm runoff, or sensor failure)
        if inject_anomaly:
            selected_type = anomaly_type or random.choice(["runoff", "acid_spike", "hypoxia"])
            if selected_type == "runoff":
                # Storm runoff: Turbidity jumps significantly, conductivity increases
                turbidity += np.random.uniform(12.0, 25.0)
                conductivity += np.random.uniform(200.0, 450.0)
                tds += np.random.uniform(150.0, 300.0)
            elif selected_type == "acid_spike":
                # Chemical runoff: pH plummets or spikes
                ph -= np.random.uniform(1.8, 2.5)
            elif selected_type == "hypoxia":
                # Organic loading / algal die-off: Dissolved oxygen severely depleted
                dissolved_oxygen = max(0.8, dissolved_oxygen - np.random.uniform(4.0, 5.5))

        # Clamp physical lower limits
        ph = float(np.clip(ph, 0.0, 14.0))
        turbidity = float(np.clip(turbidity, 0.0, 100.0))
        dissolved_oxygen = float(max(0.0, dissolved_oxygen))
        temperature = float(max(0.0, temperature))
        conductivity = float(max(0.0, conductivity))
        tds = float(max(0.0, tds))

        return {
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "station_id": self.station_id,
            "ph": round(ph, 2),
            "turbidity": round(turbidity, 2),
            "dissolved_oxygen": round(dissolved_oxygen, 2),
            "temperature": round(temperature, 2),
            "conductivity": round(conductivity, 2),
            "total_dissolved_solids": round(tds, 2),
            "is_synthetic": True
        }

    def generate_timeseries(
        self,
        days: int = 14,
        interval_minutes: int = 60,
        anomaly_probability: float = 0.04
    ) -> pd.DataFrame:
        """
        Generate a full historical time-series of simulated readings.

        Args:
            days (int): Number of historical days to simulate.
            interval_minutes (int): Frequency between readings in minutes.
            anomaly_probability (float): Likelihood of an injected anomaly reading.

        Returns:
            pd.DataFrame: Formatted DataFrame of simulated telemetry.
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        records: List[Dict[str, any]] = []
        current = start_time
        step = timedelta(minutes=interval_minutes)

        while current <= end_time:
            inject = (random.random() < anomaly_probability)
            reading = self.generate_single_reading(current, inject_anomaly=inject)
            records.append(reading)
            current += step

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df


# TODO: Team Member 1 - Add weather-correlated simulation (e.g. rainfall volume impacting turbidity)
# TODO: Team Member 1 - Add sensor failure modes (e.g. flatlining constant values or missing telemetry gaps)
if __name__ == "__main__":
    simulator = WaterDataSimulator(station_id="Station-Alpha")
    sample_df = simulator.generate_timeseries(days=1, interval_minutes=60)
    print("Generated simulated sample dataset:")
    print(sample_df.head())
