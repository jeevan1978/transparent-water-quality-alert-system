"""
forecasting.py
==============
Multi-step time-series forecasting for water quality parameters at
6-hour and 24-hour horizons, plus future category projection.

CRITICAL DISCLAIMER:
--------------------
ALL FORECASTS ARE GENERATED ON SIMULATED TELEMETRY FOR EXPERIMENTAL USE.
THEY CANNOT BE RELIED UPON FOR OPERATIONAL PUBLIC HEALTH MANDATES.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.labeling import determine_quality_label
from src.preprocessing import FEATURE_COLUMNS, clean_telemetry_data, engineer_time_features


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class WaterParameterForecaster:
    """
    Predicts future numerical values of water quality parameters at specified
    horizons (6 hours and 24 hours), and determines projected future categories.
    """

    def __init__(self, horizon_hours: int = 6):
        self.horizon_hours = horizon_hours
        self.feature_columns = FEATURE_COLUMNS
        self.target_columns = FEATURE_COLUMNS
        self.model: Optional[RandomForestRegressor] = None
        self.model_filename = f"forecaster_{horizon_hours}h.joblib"
        self.model_path = os.path.join(MODELS_DIR, self.model_filename)

        if os.path.exists(self.model_path):
            self.load()

    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds feature matrix X(t) and target matrix Y(t + horizon).
        """
        df_clean = clean_telemetry_data(df)
        df_clean = engineer_time_features(df_clean)

        input_cols = self.feature_columns + ["hour_sin", "hour_cos"]

        # Shift targets by horizon_hours
        X_rows = []
        Y_rows = []

        # Assuming hourly telemetry data
        shift_steps = self.horizon_hours
        for i in range(len(df_clean) - shift_steps):
            x_vals = df_clean.iloc[i][input_cols].values.astype(float)
            y_vals = df_clean.iloc[i + shift_steps][self.target_columns].values.astype(float)
            X_rows.append(x_vals)
            Y_rows.append(y_vals)

        return np.array(X_rows), np.array(Y_rows)

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Fits multi-output regression model to predict parameters at (t + horizon).

        Args:
            df (pd.DataFrame): Time-series telemetry dataset.

        Returns:
            Dict[str, float]: Training performance metrics (MAE, RMSE).
        """
        X, Y = self._prepare_training_data(df)
        if len(X) < 10:
            raise ValueError("Insufficient data points to train forecaster.")

        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X, Y)

        Y_pred = self.model.predict(X)
        mae = float(mean_absolute_error(Y, Y_pred))
        rmse = float(np.sqrt(mean_squared_error(Y, Y_pred)))

        return {"horizon_hours": self.horizon_hours, "mae": round(mae, 4), "rmse": round(rmse, 4)}

    def forecast_parameters(self, recent_df: pd.DataFrame) -> Dict[str, float]:
        """
        Generates numerical forecast of parameters at (t + horizon_hours).

        Args:
            recent_df (pd.DataFrame): Telemetry up to current time t.

        Returns:
            Dict[str, float]: Forecasted parameter dictionary.
        """
        df_clean = clean_telemetry_data(recent_df)
        df_clean = engineer_time_features(df_clean)
        latest_row = df_clean.iloc[-1]

        # Use trained model if available
        if self.model is not None:
            input_cols = self.feature_columns + ["hour_sin", "hour_cos"]
            x_input = latest_row[input_cols].values.reshape(1, -1).astype(float)
            pred_vals = self.model.predict(x_input)[0]
            forecast = {col: round(float(pred_vals[i]), 2) for i, col in enumerate(self.target_columns)}
            # Physical clamping
            forecast["ph"] = round(float(np.clip(forecast.get("ph", 7.0), 0.0, 14.0)), 2)
            forecast["turbidity"] = round(float(max(0.0, forecast.get("turbidity", 1.0))), 2)
            forecast["dissolved_oxygen"] = round(float(max(0.0, forecast.get("dissolved_oxygen", 7.0))), 2)
            return forecast

        # Fallback: Naive smoothing with slight trend persistence
        fallback_forecast = {}
        for col in self.target_columns:
            val = float(latest_row.get(col, 7.0))
            # Minor heuristic diurnal drift
            drift = np.random.normal(0, 0.05 * val) if val > 0 else 0
            fallback_forecast[col] = round(max(0.0, val + drift), 2)

        if "ph" in fallback_forecast:
            fallback_forecast["ph"] = round(float(np.clip(fallback_forecast["ph"], 0.0, 14.0)), 2)
        return fallback_forecast

    def predict_future_category(
        self,
        recent_df: pd.DataFrame,
        classifier=None
    ) -> Tuple[str, str, Dict[str, float]]:
        """
        Forecasts parameters at horizon and predicts future quality category.

        Args:
            recent_df: Historical telemetry up to time t.
            classifier: Optional trained classifier instance.

        Returns:
            Tuple[str, str, Dict[str, float]]:
                (Predicted Category, Reason string, Forecasted parameters)
        """
        forecasted_params = self.forecast_parameters(recent_df)

        # If a trained classifier is provided, use it
        if classifier is not None and getattr(classifier, "model", None) is not None:
            features = np.array([[forecasted_params[col] for col in FEATURE_COLUMNS]])
            category = classifier.predict(features)[0]
            _, reason = determine_quality_label(forecasted_params)
            return category, f"Projected {self.horizon_hours}h ahead: {reason}", forecasted_params

        # Fallback to deterministic rules
        category, reason = determine_quality_label(forecasted_params)
        return category, f"Projected {self.horizon_hours}h ahead: {reason}", forecasted_params

    def save(self, filepath: Optional[str] = None) -> str:
        """Saves forecaster to disk."""
        save_path = filepath or self.model_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(self.model, save_path)
        return save_path

    def load(self, filepath: Optional[str] = None) -> None:
        """Loads forecaster from disk."""
        load_path = filepath or self.model_path
        if os.path.exists(load_path):
            self.model = joblib.load(load_path)


# TODO: Team Member 3 - Implement ARIMA / Prophet / LSTM comparative forecasting baselines
# TODO: Team Member 3 - Implement prediction intervals (confidence bounds 95%) for forecasted parameters
