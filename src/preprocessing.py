"""
preprocessing.py
================
Cleans, scales, and engineers features for water quality classification
and forecasting models.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "ph",
    "turbidity",
    "dissolved_oxygen",
    "temperature",
    "conductivity",
    "total_dissolved_solids",
]


def clean_telemetry_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw telemetry dataframe:
    - Sorts by timestamp
    - Removes duplicate timestamps
    - Fills or interpolates minor gaps
    - Clips extreme non-physical values

    Args:
        df (pd.DataFrame): Raw telemetry DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    if "timestamp" in df_clean.columns:
        df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])
        df_clean = df_clean.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

    # Interpolate missing numeric values using linear time interpolation
    numeric_cols = [c for c in FEATURE_COLUMNS if c in df_clean.columns]
    df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method="linear").bfill().ffill()

    # Physical clamping
    if "ph" in df_clean.columns:
        df_clean["ph"] = df_clean["ph"].clip(0.0, 14.0)
    if "dissolved_oxygen" in df_clean.columns:
        df_clean["dissolved_oxygen"] = df_clean["dissolved_oxygen"].clip(lower=0.0)
    if "turbidity" in df_clean.columns:
        df_clean["turbidity"] = df_clean["turbidity"].clip(lower=0.0)

    return df_clean


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts cyclical temporal features (sine/cosine of hour) to help models
    learn diurnal water parameter fluctuations.

    Args:
        df (pd.DataFrame): Telemetry DataFrame with 'timestamp' column.

    Returns:
        pd.DataFrame: DataFrame augmented with temporal features.
    """
    df_feat = df.copy()
    if "timestamp" not in df_feat.columns:
        return df_feat

    hours = df_feat["timestamp"].dt.hour + df_feat["timestamp"].dt.minute / 60.0
    df_feat["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df_feat["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    return df_feat


def create_lag_features(
    df: pd.DataFrame,
    target_cols: List[str] = FEATURE_COLUMNS,
    lags: List[int] = [1, 3, 6]
) -> pd.DataFrame:
    """
    Creates historical lag features for time-series forecasting.

    Args:
        df (pd.DataFrame): Ordered time-series DataFrame.
        target_cols (List[str]): Columns to create lags for.
        lags (List[int]): Lag steps (e.g. 1 hour, 3 hours ago).

    Returns:
        pd.DataFrame: DataFrame containing original and lag features.
    """
    df_lags = df.copy()
    for col in target_cols:
        if col in df_lags.columns:
            for lag in lags:
                df_lags[f"{col}_lag{lag}"] = df_lags[col].shift(lag)

    return df_lags.dropna().reset_index(drop=True)


def prepare_features_and_scaler(
    df: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLUMNS
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Extracts features and fits a standard scaler for downstream models.

    Args:
        df (pd.DataFrame): DataFrame containing feature columns.
        feature_cols (List[str]): Feature column names.

    Returns:
        Tuple[np.ndarray, StandardScaler]: Scaled feature array and fitted scaler.
    """
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# TODO: Team Member 2 - Implement rolling window statistical aggregations (mean, std dev over 6h and 24h)
# TODO: Team Member 2 - Add automated outlier detection using IsolationForest or Z-score thresholds
