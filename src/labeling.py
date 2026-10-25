"""
labeling.py
===========
Determines water quality categories ('Good', 'Needs Treatment', 'Poor')
from water quality parameters based on configurable reference thresholds.

CRITICAL DISCLAIMER:
--------------------
THIS MODULE LABELS SIMULATED TELEMETRY FOR MACHINE LEARNING EXPERIMENTS.
IT DOES NOT CONSTITUTE A FORMAL SANITARY WATER SAFETY INSPECTION.
"""

from typing import Dict, Tuple
import pandas as pd


def determine_quality_label(row: Dict[str, float]) -> Tuple[str, str]:
    """
    Evaluates a single reading and assigns a quality label along with the primary rationale.

    Categories:
    - 'Poor': Severe deviation posing substantial ecosystem/contamination concern.
    - 'Needs Treatment': Parameter moderately elevated/depleted; requires standard remediation.
    - 'Good': All monitored parameters within typical baseline acceptable limits.

    Args:
        row (Dict[str, float]): Dictionary or Series with keys:
            ph, turbidity, dissolved_oxygen, conductivity, total_dissolved_solids.

    Returns:
        Tuple[str, str]: (Category label, explanation reason)
    """
    ph = row.get("ph", 7.0)
    turbidity = row.get("turbidity", 1.0)
    do = row.get("dissolved_oxygen", 7.0)
    cond = row.get("conductivity", 300.0)
    tds = row.get("total_dissolved_solids", 200.0)

    # 1. Check for 'Poor' condition (Severe breaches)
    reasons_poor = []
    if do < 4.0:
        reasons_poor.append(f"Severe hypoxia (DO {do} < 4.0 mg/L)")
    if ph < 6.0 or ph > 9.0:
        reasons_poor.append(f"Extreme pH level ({ph})")
    if turbidity >= 15.0:
        reasons_poor.append(f"Excessive turbidity ({turbidity} >= 15.0 NTU)")
    if cond >= 1200.0 or tds >= 1000.0:
        reasons_poor.append(f"Extremely high dissolved solids/conductivity ({cond} µS/cm)")

    if len(reasons_poor) > 0:
        return "Poor", "; ".join(reasons_poor)

    # 2. Check for 'Needs Treatment' condition (Moderate breaches)
    reasons_treatment = []
    if do < 6.0:
        reasons_treatment.append(f"Sub-optimal dissolved oxygen ({do} < 6.0 mg/L)")
    if ph < 6.5 or ph > 8.5:
        reasons_treatment.append(f"Non-neutral pH ({ph} outside 6.5-8.5)")
    if turbidity > 5.0:
        reasons_treatment.append(f"Elevated turbidity ({turbidity} > 5.0 NTU)")
    if cond > 800.0 or tds > 500.0:
        reasons_treatment.append(f"High dissolved solids/conductivity ({cond} µS/cm)")

    if len(reasons_treatment) > 0:
        return "Needs Treatment", "; ".join(reasons_treatment)

    # 3. Baseline 'Good'
    return "Good", "All monitored parameters are within safe reference boundaries."


def label_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies quality labeling across an entire telemetry DataFrame.

    Args:
        df (pd.DataFrame): Input telemetry DataFrame.

    Returns:
        pd.DataFrame: DataFrame augmented with 'quality_label' and 'label_reason'.
    """
    df_labeled = df.copy()
    labels = []
    reasons = []

    for _, row in df_labeled.iterrows():
        cat, reason = determine_quality_label(row.to_dict())
        labels.append(cat)
        reasons.append(reason)

    df_labeled["quality_label"] = labels
    df_labeled["label_reason"] = reasons
    return df_labeled


# TODO: Team Member 2 - Load dynamic thresholds directly from config/thresholds.yaml
# TODO: Team Member 2 - Add customizable weighted Water Quality Index (WQI) score calculation (0 - 100)
if __name__ == "__main__":
    test_reading = {"ph": 5.8, "turbidity": 8.0, "dissolved_oxygen": 5.0, "conductivity": 400.0}
    cat, reason = determine_quality_label(test_reading)
    print(f"Sample Reading Label: {cat} | Reason: {reason}")
