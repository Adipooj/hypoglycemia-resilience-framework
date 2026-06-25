import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from src.features.builder import register_feature

logger = logging.getLogger("hypo_resilience.features.sleep")


@register_feature(category="sleep", sources=["sleep"])
def extract_sleep_features(df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, Dict[str, str]]]:
    """
    Extracts metrics related to sleep quality, heart rate variability, and stress.
    """
    features = {}
    metadata = {}

    has_sleep = "sleep" in df.columns
    has_hr = "heart_rate" in df.columns
    has_rhr = "resting_heart_rate" in df.columns
    has_stress = "stress" in df.columns

    if not has_sleep and not has_hr:
        cols = ["mean_sleep_duration", "sleep_fragmentation", "mean_heart_rate_sleep", "mean_resting_heart_rate", "mean_stress"]
        features = {c: np.nan for c in cols}
        metadata = {c: {"description": f"Sleep/Telemetry metric: {c}", "units": "hours, counts, bpm, or index"} for c in cols}
        return features, metadata

    dates = df["timestamp"].dt.date
    n_days = len(dates.unique())

    # 1. Sleep Duration
    if has_sleep and not df["sleep"].isna().all():
        # Count sleep bins (each is 5 minutes) per day
        daily_sleep_bins = df.groupby(dates)["sleep"].sum()
        features["mean_sleep_duration"] = float(daily_sleep_bins.mean() * 5 / 60) # Convert to hours
        
        # Sleep fragmentation: transitions from sleep (1) to wake (0)
        # Shift sleep status
        is_asleep = df["sleep"].fillna(0.0)
        transitions = ((is_asleep.shift(1) == 1) & (is_asleep == 0)).astype(int)
        daily_awakenings = transitions.groupby(dates).sum()
        features["sleep_fragmentation"] = float(daily_awakenings.mean())
    else:
        features["mean_sleep_duration"] = 0.0
        features["sleep_fragmentation"] = 0.0
        
    metadata["mean_sleep_duration"] = {"description": "Mean daily sleep duration", "units": "hours"}
    metadata["sleep_fragmentation"] = {"description": "Mean number of nightly awakenings", "units": "Count"}

    # 2. Heart Rate during sleep
    if has_hr and has_sleep and not df["heart_rate"].isna().all():
        sleep_hr = df[df["sleep"] == 1]["heart_rate"].dropna()
        features["mean_heart_rate_sleep"] = float(sleep_hr.mean()) if not sleep_hr.empty else np.nan
        
        awake_hr = df[df["sleep"] == 0]["heart_rate"].dropna()
        features["mean_heart_rate_awake"] = float(awake_hr.mean()) if not awake_hr.empty else np.nan
    else:
        features["mean_heart_rate_sleep"] = np.nan
        features["mean_heart_rate_awake"] = np.nan
        
    metadata["mean_heart_rate_sleep"] = {"description": "Mean heart rate during sleep", "units": "bpm"}
    metadata["mean_heart_rate_awake"] = {"description": "Mean heart rate while awake", "units": "bpm"}

    # 3. Resting Heart Rate
    if has_rhr and not df["resting_heart_rate"].isna().all():
        valid_rhr = df[df["resting_heart_rate"] > 0]["resting_heart_rate"].dropna()
        features["mean_resting_heart_rate"] = float(valid_rhr.mean()) if not valid_rhr.empty else np.nan
    else:
        features["mean_resting_heart_rate"] = np.nan
    metadata["mean_resting_heart_rate"] = {"description": "Mean resting heart rate", "units": "bpm"}

    # 4. Stress level
    if has_stress and not df["stress"].isna().all():
        valid_stress = df["stress"].dropna()
        features["mean_stress"] = float(valid_stress.mean())
        features["stress_variance"] = float(valid_stress.var()) if len(valid_stress) > 1 else 0.0
    else:
        features["mean_stress"] = np.nan
        features["stress_variance"] = np.nan
        
    metadata["mean_stress"] = {"description": "Mean stress level value", "units": "Index"}
    metadata["stress_variance"] = {"description": "Variance of stress level values", "units": "Index^2"}

    return features, metadata
