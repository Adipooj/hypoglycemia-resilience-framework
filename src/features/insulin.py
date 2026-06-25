import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from src.features.builder import register_feature

logger = logging.getLogger("hypo_resilience.features.insulin")


@register_feature(category="insulin", sources=["basal", "bolus"])
def extract_insulin_features(df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, Dict[str, str]]]:
    """
    Extracts metrics related to insulin delivery patterns.
    """
    features = {}
    metadata = {}

    # Check if insulin columns exist
    has_basal = "basal" in df.columns
    has_bolus = "bolus" in df.columns

    if not has_basal and not has_bolus:
        cols = ["mean_daily_basal", "mean_daily_bolus", "total_daily_insulin", "basal_bolus_ratio", "bolus_count_daily"]
        features = {c: np.nan for c in cols}
        metadata = {c: {"description": f"Insulin metric: {c}", "units": "units or ratio"} for c in cols}
        return features, metadata

    dates = df["timestamp"].dt.date

    # 1. Daily Basal Total
    if has_basal and not df["basal"].isna().all():
        daily_basal = df.groupby(dates)["basal"].sum()
        features["mean_daily_basal"] = float(daily_basal.mean())
    else:
        features["mean_daily_basal"] = 0.0
    metadata["mean_daily_basal"] = {"description": "Mean daily basal insulin dose", "units": "units"}

    # 2. Daily Bolus Total and Injection Counts
    if has_bolus and not df["bolus"].isna().all():
        # Doses
        daily_bolus = df.groupby(dates)["bolus"].sum()
        features["mean_daily_bolus"] = float(daily_bolus.mean())
        
        # Count non-zero bolus events per day
        daily_bolus_events = df[df["bolus"] > 0].groupby(dates)["bolus"].count()
        # Reindex to include all dates (fill missing days with 0)
        daily_bolus_events = daily_bolus_events.reindex(daily_bolus.index, fill_value=0)
        features["bolus_count_daily"] = float(daily_bolus_events.mean())

        # Bolus delivery hour variability
        bolus_indices = df[df["bolus"] > 0].index
        if len(bolus_indices) > 2:
            bolus_hours = df.loc[bolus_indices, "timestamp"].dt.hour
            features["bolus_timing_var"] = float(bolus_hours.std())
        else:
            features["bolus_timing_var"] = 0.0
    else:
        features["mean_daily_bolus"] = 0.0
        features["bolus_count_daily"] = 0.0
        features["bolus_timing_var"] = 0.0
        
    metadata["mean_daily_bolus"] = {"description": "Mean daily bolus insulin dose", "units": "units"}
    metadata["bolus_count_daily"] = {"description": "Mean daily count of bolus injections", "units": "Count"}
    metadata["bolus_timing_var"] = {"description": "Standard deviation of bolus delivery hours", "units": "hours"}

    # 3. Aggregations (Total Daily Insulin & Ratio)
    features["total_daily_insulin"] = features["mean_daily_basal"] + features["mean_daily_bolus"]
    metadata["total_daily_insulin"] = {"description": "Total daily insulin dose (Basal + Bolus)", "units": "units"}

    if features["mean_daily_bolus"] > 0:
        features["basal_bolus_ratio"] = features["mean_daily_basal"] / features["mean_daily_bolus"]
    else:
        features["basal_bolus_ratio"] = 0.0
    metadata["basal_bolus_ratio"] = {"description": "Basal to Bolus insulin dose ratio", "units": "Ratio"}

    return features, metadata
