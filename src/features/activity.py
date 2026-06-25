import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from src.features.builder import register_feature

logger = logging.getLogger("hypo_resilience.features.activity")


@register_feature(category="activity", sources=["activity"])
def extract_activity_features(df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, Dict[str, str]]]:
    """
    Extracts metrics related to physical activity and exercise frequency.
    """
    features = {}
    metadata = {}

    has_steps = "steps" in df.columns
    has_met = "MET" in df.columns
    has_type = "activity_type" in df.columns

    if not has_steps and not has_met:
        cols = ["mean_daily_steps", "mean_daily_distance", "mean_met", "max_met", "active_minutes_daily", "sedentary_proportion"]
        features = {c: np.nan for c in cols}
        metadata = {c: {"description": f"Activity metric: {c}", "units": "steps or MET"} for c in cols}
        return features, metadata

    dates = df["timestamp"].dt.date
    n_days = len(dates.unique()) if len(dates) > 0 else 1

    # 1. Step Count & Distance
    if has_steps and not df["steps"].isna().all():
        daily_steps = df.groupby(dates)["steps"].sum()
        features["mean_daily_steps"] = float(daily_steps.mean())
        
        daily_dist = df.groupby(dates)["distance"].sum()
        features["mean_daily_distance"] = float(daily_dist.mean())
    else:
        features["mean_daily_steps"] = 0.0
        features["mean_daily_distance"] = 0.0
        
    metadata["mean_daily_steps"] = {"description": "Mean daily step count", "units": "steps"}
    metadata["mean_daily_distance"] = {"description": "Mean daily walking/running distance", "units": "meters"}

    # 2. MET Stats
    if has_met and not df["MET"].isna().all():
        features["mean_met"] = float(df["MET"].mean())
        features["max_met"] = float(df["MET"].max())
        
        # Active minutes per day: MET > 1.5 (each 5-min bin is 5 minutes)
        is_active = (df["MET"] > 1.5).astype(int)
        daily_active_bins = is_active.groupby(dates).sum()
        features["active_minutes_daily"] = float(daily_active_bins.mean() * 5)
        
        # Sedentary proportion: MET <= 1.5
        features["sedentary_proportion"] = float((df["MET"] <= 1.5).sum() / len(df) * 100)
    else:
        features["mean_met"] = 1.0
        features["max_met"] = 1.0
        features["active_minutes_daily"] = 0.0
        features["sedentary_proportion"] = 100.0

    metadata["mean_met"] = {"description": "Mean metabolic equivalent rate", "units": "METs"}
    metadata["max_met"] = {"description": "Maximum metabolic equivalent rate", "units": "METs"}
    metadata["active_minutes_daily"] = {"description": "Mean daily active minutes (MET > 1.5)", "units": "minutes"}
    metadata["sedentary_proportion"] = {"description": "Percentage of epoch spent in sedentary state (MET <= 1.5)", "units": "%"}

    # 3. Exercise frequency (MET >= 3.0 for >= 15 min)
    if has_met and not df["MET"].isna().all():
        is_exertion = (df["MET"] >= 3.0).astype(int)
        # Find runs of >= 3 consecutive 5-min intervals (>=15 min)
        runs = is_exertion.groupby((is_exertion != is_exertion.shift()).cumsum()).transform("sum")
        ex_sessions = (is_exertion == 1) & (runs >= 3)
        
        # Starts of exercise sessions
        session_starts = (ex_sessions.astype(int).diff() == 1)
        if ex_sessions.iloc[0]:
            session_starts.iloc[0] = True
            
        total_sessions = session_starts.sum()
        # Convert to weekly frequency
        features["exercise_frequency_weekly"] = float(total_sessions / (n_days / 7.0)) if n_days > 0 else 0.0
    else:
        features["exercise_frequency_weekly"] = 0.0

    metadata["exercise_frequency_weekly"] = {"description": "Mean weekly exercise sessions (>=15 min at MET >= 3.0)", "units": "sessions/week"}

    return features, metadata
