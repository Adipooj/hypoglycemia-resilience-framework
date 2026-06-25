import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from src.features.builder import register_feature
from src.constants import (
    GLUCOSE_HYPO_LEVEL1,
    GLUCOSE_HYPO_LEVEL2,
    GLUCOSE_TARGET_LOW,
    GLUCOSE_TARGET_HIGH,
    GLUCOSE_HYPER_LEVEL2,
    MMOL_TO_MGDL
)

logger = logging.getLogger("hypo_resilience.features.glucose")


def _bg_risk_function(glucose_mmol: float | pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Computes low and high blood glucose risk indices based on Kovatchev's formula.
    First converts glucose from mmol/L to mg/dL.
    """
    # Convert to mg/dL
    bg_mg = glucose_mmol * MMOL_TO_MGDL
    
    # Clip to avoid math errors (e.g. log of <= 0)
    bg_mg = np.clip(bg_mg, 1.0, 1000.0)
    
    # Kovatchev transformation
    f_bg = 1.509 * (np.power(np.log(bg_mg), 1.084) - 5.381)
    risk = 10 * np.power(f_bg, 2)
    
    low_risk = np.where(f_bg < 0, risk, 0.0)
    high_risk = np.where(f_bg > 0, risk, 0.0)
    
    return pd.Series(low_risk), pd.Series(high_risk)


@register_feature(category="glucose", sources=["glucose"])
def extract_glucose_features(df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, Dict[str, str]]]:
    """
    Extracts summary statistics and clinical indices from glucose timeseries.
    """
    features = {}
    metadata = {}

    glucose = df["glucose"].dropna()

    if glucose.empty:
        # Return default NaNs if glucose data is entirely missing
        cols = [
            "mean_glucose", "median_glucose", "min_glucose", "max_glucose",
            "time_in_range", "time_below_range_l1", "time_below_range_l2",
            "time_above_range_l1", "time_above_range_l2", "lbgi", "hbgi", "adrr",
            "mage", "hypo_episode_count", "hypo_duration_mean", "hypo_recovery_speed"
        ]
        features = {c: np.nan for c in cols}
        metadata = {c: {"description": f"Glycemic metric: {c}", "units": "mmol/L or %"} for c in cols}
        return features, metadata

    # 1. Basic Stats
    features["mean_glucose"] = float(glucose.mean())
    metadata["mean_glucose"] = {"description": "Mean glucose value", "units": "mmol/L"}
    
    features["median_glucose"] = float(glucose.median())
    metadata["median_glucose"] = {"description": "Median glucose value", "units": "mmol/L"}
    
    features["min_glucose"] = float(glucose.min())
    metadata["min_glucose"] = {"description": "Minimum glucose value", "units": "mmol/L"}
    
    features["max_glucose"] = float(glucose.max())
    metadata["max_glucose"] = {"description": "Maximum glucose value", "units": "mmol/L"}

    # 2. Time in Ranges (TIR, TBR, TAR)
    n_readings = len(glucose)
    
    features["time_in_range"] = float((glucose.between(GLUCOSE_TARGET_LOW, GLUCOSE_TARGET_HIGH)).sum() / n_readings * 100)
    metadata["time_in_range"] = {"description": "Time in Target Range (3.9 - 10.0 mmol/L)", "units": "%"}
    
    features["time_below_range_l1"] = float((glucose.between(GLUCOSE_HYPO_LEVEL2, GLUCOSE_HYPO_LEVEL1, inclusive="left")).sum() / n_readings * 100)
    metadata["time_below_range_l1"] = {"description": "Time Below Range Level 1 (3.0 - 3.9 mmol/L)", "units": "%"}
    
    features["time_below_range_l2"] = float((glucose < GLUCOSE_HYPO_LEVEL2).sum() / n_readings * 100)
    metadata["time_below_range_l2"] = {"description": "Time Below Range Level 2 (< 3.0 mmol/L)", "units": "%"}
    
    features["time_above_range_l1"] = float((glucose.between(GLUCOSE_TARGET_HIGH, GLUCOSE_HYPER_LEVEL2, inclusive="right")).sum() / n_readings * 100)
    metadata["time_above_range_l1"] = {"description": "Time Above Range Level 1 (10.0 - 13.9 mmol/L)", "units": "%"}
    
    features["time_above_range_l2"] = float((glucose > GLUCOSE_HYPER_LEVEL2).sum() / n_readings * 100)
    metadata["time_above_range_l2"] = {"description": "Time Above Range Level 2 (> 13.9 mmol/L)", "units": "%"}

    # 3. Low/High Blood Glucose Risk Indices (LBGI, HBGI)
    low_risk, high_risk = _bg_risk_function(glucose)
    features["lbgi"] = float(low_risk.mean())
    metadata["lbgi"] = {"description": "Low Blood Glucose Index", "units": "Index"}
    
    features["hbgi"] = float(high_risk.mean())
    metadata["hbgi"] = {"description": "High Blood Glucose Index", "units": "Index"}

    # 4. Average Daily Risk Range (ADRR)
    # Group risk by date
    dates = df.loc[glucose.index, "timestamp"].dt.date
    daily_df = pd.DataFrame({"low": low_risk, "high": high_risk, "date": dates.values})
    daily_max = daily_df.groupby("date").agg({"low": "max", "high": "max"})
    features["adrr"] = float((daily_max["low"] + daily_max["high"]).mean())
    metadata["adrr"] = {"description": "Average Daily Risk Range", "units": "Index"}

    # 5. Mean Amplitude of Glycemic Excursions (MAGE)
    # MAGE is the mean of glucose fluctuations that exceed 1 SD of the glucose values
    sd = glucose.std()
    diffs = glucose.diff().dropna()
    # Approximate MAGE by filtering excursions (direction changes exceeding 1 SD)
    excursions = []
    current_excursion = 0.0
    for diff in diffs:
        current_excursion += diff
        if abs(current_excursion) > sd:
            excursions.append(abs(current_excursion))
            current_excursion = 0.0
    features["mage"] = float(np.mean(excursions)) if excursions else 0.0
    metadata["mage"] = {"description": "Mean Amplitude of Glycemic Excursions", "units": "mmol/L"}

    # 6. Hypoglycemic Episodes and Recovery
    # Hypo episode is defined as glucose < 3.9 mmol/L for >= 15 consecutive minutes (3 readings)
    is_hypo = (df["glucose"] < GLUCOSE_HYPO_LEVEL1).astype(int)
    # Detect runs of 1s
    runs = is_hypo.groupby((is_hypo != is_hypo.shift()).cumsum()).transform("sum")
    # Identify intervals where it is a hypo and run length >= 3
    hypo_episodes = (is_hypo == 1) & (runs >= 3)
    
    # Count distinct episodes (diff is positive)
    episode_starts = (hypo_episodes.astype(int).diff() == 1)
    if hypo_episodes.iloc[0]:
        episode_starts.iloc[0] = True
    
    features["hypo_episode_count"] = int(episode_starts.sum())
    metadata["hypo_episode_count"] = {"description": "Number of prolonged hypoglycemic episodes (>=15 mins)", "units": "Count"}

    # Calculate average episode duration and recovery speed
    durations = []
    recovery_speeds = []
    
    starts_idx = np.where(episode_starts)[0]
    for start_idx in starts_idx:
        # Find where episode ends
        end_idx = start_idx
        while end_idx < len(df) and df.iloc[end_idx]["glucose"] < GLUCOSE_HYPO_LEVEL1:
            end_idx += 1
        
        duration_mins = (end_idx - start_idx) * 5
        durations.append(duration_mins)

        # Recovery speed: rate of change from the lowest point of hypo to target range
        hypo_segment = df.iloc[start_idx:end_idx]["glucose"]
        if not hypo_segment.empty:
            min_val = hypo_segment.min()
            min_ts = df.iloc[hypo_segment.idxmin()]["timestamp"]
            
            # Find where it recovers to >= 3.9
            recover_idx = end_idx
            while recover_idx < len(df) and df.iloc[recover_idx]["glucose"] < GLUCOSE_HYPO_LEVEL1:
                recover_idx += 1
                
            if recover_idx < len(df):
                recover_val = df.iloc[recover_idx]["glucose"]
                recover_ts = df.iloc[recover_idx]["timestamp"]
                time_diff_hours = (recover_ts - min_ts).total_seconds() / 3600.0
                if time_diff_hours > 0:
                    speed = (recover_val - min_val) / time_diff_hours
                    recovery_speeds.append(speed)

    features["hypo_duration_mean"] = float(np.mean(durations)) if durations else 0.0
    metadata["hypo_duration_mean"] = {"description": "Mean duration of hypoglycemic episodes", "units": "minutes"}
    
    features["hypo_recovery_speed"] = float(np.mean(recovery_speeds)) if recovery_speeds else 0.0
    metadata["hypo_recovery_speed"] = {"description": "Mean rate of glucose recovery from hypoglycemia lowest point", "units": "mmol/L/hour"}

    return features, metadata
