import pandas as pd
import numpy as np
from typing import Tuple


def _compute_mage(df: pd.DataFrame) -> float:
    """Mean Amplitude of Glycemic Excursions (MAGE).
    Calculates the average of absolute differences between successive peaks and troughs
    that exceed one standard deviation of the glucose series.
    """
    if df.empty:
        return np.nan
    glucose = df['glucose'].astype(float)
    # Identify peaks and troughs
    diff = glucose.diff()
    peaks = (diff.shift(-1) < 0) & (diff > 0)
    troughs = (diff.shift(-1) > 0) & (diff < 0)
    extrema = glucose[peaks | troughs]
    if len(extrema) < 2:
        return np.nan
    # Compute successive absolute differences
    diffs = np.abs(np.diff(extrema.values))
    std = glucose.std()
    # Keep only excursions > 1 SD
    large_excursions = diffs[diffs > std]
    if len(large_excursions) == 0:
        return np.nan
    return large_excursions.mean()


def _compute_modd(df: pd.DataFrame) -> float:
    """Mean of Daily Differences (MODD).
    Average of absolute differences between glucose values at the same clock time on consecutive days.
    """
    if df.empty:
        return np.nan
    # Expect a datetime index named 'timestamp' and a 'glucose' column
    if 'timestamp' not in df.columns:
        return np.nan
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    # Resample to 5‑minute intervals (common CGM resolution) using forward fill
    df = df['glucose'].resample('5T').ffill()
    # Group by time of day
    groups = df.groupby(df.index.time)
    daily_diffs = []
    for _, vals in groups:
        vals = vals.dropna()
        if len(vals) < 2:
            continue
        diffs = np.abs(np.diff(vals.values))
        daily_diffs.append(diffs.mean())
    if not daily_diffs:
        return np.nan
    return np.mean(daily_diffs)


def _compute_conga(df: pd.DataFrame, delta_minutes: int = 30) -> float:
    """Continuous Overall Net Glycemic Action (CONGA).
    Standard deviation of the differences between glucose values separated by `delta_minutes`.
    """
    if df.empty:
        return np.nan
    if 'timestamp' not in df.columns:
        return np.nan
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df['glucose'].resample('5T').ffill()
    shifted = df.shift(delta_minutes // 5)  # assuming 5‑minute bins
    diffs = (df - shifted).dropna()
    return diffs.std()


def _compute_grade(df: pd.DataFrame) -> float:
    """Glycemic Risk Assessment of Diabetes Episodes (GRADE).
    Ratio of time spent in high‑risk zones (e.g., <70 or >180 mg/dL) to total monitoring time.
    """
    if df.empty:
        return np.nan
    total = len(df)
    high_risk = ((df['glucose'] < 70) | (df['glucose'] > 180)).sum()
    return high_risk / total if total > 0 else np.nan


def _compute_j_index(df: pd.DataFrame) -> float:
    """J‑index – combines variability and mean glucose.
    Formula: (Mean Glucose + SD) / 2.
    """
    if df.empty:
        return np.nan
    mean = df['glucose'].mean()
    sd = df['glucose'].std()
    return (mean + sd) / 2.0


def compute_variability_metrics(df: pd.DataFrame) -> dict:
    """Aggregate all variability metrics for a patient.

    Returns a dictionary mapping metric names to values.
    """
    return {
        'MAGE': _compute_mage(df),
        'MODD': _compute_modd(df),
        'CONGA': _compute_conga(df),
        'GRADE': _compute_grade(df),
        'J_index': _compute_j_index(df),
    }
