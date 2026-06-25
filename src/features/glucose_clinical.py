import pandas as pd
import numpy as np
from src.core.patient import Patient


def compute_time_in_range(df: pd.DataFrame, low: float = 70, high: float = 180) -> float:
    """Return percentage of glucose readings within [low, high] mg/dL."""
    if df.empty:
        return np.nan
    count = ((df['glucose'] >= low) & (df['glucose'] <= high)).sum()
    return 100.0 * count / len(df)


def compute_time_below_range(df: pd.DataFrame, threshold: float = 70) -> float:
    if df.empty:
        return np.nan
    count = (df['glucose'] < threshold).sum()
    return 100.0 * count / len(df)


def compute_time_above_range(df: pd.DataFrame, threshold: float = 180) -> float:
    if df.empty:
        return np.nan
    count = (df['glucose'] > threshold).sum()
    return 100.0 * count / len(df)


def compute_lbgI(df: pd.DataFrame) -> float:
    """Low Blood Glucose Index (LBGI) as per Kovatchev et al."""
    if df.empty:
        return np.nan
    glucose = df['glucose']
    rr = 1.509 * np.log(glucose) - 5.381
    risk = 10 * rr ** 2 if rr < 0 else 0
    return risk.mean()


def compute_hbgI(df: pd.DataFrame) -> float:
    if df.empty:
        return np.nan
    glucose = df['glucose']
    rr = 1.509 * np.log(glucose) - 5.381
    risk = 10 * rr ** 2 if rr > 0 else 0
    return risk.mean()


def compute_adrr(df: pd.DataFrame) -> float:
    """Average Daily Risk Range (ADRR)."""
    if df.empty:
        return np.nan
    lbgI = compute_lbgI(df)
    hbgI = compute_hbgI(df)
    return (lbgI + hbgI) / 2.0


def compute_gmi(df: pd.DataFrame) -> float:
    """Estimated A1C (GMI) based on mean glucose."""
    if df.empty:
        return np.nan
    mean_glucose = df['glucose'].mean()
    return 3.31 + 0.02392 * mean_glucose


def compute_clinical_metrics(patient: Patient) -> dict:
    """Aggregate clinical glycemic metrics for a patient.

    Returns a dict mapping metric names to values.
    """
    if patient.glucose is None or patient.glucose.empty:
        return {}
    df = patient.glucose.copy()
    df['glucose'] = pd.to_numeric(df['glucose'], errors='coerce')
    return {
        'TiR_%': compute_time_in_range(df),
        'TiBR_%': compute_time_below_range(df),
        'TiAR_%': compute_time_above_range(df),
        'LBGI': compute_lbgI(df),
        'HBGI': compute_hbgI(df),
        'ADRR': compute_adrr(df),
        'GMI': compute_gmi(df),
    }
