import pandas as pd
from sklearn.ensemble import IsolationForest
from src.features import register_feature

@register_feature('outlier_detection')
def detect_outliers(df: pd.DataFrame, modality: str = 'glucose', contamination: float = 0.01):
    """Detect outliers per modality using IsolationForest.
    Returns a boolean Series indicating outlier rows.
    """
    if modality not in df.columns:
        raise ValueError(f"Modality {modality} not found in DataFrame columns.")
    model = IsolationForest(contamination=contamination, random_state=42)
    outlier_mask = model.fit_predict(df[[modality]]) == -1
    return outlier_mask
