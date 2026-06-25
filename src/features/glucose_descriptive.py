import pandas as pd
from src.features.registry import register_feature

@register_feature('glucose_descriptive')
def compute_glucose_descriptive(df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic descriptive statistics for glucose time series.
    Returns a DataFrame with a single row of features.
    Features:
        - mean_glucose
        - std_glucose
        - min_glucose
        - max_glucose
        - median_glucose
        - cv_glucose (coefficient of variation)
    """
    glucose = df['glucose']
    features = {
        'mean_glucose': glucose.mean(),
        'std_glucose': glucose.std(),
        'min_glucose': glucose.min(),
        'max_glucose': glucose.max(),
        'median_glucose': glucose.median(),
        'cv_glucose': glucose.std() / glucose.mean() if glucose.mean() != 0 else 0,
    }
    return pd.DataFrame([features])
