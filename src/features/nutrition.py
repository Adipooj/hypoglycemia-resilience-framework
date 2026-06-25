import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from src.features.builder import register_feature

logger = logging.getLogger("hypo_resilience.features.nutrition")


@register_feature(category="nutrition", sources=["nutrition"])
def extract_nutrition_features(df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, Dict[str, str]]]:
    """
    Extracts metrics related to meal frequency, macronutrients, and timing.
    """
    features = {}
    metadata = {}

    has_carbs = "carbs" in df.columns

    if not has_carbs:
        cols = ["mean_daily_carbs", "mean_daily_protein", "mean_daily_fat", "mean_daily_fiber", "meal_count_daily", "meal_timing_var"]
        features = {c: np.nan for c in cols}
        metadata = {c: {"description": f"Nutrition metric: {c}", "units": "grams, count, or hours"} for c in cols}
        return features, metadata

    dates = df["timestamp"].dt.date

    # 1. Macronutrient Totals
    for macro in ["carbs", "protein", "fat", "fiber"]:
        if macro in df.columns and not df[macro].isna().all():
            daily_macro = df.groupby(dates)[macro].sum()
            features[f"mean_daily_{macro}"] = float(daily_macro.mean())
        else:
            features[f"mean_daily_{macro}"] = 0.0
            
        metadata[f"mean_daily_{macro}"] = {"description": f"Mean daily {macro} intake", "units": "grams"}

    # 2. Meal Counts and Timing
    if "carbs" in df.columns and not df["carbs"].isna().all():
        # Count events where carbs are ingested
        daily_meals = df[df["carbs"] > 0].groupby(dates)["carbs"].count()
        # Reindex to include all dates
        all_dates = df.groupby(dates)["carbs"].sum().index
        daily_meals = daily_meals.reindex(all_dates, fill_value=0)
        
        features["meal_count_daily"] = float(daily_meals.mean())

        # Meal hours standard deviation
        meal_indices = df[df["carbs"] > 0].index
        if len(meal_indices) > 2:
            meal_hours = df.loc[meal_indices, "timestamp"].dt.hour
            features["meal_timing_var"] = float(meal_hours.std())
        else:
            features["meal_timing_var"] = 0.0
    else:
        features["meal_count_daily"] = 0.0
        features["meal_timing_var"] = 0.0

    metadata["meal_count_daily"] = {"description": "Mean daily meal count (carbs > 0)", "units": "Count"}
    metadata["meal_timing_var"] = {"description": "Standard deviation of meal hours", "units": "hours"}

    return features, metadata
