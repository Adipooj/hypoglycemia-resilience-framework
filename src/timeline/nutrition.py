import logging
import pandas as pd

from src.core.patient import Patient
from src.timeline.builder import register_merger, TimelineMerger

logger = logging.getLogger("hypo_resilience.timeline.nutrition")


@register_merger("nutrition")
class NutritionMerger(TimelineMerger):
    """
    Timeline merger plugin for nutrition logs.
    Aggregates macronutrients (carbs, protein, fat, fiber) into 5-minute intervals.
    """

    def merge(self, df: pd.DataFrame, patient: Patient) -> pd.DataFrame:
        nutrition_df = patient.nutrition
        if nutrition_df is None or nutrition_df.empty:
            logger.debug(f"Patient {patient.patient_id} has no nutrition data.")
            df = df.copy()
            df["carbs"] = 0.0
            df["protein"] = 0.0
            df["fat"] = 0.0
            df["fiber"] = 0.0
            return df

        raw_df = nutrition_df.copy()
        raw_df["rounded_ts"] = raw_df["timestamp"].dt.round("5min")

        # Sum macronutrients in case of multiple logs in the same 5-minute bucket
        grouped = (
            raw_df.groupby("rounded_ts")[["carbs", "protein", "fat", "fiber"]]
            .sum()
            .reset_index()
            .rename(columns={"rounded_ts": "timestamp"})
        )

        merged_df = df.merge(grouped, on="timestamp", how="left")

        # Fill NaNs with 0.0 (no food consumed)
        for col in ["carbs", "protein", "fat", "fiber"]:
            merged_df[col] = merged_df[col].fillna(0.0)

        return merged_df
