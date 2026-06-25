import logging
import pandas as pd

from src.core.patient import Patient
from src.timeline.builder import register_merger, TimelineMerger

logger = logging.getLogger("hypo_resilience.timeline.activity")


@register_merger("activity")
class ActivityMerger(TimelineMerger):
    """
    Timeline merger plugin for physical activity data.
    Aggregates steps, distance, MET, and activity types into 5-minute intervals.
    """

    def merge(self, df: pd.DataFrame, patient: Patient) -> pd.DataFrame:
        activity_df = patient.activity
        if activity_df is None or activity_df.empty:
            logger.debug(f"Patient {patient.patient_id} has no activity data.")
            df = df.copy()
            df["steps"] = 0.0
            df["distance"] = 0.0
            df["MET"] = 1.0
            df["activity_type"] = "SEDENTARY"
            return df

        raw_df = activity_df.copy()
        raw_df["rounded_ts"] = raw_df["timestamp"].dt.round("5min")

        # Group and aggregate
        # - steps: sum
        # - distance: sum
        # - MET: max (to capture peak exertion)
        # - activity_type: most frequent (mode)
        agg_funcs = {
            "steps": "sum",
            "distance": "sum",
            "MET": "max",
            "activity_type": lambda x: x.mode().iloc[0] if not x.mode().empty else "SEDENTARY",
        }

        grouped = (
            raw_df.groupby("rounded_ts")
            .agg(agg_funcs)
            .reset_index()
            .rename(columns={"rounded_ts": "timestamp"})
        )

        merged_df = df.merge(grouped, on="timestamp", how="left")

        # Fill missing values
        merged_df["steps"] = merged_df["steps"].fillna(0.0)
        merged_df["distance"] = merged_df["distance"].fillna(0.0)
        merged_df["MET"] = merged_df["MET"].fillna(1.0)
        merged_df["activity_type"] = merged_df["activity_type"].fillna("SEDENTARY")

        return merged_df
