import logging
import pandas as pd

from src.core.patient import Patient
from src.timeline.builder import register_merger, TimelineMerger

logger = logging.getLogger("hypo_resilience.timeline.glucose")


@register_merger("glucose")
class GlucoseMerger(TimelineMerger):
    """
    Timeline merger plugin for glucose readings.
    Rounds timestamps to the nearest 5-minute mark and takes the mean in case of collisions.
    """

    def merge(self, df: pd.DataFrame, patient: Patient) -> pd.DataFrame:
        glucose_df = patient.glucose
        if glucose_df is None or glucose_df.empty:
            logger.warning(f"Patient {patient.patient_id} has no glucose data. Adding empty column.")
            df = df.copy()
            df["glucose"] = pd.NA
            return df

        # Copy and round timestamps to nearest 5 minutes
        raw_df = glucose_df.copy()
        raw_df["rounded_ts"] = raw_df["timestamp"].dt.round("5min")

        # Handle duplicates: group by the rounded timestamp and take the mean
        grouped = (
            raw_df.groupby("rounded_ts")["glucose"]
            .mean()
            .reset_index()
            .rename(columns={"rounded_ts": "timestamp"})
        )

        # Merge with master index
        merged_df = df.merge(grouped, on="timestamp", how="left")
        return merged_df
