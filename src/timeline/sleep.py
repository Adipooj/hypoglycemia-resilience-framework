import logging
import pandas as pd

from src.core.patient import Patient
from src.timeline.builder import register_merger, TimelineMerger

logger = logging.getLogger("hypo_resilience.timeline.sleep")


@register_merger("sleep")
class SleepMerger(TimelineMerger):
    """
    Timeline merger plugin for sleep logs and wearable telemetry.
    Aggregates sleep level (binary), heart rate, resting heart rate, and stress indices.
    """

    def merge(self, df: pd.DataFrame, patient: Patient) -> pd.DataFrame:
        sleep_df = patient.sleep
        if sleep_df is None or sleep_df.empty:
            logger.debug(f"Patient {patient.patient_id} has no sleep data.")
            df = df.copy()
            df["sleep"] = 0.0
            df["heart_rate"] = pd.NA
            df["resting_heart_rate"] = pd.NA
            df["stress"] = pd.NA
            return df

        raw_df = sleep_df.copy()
        raw_df["rounded_ts"] = raw_df["timestamp"].dt.round("5min")

        # Group and aggregate
        # - sleep (sleep_level): max (if asleep at all in the 5 min, sleep=1)
        # - heart_rate: mean
        # - resting_heart_rate: mean
        # - stress: mean
        agg_funcs = {
            "sleep": "max",
            "heart_rate": "mean",
            "resting_heart_rate": "mean",
            "stress": "mean",
        }

        grouped = (
            raw_df.groupby("rounded_ts")
            .agg(agg_funcs)
            .reset_index()
            .rename(columns={"rounded_ts": "timestamp"})
        )

        merged_df = df.merge(grouped, on="timestamp", how="left")

        # Fill sleep status NaNs with 0.0 (awake)
        merged_df["sleep"] = merged_df["sleep"].fillna(0.0)

        # For continuous metrics (heart rate, stress), we forward-fill up to 30 mins (6 steps)
        # to handle brief missing intervals in wearable data.
        for col in ["heart_rate", "resting_heart_rate", "stress"]:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].ffill(limit=6)

        return merged_df
