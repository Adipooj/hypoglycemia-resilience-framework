import logging
import pandas as pd

from src.core.patient import Patient
from src.timeline.builder import register_merger, TimelineMerger

logger = logging.getLogger("hypo_resilience.timeline.insulin")


@register_merger("insulin")
class InsulinMerger(TimelineMerger):
    """
    Timeline merger plugin for insulin doses (basal and bolus).
    Aggregates doses falling inside the same 5-minute window by summing them.
    """

    def merge(self, df: pd.DataFrame, patient: Patient) -> pd.DataFrame:
        df = df.copy()

        # 1. Process Basal Insulin
        basal_df = patient.basal
        if basal_df is not None and not basal_df.empty:
            raw_basal = basal_df.copy()
            raw_basal["rounded_ts"] = raw_basal["timestamp"].dt.round("5min")
            grouped_basal = (
                raw_basal.groupby("rounded_ts")["basal"]
                .sum()
                .reset_index()
                .rename(columns={"rounded_ts": "timestamp"})
            )
            df = df.merge(grouped_basal, on="timestamp", how="left")
            df["basal"] = df["basal"].fillna(0.0)
        else:
            logger.debug(f"Patient {patient.patient_id} has no basal insulin data.")
            df["basal"] = 0.0

        # 2. Process Bolus Insulin
        bolus_df = patient.bolus
        if bolus_df is not None and not bolus_df.empty:
            raw_bolus = bolus_df.copy()
            raw_bolus["rounded_ts"] = raw_bolus["timestamp"].dt.round("5min")
            grouped_bolus = (
                raw_bolus.groupby("rounded_ts")["bolus"]
                .sum()
                .reset_index()
                .rename(columns={"rounded_ts": "timestamp"})
            )
            df = df.merge(grouped_bolus, on="timestamp", how="left")
            df["bolus"] = df["bolus"].fillna(0.0)
        else:
            logger.debug(f"Patient {patient.patient_id} has no bolus insulin data.")
            df["bolus"] = 0.0

        return df
