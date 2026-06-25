import logging
from pathlib import Path
from typing import Any, Dict, List, Type
import pandas as pd

from src.core.stage import PipelineStage, RunContext
from src.core.patient import Patient
from src.core.timeline import Timeline

logger = logging.getLogger("hypo_resilience.timeline.builder")

# Global registry for timeline merger plugins
MERGER_REGISTRY: Dict[str, Type["TimelineMerger"]] = {}


def register_merger(name: str):
    """Decorator to register a timeline merger class."""
    def decorator(cls: Type["TimelineMerger"]):
        MERGER_REGISTRY[name] = cls
        logger.debug(f"Registered timeline merger plugin: {name}")
        return cls
    return decorator


class TimelineMerger(Any):
    """Base class for all timeline merger plugins."""

    def merge(self, df: pd.DataFrame, patient: Patient) -> pd.DataFrame:
        """
        Merges a clinical modality from the Patient object into the master timeline.
        
        Args:
            df (pd.DataFrame): Current master timeline dataframe (contains at least 'timestamp').
            patient (Patient): Patient object containing raw modality data.
            
        Returns:
            pd.DataFrame: Merged timeline dataframe.
        """
        raise NotImplementedError


class TimelineBuilder(PipelineStage):
    """
    Coordinates building a unified 5-minute clinical timeline for a patient.
    Loads and runs all active merger plugins sequentially.
    """

    def __init__(self, patient: Patient, context: RunContext = None):
        super().__init__("timeline_builder", context)
        self.patient = patient
        self.timeline: Timeline | None = None

        # Resolve frequency and active mergers from config or defaults
        if context and "timeline" in context.config:
            self.frequency = context.config["timeline"].get("frequency", "5min")
            self.active_mergers = context.config["timeline"].get(
                "active_mergers", ["glucose", "insulin", "activity", "nutrition", "sleep"]
            )
        else:
            self.frequency = "5min"
            self.active_mergers = ["glucose", "insulin", "activity", "nutrition", "sleep"]

    def _build_master_index(self) -> pd.DataFrame:
        """Establishes the master 5-minute timestamp index from glucose logs."""
        glucose_df = self.patient.glucose
        if glucose_df is None or glucose_df.empty:
            raise ValueError(f"Patient {self.patient.patient_id} has no glucose data to anchor timeline.")

        start = glucose_df["timestamp"].min().floor("5min")
        end = glucose_df["timestamp"].max().ceil("5min")

        # Create master index
        timeline_df = pd.DataFrame(
            {"timestamp": pd.date_range(start, end, freq=self.frequency)}
        )
        logger.debug(f"Master index created with {len(timeline_df)} intervals from {start} to {end}.")
        return timeline_df

    def run(self) -> Timeline:
        """
        Executes all active merger plugins to build the complete aligned timeline.
        """
        # Ensure plugins are loaded and registered
        self._import_mergers()

        # Start with master index
        df = self._build_master_index()

        # Run registered mergers
        for merger_name in self.active_mergers:
            if merger_name not in MERGER_REGISTRY:
                logger.warning(f"Merger plugin '{merger_name}' is active but not registered. Skipping.")
                continue

            merger = MERGER_REGISTRY[merger_name]()
            logger.info(f"Merging modality: {merger_name}")
            df = merger.merge(df, self.patient)

        # Build final immutable Timeline object
        self.timeline = Timeline(self.patient.patient_id, df)
        return self.timeline

    # ---------------------------------------------------------------------
    #   Timeline Repair, Validation, and Summarization
    # ---------------------------------------------------------------------
    def repair(self) -> pd.DataFrame:
        """Repair structural problems in the built timeline.

        * Duplicate timestamps → keep the first occurrence, drop others.
        * Ensure timestamp column is monotonic increasing.
        * Forward-fill missing values for numeric columns.
        * Interpolate missing numeric values using linear interpolation.
        """
        if self.timeline is None:
            raise RuntimeError("Timeline must be built before repair.")
        df = self.timeline.data.copy()

        # Drop duplicate timestamps, keep first
        before_dups = len(df)
        df = df.drop_duplicates(subset="timestamp", keep="first")
        logger.info(f"Dropped {before_dups - len(df)} duplicate timestamps.")

        # Ensure ordering
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Forward fill then linear interpolate numeric columns
        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = df[numeric_cols].ffill().interpolate(method="linear")
        self.timeline = Timeline(self.patient.patient_id, df)
        return df

    def validate(self) -> Dict[str, Any]:
        """Validate scientific correctness of the timeline.

        Returns a dictionary with validation metrics, e.g.:
        * impossible glucose values (<0 or > 500 mg/dL)
        * modality coverage percentages
        * continuity score (fraction of uninterrupted 5-min windows)
        * missingness per modality
        """
        if self.timeline is None:
            raise RuntimeError("Timeline must be built before validation.")
        df = self.timeline.data
        metrics: Dict[str, Any] = {}

        # Glucose range check (example thresholds)
        if "glucose" in df.columns:
            invalid_glucose = df[(df["glucose"] < 0) | (df["glucose"] > 500)]
            metrics["invalid_glucose_percent"] = 100 * len(invalid_glucose) / len(df)
        else:
            metrics["invalid_glucose_percent"] = None

        # Modality coverage (percentage of non-null entries)
        coverage = df.notnull().mean().to_dict()
        metrics["modality_coverage"] = coverage

        # Continuity: expected number of rows vs actual
        expected_rows = int(((df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 300) + 1)
        metrics["expected_rows"] = expected_rows
        metrics["actual_rows"] = len(df)
        metrics["continuity_score"] = len(df) / expected_rows if expected_rows else 0

        return metrics

    def summarize(self) -> Dict[str, Any]:
        """Produce a concise summary of timeline statistics for reporting.
        Returns a dictionary that can be serialized to JSON or HTML.
        """
        if self.timeline is None:
            raise RuntimeError("Timeline must be built before summarization.")
        df = self.timeline.data
        summary: Dict[str, Any] = {}
        summary["patient_id"] = self.patient.patient_id
        summary["start"] = df["timestamp"].min().isoformat()
        summary["end"] = df["timestamp"].max().isoformat()
        summary["duration_minutes"] = int((df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 60)
        summary["row_count"] = len(df)
        # Basic modality stats
        for col in df.columns:
            if col == "timestamp":
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                summary[f"{col}_mean"] = df[col].mean()
                summary[f"{col}_std"] = df[col].std()
        return summary

    def save(self) -> List[Path]:
        """Saves the final timeline parquet to the run context directory."""
        if self.timeline is None:
            logger.warning("Timeline has not been built yet. Save skipped.")
            return []

        # Save to run context
        processed_dir = self.context.run_dir / "processed" if self.context else Path("data/processed")
        out_path = processed_dir / f"{self.patient.patient_id}.parquet"
        self.timeline.save(out_path)
        return [out_path]

    def _import_mergers(self):
        """Discovers and imports merger files to trigger registration decorators."""
        # Standard explicit imports to register the plugin classes
        import src.timeline.glucose
        import src.timeline.insulin
        import src.timeline.activity
        import src.timeline.nutrition
        import src.timeline.sleep
