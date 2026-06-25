import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd


class FeatureMatrix:
    """
    Manages the cohort-wide engineered feature matrix (patients as rows, features as columns)
    and handles loading/saving alongside the feature metadata sidecar.
    """

    def __init__(self, df: pd.DataFrame, metadata: Dict[str, Dict[str, Any]] = None):
        self._df = df.copy()
        if "patient_id" in self._df.columns:
            self._df = self._df.set_index("patient_id")
        self._metadata = metadata or {}

    @property
    def df(self) -> pd.DataFrame:
        """Returns the features dataframe with patient_id as the index."""
        return self._df.copy()

    @property
    def metadata(self) -> Dict[str, Dict[str, Any]]:
        """Returns the sidecar features metadata."""
        return self._metadata.copy()

    def get_features_by_category(self, category: str) -> pd.DataFrame:
        """Slices the dataframe to only include features from a specific category."""
        matching_cols = [
            col for col, meta in self._metadata.items()
            if meta.get("category") == category and col in self._df.columns
        ]
        return self._df[matching_cols].copy()

    def save(self, parquet_path: Path) -> Path:
        """
        Saves the dataframe to parquet and the feature metadata to a sidecar JSON.
        """
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        # Reset index to include patient_id in the parquet
        output_df = self._df.reset_index()
        output_df.to_parquet(parquet_path, index=False)

        # Write sidecar JSON
        json_path = parquet_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=4)

        return parquet_path

    @classmethod
    def load(cls, parquet_path: Path) -> "FeatureMatrix":
        """
        Loads a feature matrix and its sidecar metadata from disk.
        """
        if not parquet_path.exists():
            raise FileNotFoundError(f"Feature matrix parquet not found at: {parquet_path}")

        df = pd.read_parquet(parquet_path)
        
        json_path = parquet_path.with_suffix(".json")
        metadata = {}
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        return cls(df, metadata)

    def __repr__(self) -> str:
        return f"FeatureMatrix(shape={self._df.shape}, num_metadata_entries={len(self._metadata)})"
