from pathlib import Path
import pandas as pd

class Timeline:
    """
    An immutable wrapper around the aligned 5-minute patient timeline dataframe.
    Downstream modules use this representation to read aligned physiological time series.
    """

    def __init__(self, patient_id: str, df: pd.DataFrame):
        self._patient_id = patient_id
        # Keep an immutable copy
        self._df = df.copy().reset_index(drop=True)
        # Ensure timestamp is datetime
        if "timestamp" in self._df.columns:
            self._df["timestamp"] = pd.to_datetime(self._df["timestamp"])

    @property
    def patient_id(self) -> str:
        """Returns the patient ID associated with this timeline."""
        return self._patient_id

    @property
    def df(self) -> pd.DataFrame:
        """Returns a copy of the underlying dataframe to preserve immutability."""
        return self._df.copy()

    def get_column(self, col_name: str) -> pd.Series | None:
        """Safely retrieves a column if it exists, otherwise returns None."""
        if col_name in self._df.columns:
            return self._df[col_name].copy()
        return None

    def save(self, path: Path) -> Path:
        """Saves the timeline as a Parquet file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_parquet(path, index=False)
        return path

    @classmethod
    def load(cls, path: Path, patient_id: str = None) -> "Timeline":
        """Loads a timeline from a Parquet file."""
        if not path.exists():
            raise FileNotFoundError(f"Timeline parquet not found at: {path}")
        df = pd.read_parquet(path)
        p_id = patient_id or path.stem
        return cls(p_id, df)

    def __repr__(self) -> str:
        shape_str = f"shape={self._df.shape}"
        cols = list(self._df.columns)
        return f"Timeline(patient_id={self._patient_id}, {shape_str}, columns={cols})"
