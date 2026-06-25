import logging
from typing import Any, Dict, List
import pandas as pd
from src.preprocessing.datetime_utils import parse_datetime

logger = logging.getLogger("hypo_resilience.preprocessing.validators")

# ==============================================================================
# Canonical Schemas Defining Column Names and Datatypes
# ==============================================================================
CANONICAL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "glucose": {
        "timestamp": "datetime64[ns]",
        "glucose": "float64",
    },
    "basal": {
        "timestamp": "datetime64[ns]",
        "basal": "float64",
    },
    "bolus": {
        "timestamp": "datetime64[ns]",
        "bolus": "float64",
    },
    "nutrition": {
        "timestamp": "datetime64[ns]",
        "carbs": "float64",
        "protein": "float64",
        "fat": "float64",
        "fiber": "float64",
    },
    "activity": {
        "timestamp": "datetime64[ns]",
        "activity_type": "object",
        "steps": "float64",
        "distance": "float64",
        "MET": "float64",
    },
    "sleep": {
        "timestamp": "datetime64[ns]",
        "sleep": "float64",
        "heart_rate": "float64",
        "resting_heart_rate": "float64",
        "stress": "float64",
    },
    "demographics": {
        "patient_id": "object",
    }
}

# ==============================================================================
# Physiological Bounds for Quality Control & Validation
# ==============================================================================
PHYSIOLOGICAL_BOUNDS = {
    "glucose": {"min": 1.0, "max": 35.0},          # mmol/L
    "basal": {"min": 0.0, "max": 20.0},            # units per hour or dose
    "bolus": {"min": 0.0, "max": 50.0},            # units
    "carbs": {"min": 0.0, "max": 250.0},           # grams
    "protein": {"min": 0.0, "max": 150.0},         # grams
    "fat": {"min": 0.0, "max": 150.0},             # grams
    "fiber": {"min": 0.0, "max": 50.0},            # grams
    "steps": {"min": 0.0, "max": 10000.0},         # steps per 5-min
    "heart_rate": {"min": 30.0, "max": 220.0},     # bpm
    "resting_heart_rate": {"min": 30.0, "max": 120.0}, # bpm
    "stress": {"min": 0.0, "max": 100.0},          # index
}


class CanonicalSchemaAdapter:
    """
    Adapts and validates raw dataframes into standard canonical schemas.
    """

    @staticmethod
    def adapt(df: pd.DataFrame, modality: str, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Maps column names, parses datetime columns, and standardizes types.
        
        Args:
            df (pd.DataFrame): Raw input dataframe.
            modality (str): Name of clinical modality.
            column_mapping (Dict[str, str]): Dict mapping raw columns to canonical columns.
            
        Returns:
            pd.DataFrame: DataFrame aligned to canonical schema, or empty if invalid.
        """
        if df is None or df.empty:
            return pd.DataFrame(columns=list(CANONICAL_SCHEMAS.get(modality, {}).keys()))

        df = df.copy()

        # 1. Rename columns according to mapping
        df = df.rename(columns=column_mapping)

        # 2. Select only columns defined in the canonical schema
        schema = CANONICAL_SCHEMAS.get(modality)
        if not schema:
            raise ValueError(f"Modality '{modality}' does not have a canonical schema defined.")

        # Ensure all columns in schema exist (create with NaN if missing, EXCEPT timestamp)
        for col in schema:
            if col not in df.columns:
                if col == "timestamp" and modality != "demographics":
                    logger.error(f"Cannot adapt modality '{modality}': missing timestamp.")
                    return pd.DataFrame(columns=list(schema.keys()))
                df[col] = pd.NA

        df = df[list(schema.keys())]

        # 3. Parse timestamp
        if modality != "demographics":
            df = parse_datetime(df, "timestamp")
            if df.empty:
                return pd.DataFrame(columns=list(schema.keys()))

        # 4. Standardize types
        for col, dtype in schema.items():
            if col == "timestamp" or col == "patient_id":
                continue
            # Try to convert to float64 for metrics
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        # 5. Apply physiological bounds checks
        df = CanonicalSchemaAdapter.apply_bounds(df, modality)

        return df

    @staticmethod
    def apply_bounds(df: pd.DataFrame, modality: str) -> pd.DataFrame:
        """Clips or filters data points based on physiological reasonability."""
        df = df.copy()
        for col, bounds in PHYSIOLOGICAL_BOUNDS.items():
            if col in df.columns:
                # Count out of bound values
                too_low = (df[col] < bounds["min"]).sum()
                too_high = (df[col] > bounds["max"]).sum()
                if too_low > 0 or too_high > 0:
                    logger.debug(
                        f"Modality '{modality}' column '{col}' has out-of-bound values: "
                        f"{too_low} low (<{bounds['min']}), {too_high} high (>{bounds['max']})."
                    )
                # Apply clipping or replace with NaN. Let's clip to valid physiological ranges
                df[col] = df[col].clip(bounds["min"], bounds["max"])
        return df

    @staticmethod
    def validate(df: pd.DataFrame, modality: str) -> bool:
        """
        Validates whether a dataframe conforms to its canonical schema.
        """
        if df is None:
            return False

        schema = CANONICAL_SCHEMAS.get(modality)
        if not schema:
            return False

        # Check column presence
        for col in schema:
            if col not in df.columns:
                logger.warning(f"Validation failed for '{modality}': missing column '{col}'")
                return False

        return True
