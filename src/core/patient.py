from dataclasses import dataclass
from typing import List
import pandas as pd


@dataclass
class Patient:
    """
    Standardized dataclass holding raw/loaded clinical dataframes for a single patient.
    
    Attributes:
        patient_id (str): Unique patient ID.
        glucose (pd.DataFrame): Glucose readings (bg_ts, value).
        activity (pd.DataFrame): Physical activity metrics (activity_ts, activity_type, step_count, distance_m, met).
        nutrition (pd.DataFrame): Carbs/meals ingested (meal_ts, meal_type, meal_tag, carbs_g, prot_g, fat_g, fibre_g).
        basal (pd.DataFrame): Continuous insulin delivery (basal_ts, basal_dose, insulin_kind).
        bolus (pd.DataFrame): Discrete insulin injections (bolus_ts, bolus_dose).
        sleep (pd.DataFrame): Sleep state and heart rate logs (sleep_ts, sleep_level, step_count, heart_rate, stress_level_value, resting_heart_rate).
        demographics (pd.DataFrame): Demographic information (age, gender, diagnosis_duration, etc.).
    """
    patient_id: str
    glucose: pd.DataFrame | None = None
    activity: pd.DataFrame | None = None
    nutrition: pd.DataFrame | None = None
    basal: pd.DataFrame | None = None
    bolus: pd.DataFrame | None = None
    sleep: pd.DataFrame | None = None
    demographics: pd.DataFrame | None = None

    @property
    def available_modalities(self) -> List[str]:
        """Returns the list of modalities that are not None and have non-empty data."""
        modalities = []
        for mod in ["glucose", "activity", "nutrition", "basal", "bolus", "sleep", "demographics"]:
            df = getattr(self, mod)
            if df is not None and not df.empty:
                modalities.append(mod)
        return modalities

    def __repr__(self) -> str:
        return f"Patient(id={self.patient_id}, modalities={self.available_modalities})"
