import json
import logging
from pathlib import Path
import pandas as pd

from src.config import INTERIM_DATA, UOM_ROOT
from src.core.patient import Patient
from src.datasets.base_loader import BaseLoader
from src.ingestion.dataset_scanner import scan_dataset
from src.preprocessing.validators import CanonicalSchemaAdapter
from src.utils.io import load_csv

logger = logging.getLogger("hypo_resilience.datasets.uom_loader")


class UOMLoader(BaseLoader):
    """
    Concrete implementation of BaseLoader for loading UOM dataset clinical records,
    adapting them to standard canonical schemas.
    """

    # Mapping from UOM raw column names to canonical schema names
    COLUMN_MAPPINGS = {
        "glucose": {
            "bg_ts": "timestamp",
            "value": "glucose",
        },
        "basal": {
            "basal_ts": "timestamp",
            "basal_dose": "basal",
        },
        "bolus": {
            "bolus_ts": "timestamp",
            "bolus_dose": "bolus",
        },
        "nutrition": {
            "meal_ts": "timestamp",
            "carbs_g": "carbs",
            "prot_g": "protein",
            "fat_g": "fat",
            "fibre_g": "fiber",
        },
        "activity": {
            "activity_ts": "timestamp",
            "activity_type": "activity_type",
            "step_count": "steps",
            "distance_m": "distance",
            "met": "MET",
        },
        "sleep": {
            "sleep_ts": "timestamp",
            "sleep_level": "sleep",
            "heart_rate": "heart_rate",
            "resting_heart_rate": "resting_heart_rate",
            "stress_level_value": "stress",
        },
        "demographics": {
            "patient_id": "patient_id",
        }
    }

    def __init__(self, manifest_path: Path = None):
        self.manifest_path = manifest_path or (INTERIM_DATA / "manifest.json")
        self.manifest = self._load_or_create_manifest()

    def _load_or_create_manifest(self) -> dict:
        """Loads the dataset manifest from file, scanning UOM root if missing."""
        if not self.manifest_path.exists():
            logger.info("Manifest not found. Scanning dataset to create it...")
            return scan_dataset()
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read manifest at {self.manifest_path}. Re-scanning. Error: {e}")
            return scan_dataset()

    def load(self, patient_id: str) -> Patient:
        """
        Loads all available modalities for a patient and adapts them to canonical schemas.
        """
        if patient_id not in self.manifest:
            raise ValueError(f"Patient {patient_id} not found in UOM manifest.")

        patient = Patient(patient_id)
        files = self.manifest[patient_id]

        for modality, filepath in files.items():
            if modality not in self.COLUMN_MAPPINGS:
                logger.warning(f"Unknown modality '{modality}' for patient {patient_id}. Skipping.")
                continue

            path = Path(filepath)
            if not path.exists():
                logger.warning(f"File for modality '{modality}' not found at {path}. Skipping.")
                continue

            try:
                # Load raw CSV
                df_raw = load_csv(path)
                
                # Adapt to canonical schema
                mapping = self.COLUMN_MAPPINGS[modality]
                df_canonical = CanonicalSchemaAdapter.adapt(df_raw, modality, mapping)
                
                setattr(patient, modality, df_canonical)
                logger.debug(f"Successfully loaded and adapted modality '{modality}' for patient {patient_id}.")
            except Exception as e:
                logger.exception(f"Error loading modality '{modality}' for patient {patient_id}: {e}")
                
        return patient
