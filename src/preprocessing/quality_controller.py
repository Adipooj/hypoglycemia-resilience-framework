import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd

from src.core.stage import PipelineStage, RunContext
from src.core.patient import Patient
from src.constants import MIN_GLUCOSE_DAYS, MIN_GLUCOSE_DATA_PCT, EXPECTED_READINGS_PER_DAY
from src.preprocessing.validators import CANONICAL_SCHEMAS

logger = logging.getLogger("hypo_resilience.preprocessing.quality_controller")


class QualityController(PipelineStage):
    """
    Pipeline stage that checks data quality for all loaded patient records,
    identifies missing modalities/columns, filters patients based on clinical validity criteria,
    and exports a quality report and registry.
    """

    def __init__(self, context: RunContext):
        super().__init__("quality_controller", context)
        self.quality_report: pd.DataFrame | None = None
        self.patient_registry: pd.DataFrame | None = None

    def run(self, patients: List[Patient]) -> List[Patient]:
        """
        Runs quality audits on all patients. Validates physiological records and determines cohort eligibility.
        
        Args:
            patients (List[Patient]): List of patient objects to evaluate.
            
        Returns:
            List[Patient]: List of patients who meet the eligibility requirements.
        """
        records = []
        eligible_patients = []
        registry_entries = []

        # Minimum required glucose readings: 5 days * 288 readings/day * 70% threshold
        min_required_readings = int(MIN_GLUCOSE_DAYS * EXPECTED_READINGS_PER_DAY * MIN_GLUCOSE_DATA_PCT)

        for patient in patients:
            p_id = patient.patient_id
            glucose_df = patient.glucose
            
            # Count glucose entries
            glucose_rows = len(glucose_df) if glucose_df is not None else 0
            
            # Check eligibility: must have glucose data and exceed minimum readings threshold
            has_enough_glucose = glucose_rows >= min_required_readings
            
            # Determine available modalities
            avail_mods = patient.available_modalities
            
            # 1. Audit each modality
            for mod in ["glucose", "activity", "nutrition", "basal", "bolus", "sleep"]:
                df = getattr(patient, mod)
                row_entry = {
                    "patient_id": p_id,
                    "modality": mod,
                    "exists": df is not None,
                    "rows": len(df) if df is not None else 0,
                    "missing_cells": int(df.isna().sum().sum()) if df is not None else 0,
                    "duplicates": int(df.duplicated().sum()) if df is not None else 0,
                }
                
                # Check columns against canonical schema
                expected_cols = list(CANONICAL_SCHEMAS.get(mod, {}).keys())
                if df is not None:
                    missing_cols = sorted(set(expected_cols) - set(df.columns))
                    extra_cols = sorted(set(df.columns) - set(expected_cols))
                    row_entry["missing_columns"] = ", ".join(missing_cols)
                    row_entry["extra_columns"] = ", ".join(extra_cols)
                else:
                    row_entry["missing_columns"] = ", ".join(expected_cols)
                    row_entry["extra_columns"] = ""
                    
                records.append(row_entry)

            # 2. Add to cohort registry
            registry_entry = {
                "patient_id": p_id,
                "glucose_records": glucose_rows,
                "available_modalities": ", ".join(avail_mods),
                "is_eligible": has_enough_glucose,
                "reason": "" if has_enough_glucose else f"Insufficient glucose data ({glucose_rows} < {min_required_readings})"
            }
            registry_entries.append(registry_entry)

            if has_enough_glucose:
                eligible_patients.append(patient)
            else:
                self.logger.warning(
                    f"Patient {p_id} excluded from registry. Reason: {registry_entry['reason']}"
                )

        self.quality_report = pd.DataFrame(records)
        self.patient_registry = pd.DataFrame(registry_entries)
        
        self.logger.info(
            f"Quality control complete. {len(eligible_patients)} / {len(patients)} patients are eligible."
        )

        return eligible_patients

    def save(self) -> List[Path]:
        """Saves reports to the run context directory."""
        saved_paths = []
        if self.quality_report is not None:
            report_path = self.context.run_dir / "reports" / "quality_report.csv"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            self.quality_report.to_csv(report_path, index=False)
            saved_paths.append(report_path)

        if self.patient_registry is not None:
            registry_path = self.context.run_dir / "reports" / "patient_registry.csv"
            self.patient_registry.to_csv(registry_path, index=False)
            saved_paths.append(registry_path)

        return saved_paths
