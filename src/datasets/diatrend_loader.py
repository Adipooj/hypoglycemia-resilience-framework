import logging
from src.core.patient import Patient
from src.datasets.base_loader import BaseLoader

logger = logging.getLogger("hypo_resilience.datasets.diatrend_loader")


class DiaTrendLoader(BaseLoader):
    """
    Stub loader for the DiaTrend dataset to ensure dataset-agnostic pipeline design.
    """

    def load(self, patient_id: str) -> Patient:
        logger.warning("DiaTrendLoader is a stub. Returning empty Patient object.")
        return Patient(patient_id)
