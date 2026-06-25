import logging
from src.datasets.uom_loader import UOMLoader
from src.core.patient import Patient

logger = logging.getLogger("hypo_resilience.ingestion.patient_loader")


def load_patient(patient_id: str) -> Patient:
    """
    Legacy loader function preserved as part of the ingestion layer.
    Routes the loading through the canonical UOMLoader.
    """
    logger.info(f"Loading patient {patient_id} via legacy ingestion entry point.")
    loader = UOMLoader()
    return loader.load(patient_id)


if __name__ == "__main__":
    p = load_patient("2301")
    print(f"Loaded legacy wrapper: {p}")