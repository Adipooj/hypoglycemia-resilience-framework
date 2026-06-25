from abc import ABC, abstractmethod
from src.core.patient import Patient


class BaseLoader(ABC):
    """
    Abstract base class for all dataset loaders. 
     Decouples the research pipeline from specific dataset layouts (e.g., UOM, DiaTrend, OpenAPS).
    """

    @abstractmethod
    def load(self, patient_id: str) -> Patient:
        """
        Loads and returns a Patient object populated with dataframes for each available modality.
        
        Args:
            patient_id (str): ID of the patient to load.
            
        Returns:
            Patient: Fully loaded Patient object.
        """
        pass
