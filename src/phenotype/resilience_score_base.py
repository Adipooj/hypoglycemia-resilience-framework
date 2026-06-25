import abc
from typing import Any, Dict

class ResilienceScore(abc.ABC):
    """Abstract base class for resilience scoring plugins.

    Subclasses must implement ``score`` which returns a numeric resilience
    score for a given patient based on the repaired timeline and engineered
    feature matrix.
    """

    @abc.abstractmethod
    def score(self, patient_id: str, timeline: Any, features: Any) -> float:
        """Compute the resilience score.

        Args:
            patient_id: Identifier of the patient.
            timeline: ``Timeline`` object (or pandas DataFrame) containing the
                aligned clinical timeline.
            features: Feature matrix (pandas DataFrame) for the patient.
        Returns:
            A float representing the resilience score.
        """
        raise NotImplementedError
