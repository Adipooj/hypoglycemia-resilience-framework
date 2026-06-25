import logging
from pathlib import Path
from typing import List, Any
from src.core.stage import PipelineStage, RunContext

class PlaceholderStage(PipelineStage):
    """A minimal placeholder stage that does nothing and returns empty outputs."""
    def __init__(self, context: RunContext):
        super().__init__(name=self.__class__.__name__, context=context)
        self.logger = logging.getLogger(f"hypo_resilience.stages.{self.name}")

    def run(self, *args, **kwargs) -> Any:
        self.logger.info("Running PlaceholderStage (no operation).")
        return {}

    def save(self) -> List[Path]:
        # No outputs to save
        return []
