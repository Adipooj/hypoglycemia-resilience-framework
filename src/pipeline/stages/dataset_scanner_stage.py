import logging
import json
from pathlib import Path
from typing import List, Any

from src.core.stage import PipelineStage, RunContext
from src.ingestion.dataset_scanner import scan_dataset

class DatasetScannerStage(PipelineStage):
    """Pipeline stage that scans the dataset directory and creates a manifest.

    The stage runs the existing ``scan_dataset`` function from the ingestion
    module and stores the resulting manifest JSON in the run's ``interim``
    directory.
    """

    def __init__(self, context: RunContext):
        super().__init__(name=self.__class__.__name__, context=context)
        self.logger = logging.getLogger(f"hypo_resilience.stages.{self.name}")
        self._output_path: Path | None = None

    def run(self, *args, **kwargs) -> Any:
        self.logger.info("Scanning dataset to build manifest.")
        manifest = scan_dataset()
        # Determine where to write the manifest for this run
        interim_dir = self.context.run_dir / "interim"
        interim_dir.mkdir(parents=True, exist_ok=True)
        self._output_path = interim_dir / "manifest.json"
        with open(self._output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4)
        self.logger.info(f"Manifest written to {self._output_path}")
        return manifest

    def save(self) -> List[Path]:
        # Return the path of the manifest if it was created
        return [self._output_path] if self._output_path else []
