from abc import ABC, abstractmethod
import datetime
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List

logger = logging.getLogger("hypo_resilience.core.stage")


class RunContext:
    """
    Manages the context of a single pipeline run, including directories,
    configurations, and the execution manifest.
    """

    def __init__(self, run_dir: Path, config: Dict[str, Any]):
        self.run_dir = run_dir
        self.config = config
        self.manifest_path = self.run_dir / "manifest.json"
        self.manifest: Dict[str, Any] = {
            "run_id": run_dir.name,
            "timestamp": datetime.datetime.now().isoformat(),
            "config": config,
            "stages": {},
            "status": "IN_PROGRESS",
        }
        self._init_run_dir()

    def _init_run_dir(self):
        """Creates the run directory and writes the initial manifest."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Create subdirectories for stages
        (self.run_dir / "interim").mkdir(exist_ok=True)
        (self.run_dir / "processed").mkdir(exist_ok=True)
        (self.run_dir / "features").mkdir(exist_ok=True)
        (self.run_dir / "reports").mkdir(exist_ok=True)
        (self.run_dir / "figures").mkdir(exist_ok=True)
        (self.run_dir / "models").mkdir(exist_ok=True)
        self.save_manifest()

    def update_stage(
        self,
        stage_name: str,
        status: str,
        execution_time: float,
        outputs: List[str] = None,
        error_message: str = None,
    ):
        """Updates a stage's status in the manifest."""
        self.manifest["stages"][stage_name] = {
            "status": status,
            "execution_time_seconds": round(execution_time, 3),
            "completed_at": datetime.datetime.now().isoformat(),
            "outputs": outputs or [],
        }
        if error_message:
            self.manifest["stages"][stage_name]["error"] = error_message
        self.save_manifest()

    def mark_completed(self, status: str = "COMPLETED"):
        """Marks the entire run as completed."""
        self.manifest["status"] = status
        self.manifest["end_timestamp"] = datetime.datetime.now().isoformat()
        self.save_manifest()
        logger.info(f"Pipeline run {self.manifest['run_id']} finished with status: {status}")

    def save_manifest(self):
        """Saves the run manifest to disk."""
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=4)


class PipelineStage(ABC):
    """
    Base class representing a single step in the research pipeline.
    """

    def __init__(self, name: str, context: RunContext):
        self.name = name
        self.context = context
        self.logger = logging.getLogger(f"hypo_resilience.stages.{name}")

    def execute(self, *args, **kwargs) -> Any:
        """
        Wraps run() and handles timing, logging, status updates, and exceptions.
        """
        self.logger.info(f"Starting stage: {self.name}")
        start_time = time.time()
        try:
            result = self.run(*args, **kwargs)
            duration = time.time() - start_time
            self.logger.info(f"Stage {self.name} completed successfully in {duration:.2f} seconds")
            
            # Save if save method implemented
            outputs = self.save()
            
            self.context.update_stage(
                stage_name=self.name,
                status="SUCCESS",
                execution_time=duration,
                outputs=[str(Path(o).relative_to(self.context.run_dir)) for o in outputs] if outputs else []
            )
            return result
        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception(f"Stage {self.name} failed after {duration:.2f} seconds")
            self.context.update_stage(
                stage_name=self.name,
                status="FAILED",
                execution_time=duration,
                error_message=str(e)
            )
            raise e

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Executes core logic for this stage."""
        pass

    @abstractmethod
    def save(self) -> List[Path]:
        """Saves the outputs generated in this stage. Returns list of saved file paths."""
        pass
