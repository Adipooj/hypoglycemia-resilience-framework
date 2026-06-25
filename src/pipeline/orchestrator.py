import argparse
import importlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import yaml
import pkgutil

# Simple utility for checkpointing using Parquet via pandas
import pandas as pd

# Define where checkpoints are stored
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file for the pipeline.

    Args:
        config_path: Path to the YAML config file.
    Returns:
        Parsed configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_stage_classes(stage_package: str) -> List[Any]:
    """Dynamically import all stage classes from a given package.

    The convention is that each module in the package defines a class
    inheriting from ``PipelineStage`` and exposing a ``run`` method.
    """
    pkg = importlib.import_module(stage_package)
    stage_classes = []
    for _, module_name, is_pkg in pkgutil.iter_modules(pkg.__path__):
        if is_pkg:
            continue
        module = importlib.import_module(f"{stage_package}.{module_name}")
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and hasattr(attr, "run"):
                stage_classes.append(attr)
    return stage_classes


def save_checkpoint(stage_name: str, data: Any) -> None:
    """Persist stage output to a Parquet checkpoint.

    Args:
        stage_name: Identifier for the stage.
        data: Data to be saved; must be convertible to a pandas DataFrame.
    """
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        # Attempt to coerce into a DataFrame; fallback to serialize as JSON string
        try:
            df = pd.DataFrame(data)
        except Exception:
            path = CHECKPOINT_DIR / f"{stage_name}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return
    checkpoint_path = CHECKPOINT_DIR / f"{stage_name}.parquet"
    df.to_parquet(checkpoint_path, index=False)


def load_checkpoint(stage_name: str) -> Any:
    """Load a checkpoint if it exists.

    Returns the original data structure (DataFrame or JSON object).
    """
    parquet_path = CHECKPOINT_DIR / f"{stage_name}.parquet"
    json_path = CHECKPOINT_DIR / f"{stage_name}.json"
    if parquet_path.is_file():
        return pd.read_parquet(parquet_path)
    if json_path.is_file():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def execute_stage(stage_cls: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    """Instantiate and run a pipeline stage.

    The stage receives a mutable ``context`` dictionary that can be used to
    share data between stages. The function returns the potentially mutated
    context for the next stage.
    """
    stage_name = getattr(stage_cls, "__name__", str(stage_cls))
    print(f"[Orchestrator] Running stage: {stage_name}")
    # Check for existing checkpoint to enable resume
    checkpoint = load_checkpoint(stage_name)
    if checkpoint is not None:
        print(f"[Orchestrator] Loaded checkpoint for {stage_name}, skipping execution.")
        context[stage_name] = checkpoint
        return context
    try:
        stage = stage_cls()
        result = stage.run(context)
        # Save result as checkpoint (if the stage returns anything useful)
        if result is not None:
            save_checkpoint(stage_name, result)
            context[stage_name] = result
        else:
            context[stage_name] = {}
    except Exception as e:
        print(f"[Orchestrator] Error in stage {stage_name}: {e}")
        traceback.print_exc()
        sys.exit(1)
    return context


def main(argv: List[str] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Hypoglycemia Resilience Discovery pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "configs" / "pipeline.yaml"),
        help="Path to the pipeline YAML configuration file.",
    )
    parser.add_argument(
        "--skip",
        type=str,
        nargs="*",
        default=[],
        help="Stage names to skip during execution.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoints where possible.",
    )
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))
    stage_package = config.get("stage_package", "src.pipeline.stages")
    stages = config.get("stages", [])
    if not stages:
        # Fallback: discover all stage classes automatically
        stages = [cls.__name__ for cls in get_stage_classes(stage_package)]

    # Build a simple linear execution order respecting explicit list
    context: Dict[str, Any] = {"config": config}

    for stage_name in stages:
        if stage_name in args.skip:
            print(f"[Orchestrator] Skipping stage {stage_name} as requested.")
            continue
        # Dynamically import the stage class
        module_path, cls_name = stage_name.rsplit(".", 1)
        module = importlib.import_module(module_path)
        stage_cls = getattr(module, cls_name)
        context = execute_stage(stage_cls, context)

    print("[Orchestrator] Pipeline execution completed.")


if __name__ == "__main__":
    main()
