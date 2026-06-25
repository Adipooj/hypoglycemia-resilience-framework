import json
from pathlib import Path
import pandas as pd

# Directory for storing checkpoint files
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def save_checkpoint(stage_name: str, data: any) -> None:
    """Persist stage output to a Parquet file if possible, otherwise as JSON.

    Args:
        stage_name: Identifier for the pipeline stage.
        data: Data to be saved; should be convertible to a pandas DataFrame.
    """
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        try:
            df = pd.DataFrame(data)
        except Exception:
            # Fallback to JSON serialization
            path = CHECKPOINT_DIR / f"{stage_name}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return
    checkpoint_path = CHECKPOINT_DIR / f"{stage_name}.parquet"
    df.to_parquet(checkpoint_path, index=False)


def load_checkpoint(stage_name: str):
    """Load a checkpoint for a given stage if it exists.

    Returns the stored data as a DataFrame if a Parquet file exists,
    otherwise loads JSON content, or returns None if no checkpoint.
    """
    parquet_path = CHECKPOINT_DIR / f"{stage_name}.parquet"
    json_path = CHECKPOINT_DIR / f"{stage_name}.json"
    if parquet_path.is_file():
        return pd.read_parquet(parquet_path)
    if json_path.is_file():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None
