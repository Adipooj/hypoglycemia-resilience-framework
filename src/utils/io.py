from pathlib import Path

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:

    return pd.read_csv(
        path,
        encoding="utf-8",
        low_memory=False,
    )