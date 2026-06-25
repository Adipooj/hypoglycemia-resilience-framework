import logging
import pandas as pd

logger = logging.getLogger("hypo_resilience.preprocessing.datetime_utils")


def parse_datetime(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Parses a column to datetime, drops rows with invalid timestamps,
    sorts by timestamp, and resets the index.
    
    Args:
        df (pd.DataFrame): Dataframe containing the timestamp column.
        column (str): The column name to parse.
        
    Returns:
        pd.DataFrame: A copy of the sorted dataframe with parsed datetime.
    """
    if df is None or df.empty:
        return pd.DataFrame()
        
    df = df.copy()
    
    if column not in df.columns:
        logger.warning(f"Timestamp column {column} not found in dataframe. Returning empty.")
        return pd.DataFrame()
        
    # Convert and coerce errors
    df[column] = pd.to_datetime(df[column], errors="coerce", dayfirst=True)
    
    # Drop rows with NaT
    initial_len = len(df)
    df = df.dropna(subset=[column])
    dropped = initial_len - len(df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped} rows with invalid dates in column {column}")
        
    df = df.sort_values(column)
    df = df.reset_index(drop=True)
    return df