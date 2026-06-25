import pandas as pd
import numpy as np


def compute_recovery_metrics(df: pd.DataFrame) -> dict:
    """Calculate recovery‑related glycemic metrics.

    - **Time‑to‑Recovery (TTR)**: minutes from a hypoglycemic event (glucose < 70) to the first crossing of the goal range (70‑180).
    - **Recovery Slope**: average rate of glucose increase (mg/dL per minute) during the recovery period.
    - **Post‑hypoglycemia rebound**: average glucose level 30 min after recovery.
    """
    if df.empty:
        return {}
    # Ensure proper datetime handling
    if 'timestamp' not in df.columns:
        return {}
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    df.set_index('timestamp', inplace=True)
    # Resample to 5‑minute bins (common CGM resolution)
    glucose = df['glucose'].astype(float).resample('5T').ffill()
    # Identify hypoglycemic events (start of a contiguous block below 70)
    below = glucose < 70
    events = []
    in_event = False
    start = None
    for t, val in below.iteritems():
        if val and not in_event:
            in_event = True
            start = t
        elif not val and in_event:
            # End of hypoglycemia, find recovery point
            in_event = False
            # Search forward for first time glucose >= 70
            recovery_series = glucose[t:]
            recovery_point = recovery_series[recovery_series >= 70].first_valid_index()
            if recovery_point is None:
                continue
            ttr = (recovery_point - start).total_seconds() / 60.0
            # Recovery slope using linear fit over the recovery interval
            window = glucose[start:recovery_point]
            if len(window) >= 2:
                # minutes since start
                minutes = np.arange(len(window)) * 5
                slope = np.polyfit(minutes, window.values, 1)[0]
            else:
                slope = np.nan
            # Post‑hypoglycemia rebound: mean glucose 30 min after recovery
            after = glucose[recovery_point: recovery_point + pd.Timedelta(minutes=30)]
            rebound = after.mean() if not after.empty else np.nan
            events.append({
                'event_start': start,
                'recovery_time_min': ttr,
                'recovery_slope_mgdl_per_min': slope,
                'post_hypo_rebound': rebound,
            })
    # Aggregate across events
    if not events:
        return {}
    df_events = pd.DataFrame(events)
    return {
        'avg_TTR_min': df_events['recovery_time_min'].mean(),
        'avg_recovery_slope': df_events['recovery_slope_mgdl_per_min'].mean(),
        'avg_rebound': df_events['post_hypo_rebound'].mean(),
        'event_count': len(events),
    }
