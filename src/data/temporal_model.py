import numpy as np
from datetime import datetime, timezone

def generate_fy25_timestamps(n_records: int, inter_arrival_mins: np.ndarray) -> np.ndarray:
    """
    Generates vectorized timestamps strictly within the FY25 boundary (2024-07-01 to 2025-06-30).
    Uses cumulative sums of inter-arrival times and wraps them around the FY25 duration.
    """
    fy25_start = np.datetime64('2024-07-01T00:00:00')
    seconds_in_fy25 = 365 * 24 * 60 * 60
    
    cum_seconds = np.array([int(m * 60) for m in np.cumsum(inter_arrival_mins)])
    cum_seconds_wrapped = cum_seconds % seconds_in_fy25
    
    timestamps = fy25_start + np.array(
        [np.timedelta64(s, 's') for s in cum_seconds_wrapped]
    )
    return timestamps
