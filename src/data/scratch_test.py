import numpy as np
import polars as pl
import time

n_records = 500000
inter_arrival_mins = np.random.exponential(10, n_records)

fy25_start = np.datetime64('2024-07-01T00:00:00')
seconds_in_fy25 = 365 * 24 * 60 * 60

cum_seconds = np.array([int(m * 60) for m in np.cumsum(inter_arrival_mins)])
cum_seconds_wrapped = cum_seconds % seconds_in_fy25

timestamps = fy25_start + np.array(
    [np.timedelta64(s, 's') for s in cum_seconds_wrapped]
)

start = time.time()
timestamps_pl = pl.Series("timestamp", timestamps).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
print(timestamps_pl.head())
print("Time taken:", time.time() - start)
