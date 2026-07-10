"""
Temporal Model for M-PESA Transaction Generation

Implements inhomogeneous Poisson processes with:
- 168-hour weekly intensity vector (hour x day-of-week)
- Monthly seasonality (school fees, holiday surges)
- End-of-month surges
- Weekend reduction factors

Used by BehavioralTransactionGenerator for timestamp generation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

FY25_START = datetime(2024, 7, 1, tzinfo=timezone.utc)
FY25_END = datetime(2025, 6, 30, 23, 59, 59, tzinfo=timezone.utc)

# 168-hour weekly intensity vector indexed by (dow * 24 + hour)
# Values represent relative tx intensity (0 = none, 1 = peak weekday)
WEEKLY_INTENSITY: list[float] = [
    # Monday (dow=0)
    0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.40, 0.40, 0.40, 0.85, 0.85, 0.85,
    0.85, 0.75, 0.75, 0.75, 1.00, 1.00, 1.00, 1.00, 0.50, 0.50, 0.50, 0.50,
    # Tuesday (dow=1)
    0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.35, 0.35, 0.35, 0.80, 0.80, 0.80,
    0.80, 0.70, 0.70, 0.70, 0.95, 0.95, 0.95, 0.95, 0.45, 0.45, 0.45, 0.45,
    # Wednesday (dow=2)
    0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.35, 0.35, 0.35, 0.80, 0.80, 0.80,
    0.80, 0.70, 0.70, 0.70, 0.95, 0.95, 0.95, 0.95, 0.45, 0.45, 0.45, 0.45,
    # Thursday (dow=3)
    0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.35, 0.35, 0.35, 0.80, 0.80, 0.80,
    0.80, 0.70, 0.70, 0.70, 0.95, 0.95, 0.95, 0.95, 0.50, 0.50, 0.50, 0.50,
    # Friday (dow=4)
    0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.35, 0.35, 0.35, 0.75, 0.75, 0.75,
    0.75, 0.65, 0.65, 0.65, 0.90, 0.90, 0.90, 0.90, 0.60, 0.60, 0.60, 0.60,
    # Saturday (dow=5)
    0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.15, 0.15, 0.15, 0.50, 0.50, 0.50,
    0.50, 0.55, 0.55, 0.55, 0.50, 0.50, 0.50, 0.50, 0.35, 0.35, 0.35, 0.35,
    # Sunday (dow=6)
    0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.30, 0.30, 0.30,
    0.30, 0.35, 0.35, 0.35, 0.30, 0.30, 0.30, 0.30, 0.15, 0.15, 0.15, 0.15,
]

MONTHLY_FACTORS: list[float] = [
    0.95,  # Jul
    0.90,  # Aug
    0.95,  # Sep
    1.00,  # Oct
    1.05,  # Nov
    1.20,  # Dec
    1.10,  # Jan
    0.90,  # Feb
    0.95,  # Mar
    0.85,  # Apr
    0.90,  # May
    0.95,  # Jun
]

SCHOOL_FEES_MONTHS = {1, 5, 9}
HOLIDAY_MONTH = 12


def compute_intensity(dt: datetime) -> float:
    """Compute λ(t) multiplier: f_hour_dow * f_month * f_eom * f_school * f_holiday."""
    hour = dt.hour
    dow = dt.weekday()
    month_idx = dt.month - 7
    if month_idx < 0:
        month_idx += 12
    day = dt.day

    f_hour_dow = WEEKLY_INTENSITY[dow * 24 + hour]
    f_month = MONTHLY_FACTORS[month_idx]
    f_eom = 2.0 if 25 <= day <= 30 else 1.0
    f_school = 3.0 if dt.month in SCHOOL_FEES_MONTHS else 1.0
    f_holiday = 1.5 if dt.month == HOLIDAY_MONTH else 1.0

    return f_hour_dow * f_month * f_eom * f_school * f_holiday


TRUE_MAX_MULTIPLIER = 1.0 * 1.20 * 2.0 * 3.0 * 1.5  # = 10.8
MEAN_INTENSITY_ESTIMATE = 0.35
