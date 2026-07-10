"""
Stateful Temporal Pattern Tracking

Extracts temporal behavioral features from M-PESA transaction data using
stateful rolling aggregators and pattern detection across daily, weekly, and
monthly time horizons.

Outputs ``temporal_features.csv`` with one row per customer containing
temporal pattern metrics suitable for downstream AML anomaly detection.

Reference: obsidian_notes/technical/TEMPORAL_PATTERNS.md
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import polars as pl

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRANSACTION_PATH = PROJECT_ROOT / "data" / "detailed_transactions.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "temporal_features.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Daily pattern windows (hour ranges)
DAILY_PATTERNS: dict[str, tuple[int, int]] = {
    "morning_commute": (6, 9),
    "lunch": (12, 14),
    "evening": (17, 20),
}

THIRTY_DAYS = timedelta(days=30)
SEVEN_DAYS = timedelta(days=7)
ONE_DAY = timedelta(days=1)

# Month-end window (calendar day 25-31)
MONTH_END_START = 25
MONTH_END_END = 31

# Salary / B2C window
SALARY_START = 25
SALARY_END = 28

# School fees months (January, May, September)
SCHOOL_FEE_MONTHS = {1, 5, 9}

# ---------------------------------------------------------------------------
# Rolling buffer
# ---------------------------------------------------------------------------


@dataclass
class TransactionRecord:
    """A single transaction record stored in a customer's temporal buffer."""

    timestamp: datetime
    amount: float
    transaction_type: str
    direction: str


class RollingBuffer:
    """Sliding-window transaction buffer for a single customer.

    Maintains per-customer transaction history with automatic eviction of
    records older than *window_size*.  Supports efficient querying for
    velocity counts and volume aggregations over sub-windows.
    """

    def __init__(self, window_size: timedelta = THIRTY_DAYS):
        self.window_size = window_size
        self._records: list[TransactionRecord] = []

    def push(self, record: TransactionRecord) -> None:
        """Append a record and evict stale entries."""
        self._records.append(record)
        self._evict(record.timestamp)

    def _evict(self, reference_ts: datetime) -> None:
        cutoff = reference_ts - self.window_size
        self._records = [r for r in self._records if r.timestamp >= cutoff]

    def _within(self, lookback: timedelta, reference_ts: datetime) -> list[TransactionRecord]:
        cutoff = reference_ts - lookback
        return [r for r in self._records if r.timestamp >= cutoff]

    def count_1h(self, reference_ts: datetime) -> int:
        """Number of transactions in the last 1 hour."""
        return len(self._within(timedelta(hours=1), reference_ts))

    def count_24h(self, reference_ts: datetime) -> int:
        """Number of transactions in the last 24 hours."""
        return len(self._within(timedelta(hours=24), reference_ts))

    def volume_24h(self, reference_ts: datetime) -> float:
        """Total transaction amount in the last 24 hours."""
        return sum(r.amount for r in self._within(timedelta(hours=24), reference_ts))

    def avg_7d(self, reference_ts: datetime) -> float:
        """Average transaction value over the last 7 days."""
        recent = self._within(SEVEN_DAYS, reference_ts)
        return sum(r.amount for r in recent) / max(len(recent), 1)

    def volume_30d(self, reference_ts: datetime) -> float:
        """Total transaction volume over the last 30 days."""
        return sum(r.amount for r in self._records)

    def count_30d(self, reference_ts: datetime) -> int:
        """Number of transactions in the last 30 days."""
        recent = self._within(THIRTY_DAYS, reference_ts)
        return len(recent)

    def clear(self) -> None:
        self._records.clear()

    @property
    def size(self) -> int:
        return len(self._records)

    def _all_within(self, lookback: timedelta, reference_ts: datetime) -> list[TransactionRecord]:
        return self._within(lookback, reference_ts)


# ---------------------------------------------------------------------------
# Temporal pattern tracker
# ---------------------------------------------------------------------------


class TemporalPatternTracker:
    """Computes per-customer temporal features from transaction data.

    For each customer, tracks:

    - **Daily patterns**: dominant transaction hour, whether > 70 % of
      transactions fall into a consistent daily window (morning/lunch/evening).
    - **Weekly patterns**: weekend transaction ratio.
    - **Monthly patterns**: month-end activity, salary receipt behaviour, school
      fee payment waves.
    - **Rolling aggregators**: 1 h / 24 h / 7 d / 30 d stats.
    - **Anomaly flags**: high weekend activity for normally-inactive users,
      unusual hourly concentration shifts.
    """

    def __init__(self) -> None:
        self._buffers: dict[str, RollingBuffer] = defaultdict(
            lambda: RollingBuffer(THIRTY_DAYS)
        )

    def _is_salary_receipt(self, row: dict[str, Any]) -> bool:
        """Check if transaction looks like a salary / B2C payment."""
        return (
            row["transaction_type"] in ("Received Money", "Lipa Na M-PESA (Paybill)")
            and 1 <= row.get("month", row["timestamp"].month) <= 12
        )

    def compute_features(self, transactions_df: pl.DataFrame) -> pl.DataFrame:
        """Compute temporal features for all customers.

        Parameters
        ----------
        transactions_df : pl.DataFrame
            Transaction data with columns: customer_id, timestamp, amount,
            transaction_type, direction, hour, day_of_week, is_weekend, month.

        Returns
        -------
        pl.DataFrame
            One row per customer with temporal feature columns.
        """
        # Ensure timestamp is datetime
        df = transactions_df.with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
        ).sort("timestamp")

        customer_ids = df["customer_id"].unique().to_list()
        features: list[dict[str, Any]] = []

        for cid in customer_ids:
            cdf = df.filter(pl.col("customer_id") == cid).sort("timestamp")
            rows = cdf.iter_rows(named=True)
            feat = self._compute_for_customer(cid, list(rows), cdf)
            features.append(feat)

        result = pl.DataFrame(features)
        # Fill nulls that may arise from single-transaction customers
        fill_cols = [c for c in result.columns if c not in ("user_id",)]
        for col in fill_cols:
            if result[col].dtype in (pl.Float64,):
                result = result.with_columns(pl.col(col).fill_null(0.0))
            elif result[col].dtype in (pl.Int64, pl.UInt32):
                result = result.with_columns(pl.col(col).fill_null(0))

        logger.info(
            "Computed temporal features for %d customers (%d columns)",
            len(result),
            len(result.columns),
        )
        return result

    def _compute_for_customer(
        self,
        customer_id: str,
        rows: list[dict[str, Any]],
        cdf: pl.DataFrame,
    ) -> dict[str, Any]:
        """Compute all temporal features for a single customer."""
        buffer = self._buffers[customer_id]
        buffer.clear()

        # Stateful accumulators
        hour_counts: dict[int, int] = defaultdict(int)
        weekend_count = 0
        weekday_count = 0
        month_end_tx_count = 0
        salary_tx_count = 0
        school_fee_amount = 0.0
        total_amount = 0.0
        total_count = len(rows)

        # Rolling aggregator snapshots (last valid values)
        last_1h_count = 0
        last_24h_count = 0
        last_24h_volume = 0.0
        last_7d_avg = 0.0
        last_30d_volume = 0.0
        last_30d_count = 0

        for row in rows:
            ts = row["timestamp"]
            amount = row["amount"]
            ttype = row["transaction_type"]
            direction = row["direction"]

            record = TransactionRecord(
                timestamp=ts, amount=amount,
                transaction_type=ttype, direction=direction,
            )
            buffer.push(record)

            # Hour bucket
            hour = int(row.get("hour", ts.hour))
            hour_counts[hour] += 1

            # Weekend / weekday
            is_we = bool(row.get("is_weekend", ts.weekday() >= 5))
            if is_we:
                weekend_count += 1
            else:
                weekday_count += 1

            # Month-end activity (calendar day 25-31)
            day = ts.day
            if MONTH_END_START <= day <= MONTH_END_END:
                month_end_tx_count += 1

            # Salary receipt (B2C near 25th-28th)
            if SALARY_START <= day <= SALARY_END and self._is_salary_receipt(row):
                salary_tx_count += 1

            # School fees (January, May, September)
            if ts.month in SCHOOL_FEE_MONTHS and ttype in (
                "Send Money", "Lipa Na M-PESA (Paybill)"
            ):
                school_fee_amount += amount

            total_amount += amount

            # Rolling aggregators — snapshot at each step but we keep last
            last_1h_count = buffer.count_1h(ts)
            last_24h_count = buffer.count_24h(ts)
            last_24h_volume = buffer.volume_24h(ts)
            last_7d_avg = buffer.avg_7d(ts)
            last_30d_volume = buffer.volume_30d(ts)
            last_30d_count = buffer.count_30d(ts)

        # --- Aggregated metrics ---

        # Daily pattern detection
        dominant_hour = max(hour_counts, key=hour_counts.get) if hour_counts else -1
        dominant_hour_pct = (
            hour_counts[dominant_hour] / max(total_count, 1) * 100 if dominant_hour >= 0 else 0.0
        )
        has_consistent_pattern = dominant_hour_pct > 70.0

        # Check pattern windows
        morning_tx = sum(
            count for h, count in hour_counts.items()
            if DAILY_PATTERNS["morning_commute"][0] <= h < DAILY_PATTERNS["morning_commute"][1]
        )
        lunch_tx = sum(
            count for h, count in hour_counts.items()
            if DAILY_PATTERNS["lunch"][0] <= h < DAILY_PATTERNS["lunch"][1]
        )
        evening_tx = sum(
            count for h, count in hour_counts.items()
            if DAILY_PATTERNS["evening"][0] <= h < DAILY_PATTERNS["evening"][1]
        )
        morning_pct = morning_tx / max(total_count, 1) * 100
        lunch_pct = lunch_tx / max(total_count, 1) * 100
        evening_pct = evening_tx / max(total_count, 1) * 100

        # Weekly pattern
        weekend_ratio = weekend_count / max(total_count, 1)

        # Detect anomaly: user with low overall weekend activity but a spike
        weekend_anomaly = bool(
            weekend_ratio > 0.5 and (weekday_count / max(total_count, 1)) < 0.3
        )

        # Monthly pattern
        month_end_ratio = month_end_tx_count / max(total_count, 1)
        salary_receipt_ratio = salary_tx_count / max(total_count, 1)
        avg_tx_value = total_amount / max(total_count, 1)
        school_fee_flag = school_fee_amount > 0.0

        # Hourly entropy (measure of distribution spread)
        if total_count > 0:
            probs = [count / total_count for count in hour_counts.values()]
            entropy = -sum(p * __import__("math").log2(p) for p in probs if p > 0)
        else:
            entropy = 0.0

        feature_row: dict[str, Any] = {
            "user_id": customer_id,
            # --- Daily patterns ---
            "dominant_hour": dominant_hour,
            "dominant_hour_pct": round(dominant_hour_pct, 4),
            "has_consistent_pattern": has_consistent_pattern,
            "morning_pct": round(morning_pct, 4),
            "lunch_pct": round(lunch_pct, 4),
            "evening_pct": round(evening_pct, 4),
            "hourly_entropy": round(entropy, 4),
            # --- Weekly patterns ---
            "weekend_ratio": round(weekend_ratio, 4),
            "weekend_tx_count": weekend_count,
            "weekday_tx_count": weekday_count,
            "weekend_anomaly": weekend_anomaly,
            # --- Monthly patterns ---
            "month_end_tx_ratio": round(month_end_ratio, 4),
            "salary_receipt_ratio": round(salary_receipt_ratio, 4),
            "school_fee_total": round(school_fee_amount, 2),
            "school_fee_flag": school_fee_flag,
            # --- Rolling aggregators (last state) ---
            "roll_1h_count": last_1h_count,
            "roll_24h_count": last_24h_count,
            "roll_24h_volume": round(last_24h_volume, 2),
            "roll_7d_avg_value": round(last_7d_avg, 2),
            "roll_30d_volume": round(last_30d_volume, 2),
            "roll_30d_count": last_30d_count,
            # --- Summary ---
            "total_tx_count": total_count,
            "avg_tx_value": round(avg_tx_value, 2),
        }
        return feature_row

    def get_buffer_state(self, customer_id: str) -> RollingBuffer:
        """Expose a customer's rolling buffer for inspection / real-time use."""
        return self._buffers[customer_id]

    def reset_buffers(self) -> None:
        """Clear all rolling buffers (e.g. between runs)."""
        self._buffers.clear()


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


def compute_temporal_features(
    input_path: Path = TRANSACTION_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pl.DataFrame:
    """Load transactions, compute temporal features, save to CSV, return DataFrame."""
    logger.info("Loading transactions from %s", input_path)

    dtype_overrides: dict[str, Any] = {
        "is_betting": pl.Boolean,
        "is_international": pl.Boolean,
        "is_weekend": pl.Boolean,
        "is_night": pl.Boolean,
        "is_kadogo": pl.Boolean,
    }
    df = pl.read_csv(input_path, try_parse_dates=True, schema_overrides=dtype_overrides)

    tracker = TemporalPatternTracker()
    features = tracker.compute_features(df)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.write_csv(output_path)
    logger.info("Temporal features saved to %s (%d rows)", output_path, len(features))
    return features


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    compute_temporal_features()


if __name__ == "__main__":
    main()
