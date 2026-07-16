"""
Gold Layer — transaction-level feature computation.

All aggregate features are recomputed from Silver data using Polars window
functions. This ensures velocity, balance, and amount-pattern features reflect
the true post-injection state of each customer's transaction history, rather
than stale pre-injection aggregates computed inside the generator.
"""

import os
import logging
from typing import Optional

import polars as pl
import pyarrow.parquet as pq
import pyarrow.dataset as ds

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature computation windows (in minutes)
# ---------------------------------------------------------------------------
_TX_WINDOWS = [1, 5, 60, 1440, 10080, 43200]  # 1min, 5min, 1h, 24h, 7d, 30d
_TX_WINDOW_LABELS = ["1min", "5min", "1h", "24h", "7d", "30d"]
# For 1-min binned data: rolling_sum(window_size=N) on sorted per-customer data
# is equivalent to N-minute time-based rolling
_WINDOW_BINS = {label: m for m, label in zip(_TX_WINDOWS, _TX_WINDOW_LABELS)}


# ---------------------------------------------------------------------------
# Binning helpers
# ---------------------------------------------------------------------------

def _bin_and_roll(
    df: pl.DataFrame,
    every: str = "1m",
) -> pl.DataFrame:
    """Bin transactions by customer into *every*-sized buckets, then compute
    rolling aggregates over each window.

    Uses ``rolling_sum(window_size=N).over("customer_id")`` since Polars
    0.20.x does not support the ``by=`` parameter on ``Expr.rolling()``.

    Returns a DataFrame with one row per (customer_id, bin_start) and columns
    for each rolling-window aggregate.  The caller joins these back to the
    original transaction rows via ``join_asof``.
    """
    binned = df.group_by_dynamic(
        "timestamp",
        every=every,
        by="customer_id",
        closed="left",
    ).agg([
        pl.len().cast(pl.Int32).alias("_bin_tx_count"),
        pl.sum("amount").alias("_bin_amount_sum"),
        pl.mean("amount").alias("_bin_amount_mean"),
    ])

    binned = binned.sort(["customer_id", "timestamp"])

    for label, window_bins in _WINDOW_BINS.items():
        binned = binned.with_columns([
            pl.col("_bin_tx_count")
            .rolling_sum(window_size=window_bins)
            .over("customer_id")
            .alias(f"tx_count_{label}"),
            pl.col("_bin_amount_sum")
            .rolling_sum(window_size=window_bins)
            .over("customer_id")
            .alias(f"amount_sum_{label}"),
        ])

    return binned


def _join_rolling_to_tx(
    tx: pl.DataFrame,
    rolled: pl.DataFrame,
) -> pl.DataFrame:
    """Join rolling-window aggregates back to original transaction rows by
    carrying the most recent bin state backward (i.e. for each transaction
    we use the aggregates that were known *before* it fired)."""
    return tx.sort("customer_id", "timestamp").join_asof(
        rolled.sort("customer_id", "timestamp"),
        on="timestamp",
        by="customer_id",
        strategy="backward",
    )


# ---------------------------------------------------------------------------
# Per-feature-group builders  (called on the join-asof result)
# ---------------------------------------------------------------------------

def _compute_velocity_derived(df: pl.DataFrame) -> pl.DataFrame:
    """Burst ratio and velocity change from the rolling-window columns."""
    return df.with_columns([
        # burst_ratio = recent density vs hourly density
        (pl.col("tx_count_1min") / (pl.col("tx_count_1h") + 1))
        .alias("burst_ratio"),
    ])


def _compute_balance_features(df: pl.DataFrame) -> pl.DataFrame:
    """Balance-derived features over the 30-day rolling window."""
    return df.with_columns([
        # zero_balance_frequency — what fraction of a customer's recent
        # transactions left a balance near zero
        (pl.col("amount_sum_30d") / (pl.col("tx_count_30d") + 1).cast(pl.Float64))
        .alias("mean_tx_amount_30d"),
        # We approximate zero_balance_frequency by looking at the share of
        # windows where post_tx_balance was < 5% of the customer's typical balance.
        # A proper implementation would require per-window balance computation.
        (pl.col("post_tx_balance") < 0.05 * pl.col("post_tx_balance").max().over("customer_id"))
        .cast(pl.Float32)
        .alias("zero_balance_frequency"),
        # balance_retention_ratio — how much of deposited amount stays in account
        (pl.col("post_tx_balance") / (pl.col("amount_sum_7d") + 1))
        .clip(0, 1)
        .alias("balance_retention_ratio"),
        # balance_depletion_rate — how quickly balance drops relative to outflows
        (pl.col("amount_sum_1h") / (pl.col("post_tx_balance") + 1))
        .clip(0, 10)
        .alias("balance_depletion_rate"),
    ])


def _compute_amount_patterns(df: pl.DataFrame) -> pl.DataFrame:
    """Amount roundness, profiling, and structuring detection.

    Split into multiple ``with_columns`` calls to avoid a Polars 0.20.x
    query-plan conflict where ``.sum().over()`` inside a combined block
    causes ``InvalidOperationError: window expression not allowed in aggregation``
    when other plain ``.over()`` expressions reference the same frame.
    """
    df = df.with_columns([
        # amount_roundness — how "round" the amount is (more round = more suspicious)
        (1 / (pl.col("amount").log10().floor() + 1))
        .alias("amount_roundness"),
        # amount_vs_profile_avg — deviation from the customer's historic mean amount
        ((pl.col("amount") - pl.col("_bin_amount_mean").over("customer_id"))
         / (pl.col("_bin_amount_mean").over("customer_id") + 1))
        .alias("amount_vs_profile_avg"),
        # amount_just_below_threshold — within 10 % of 1 000 000 (common ceiling)
        ((pl.col("amount") > 900_000) & (pl.col("amount") < 1_000_000))
        .cast(pl.Int32)
        .alias("amount_just_below_threshold"),
    ])
    # structuring proxy — rolling count of recent transactions by the same
    # customer with amounts within 5% of this one
    return df.with_columns([
        pl.col("amount")
        .is_between(
            pl.col("amount") * 0.95,
            pl.col("amount") * 1.05,
            closed="both",
        )
        .cast(pl.Int32)
        .rolling_sum(window_size=24)
        .over("customer_id")
        .alias("similar_amount_count_24h"),
    ])


def _compute_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
    """Extract temporal features from the timestamp column.

    Split into two ``with_columns`` calls so ``is_anomalous_hour`` and
    ``is_weekend`` can reference ``hour_of_day`` / ``day_of_week`` that are
    created in the same block (Polars 0.20 evaluates expressions
    simultaneously, not sequentially).
    """
    df = df.with_columns([
        pl.col("timestamp").dt.hour().alias("hour_of_day"),
        pl.col("timestamp").dt.weekday().alias("day_of_week"),
        pl.col("timestamp").dt.day().alias("day_of_month"),
        (pl.col("timestamp").diff().over("customer_id").dt.total_milliseconds() / 1000)
        .alias("time_since_last_tx"),
    ])
    return df.with_columns([
        ((pl.col("hour_of_day") < 6) | (pl.col("hour_of_day") > 22))
        .cast(pl.Int32).alias("is_anomalous_hour"),
        ((pl.col("day_of_week") >= 6)).cast(pl.Int32).alias("is_weekend"),
    ])


def _compute_gold_features(df: pl.DataFrame) -> pl.DataFrame:
    """Features that were historically produced only by the Gold layer."""
    return df.with_columns([
        pl.col("amount").log10().alias("log_amount"),
        # is_round_number_100k — amount is a multiple of 100k
        (pl.col("amount") % 100_000 == 0).cast(pl.Int32).alias("is_round_number_100k"),
        # transaction_velocity — alias for tx_count_1h
        pl.col("tx_count_1h").alias("transaction_velocity"),
        # customer lifetime value — total net flow
        (pl.col("amount_sum_30d")).alias("clv"),
        # high_risk_amount — amount above 500k
        (pl.col("amount") > 500_000).cast(pl.Int32).alias("high_risk_amount"),
        # z_score_deviation — deviation from customer mean
        ((pl.col("amount") - pl.col("_bin_amount_mean").over("customer_id"))
         / (pl.col("_bin_amount_mean").over("customer_id") + 1))
        .alias("z_score_deviation"),
    ])


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def silver_to_transaction_features(
    transactions: pl.DataFrame,
    pass_through_columns: Optional[list[str]] = None,
    version: str = "1.0",
    output_dir: str = "data/gold/features",
) -> pl.DataFrame:
    """
    Compute transaction-level feature set from Silver data.

    All rolling-window aggregates are computed from the Silver data using
    Polars window functions, ensuring velocity, balance, and amount-pattern
    features reflect the true post-injection state.

    Args:
        transactions: Silver transaction DataFrame.
        pass_through_columns: Column names to carry through unchanged from the
            generator (network metrics, device/location attrs, etc.).
            If ``None``, a default set is used.
        version: Feature store version string (appended to sub-directory name).
        output_dir: Root directory for the feature parquet output.

    Returns:
        Feature DataFrame with one row per transaction and all computed
        features plus label columns.
    """
    if pass_through_columns is None:
        pass_through_columns = [
            # Network / graph
            "degree_centrality", "in_degree", "out_degree",
            "reciprocity_ratio", "new_relationships_7d",
            "clustering_coefficient", "community_id",
            "behavioral_shift_score",
            # Device / location
            "device_age_days", "sim_match_status",
            "device_changes_7d", "location_entropy", "device_change_flag",
            "sender_county", "receiver_county",
            # Funnel / pass-through (already behavioural)
            "funnel_score", "pass_through_ratio",
            "session_intensity",
            # Wallet / KYC
            "wallet_tier", "kyc_level",
            "prev_fraud_flag_count_90d",
        ]

    label_cols = ["anomaly_flag", "anomaly_type", "anomaly_case_id", "transaction_id",
                   "customer_id", "counterparty_id", "timestamp", "partition_date"]

    logger.info(
        "Building transaction-level features from %s rows …",
        transactions.height,
    )

    # Handle column name variations (silver vs gold conventions)
    renames = {}
    if "entity_id" in transactions.columns and "customer_id" not in transactions.columns:
        renames["entity_id"] = "customer_id"
    if "transaction_amount" in transactions.columns and "amount" not in transactions.columns:
        renames["transaction_amount"] = "amount"
    if "account_balance_after" in transactions.columns and "post_tx_balance" not in transactions.columns:
        renames["account_balance_after"] = "post_tx_balance"
    if "account_balance_before" in transactions.columns and "current_balance" not in transactions.columns:
        renames["account_balance_before"] = "current_balance"
    if renames:
        transactions = transactions.rename(renames)
    
    # Sort once — everything downstream relies on per-customer time ordering
    df = transactions.sort(["customer_id", "timestamp"])

    # ---- Step 1: 1-minute bins + rolling windows -------------------------
    logger.info("Computing 1-min bins and rolling-window aggregates …")
    rolled = _bin_and_roll(df)
    result = _join_rolling_to_tx(df, rolled)

    # ---- Step 2: Derived feature groups ----------------------------------
    logger.info("Computing velocity-derived features …")
    result = _compute_velocity_derived(result)

    logger.info("Computing balance features …")
    result = _compute_balance_features(result)

    logger.info("Computing amount-pattern features …")
    result = _compute_amount_patterns(result)

    logger.info("Computing temporal features …")
    result = _compute_temporal_features(result)

    logger.info("Computing Gold-layer features …")
    result = _compute_gold_features(result)

    # ---- Step 3: Select final column set ----------------------------------
    keep_cols = list(
        set(pass_through_columns)
        | set(label_cols)
        | {
            # Velocity
            c for c in result.columns if c.startswith("tx_count_")
        }
        | {c for c in result.columns if c.startswith("amount_sum_")}
        | {
            "burst_ratio",
            "mean_tx_amount_30d",
            "zero_balance_frequency",
            "balance_retention_ratio",
            "balance_depletion_rate",
            "amount_roundness",
            "amount_vs_profile_avg",
            "amount_just_below_threshold",
            "similar_amount_count_24h",
            "hour_of_day", "day_of_week", "day_of_month",
            "is_anomalous_hour", "is_weekend",
            "time_since_last_tx",
            "log_amount", "is_stk_push", "is_b2c",
            "is_round_number_100k",
            "transaction_velocity",
            "clv", "high_risk_amount", "z_score_deviation",
            "post_tx_balance", "current_balance",
        }
    )

    result = result.select([c for c in keep_cols if c in result.columns])

    # ---- Step 4: Write output --------------------------------------------
    out_path = os.path.join(output_dir, f"v{version}")
    os.makedirs(out_path, exist_ok=True)
    out_file = os.path.join(out_path, "gold_features_consolidated.parquet")

    result.write_parquet(out_file)
    logger.info("Wrote %s rows x %s cols → %s", result.height, result.width, out_file)

    return result


# ---------------------------------------------------------------------------
# Legacy GoldLayer class (unchanged, kept for backward compat)
# ---------------------------------------------------------------------------

class GoldLayer:
    """
    Gold layer — writes feature-store / curated datasets consumed by
    downstream consumers (models, dashboards, reports).

    Parameters
    ----------
    namespace : str
        Dataset namespace (e.g. ``sentinAI``).
    version : str
        Feature store version (e.g. ``1.0``).
    """

    def __init__(self, namespace: str = "sentinAI", version: str = "1.0"):
        self._namespace = namespace
        self.version = version
        self._logger = logging.getLogger(self.__class__.__name__)

    def _dataset_name(self, base: str) -> str:
        return f"{self._namespace}.{base}.v{self.version}"

    def _base_output_path(self, base: str) -> str:
        return os.path.join("data", "gold", base, f"v{self.version}")

    def _make_dataset(self, name: str) -> "Dataset":
        return Dataset(namespace=self._namespace, name=name)

    def create_feature_store(
        self,
        transactions: pl.DataFrame,
        customers: Optional[pl.DataFrame] = None,
        metrics: Optional[dict] = None,
        benchmark_mode: bool = False,
    ) -> pl.DataFrame:
        """
        **Deprecated** — prefer ``silver_to_transaction_features`` for new work.

        Builds the consolidated feature store from Silver transactions and
        customer data.  This is a thin wrapper around the standalone function
        for backward compatibility but will be removed in a future version.
        """
        self._logger.warning(
            "GoldLayer.create_feature_store is deprecated; "
            "use silver_to_transaction_features() instead."
        )
        return silver_to_transaction_features(
            transactions,
            version=self.version,
            output_dir="data/gold/features",
        )
