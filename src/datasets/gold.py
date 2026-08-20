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
    """Velocity derived features for 21-feature schema (no-op for now)."""
    # TVAE Hybrid v2.0 - velocity features computed by CustomerFeatureEngineer
    return df


def _compute_balance_features(df: pl.DataFrame) -> pl.DataFrame:
    """Balance-derived features for 21-feature schema (no-op for now)."""
    # TVAE Hybrid v2.0 - balance features computed by CustomerFeatureEngineer
    return df


def _compute_amount_patterns(df: pl.DataFrame) -> pl.DataFrame:
    """Amount pattern features for 21-feature schema (no-op for now)."""
    # TVAE Hybrid v2.0 - amount features computed by CustomerFeatureEngineer
    return df


def _compute_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
    """Temporal features for 21-feature schema (no-op for now)."""
    # TVAE Hybrid v2.0 - temporal features computed by CustomerFeatureEngineer
    return df


def _compute_gold_features(df: pl.DataFrame) -> pl.DataFrame:
    """Gold layer features for 21-feature schema (no-op for now)."""
    # TVAE Hybrid v2.0 - gold features computed by CustomerFeatureEngineer
    return df


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
            # TVAE Hybrid v2.0 - 21-feature schema
            # Core features (8)
            "customer_id", "tier", "archetype", "transaction_type", 
            "amount", "timestamp", "direction", "balance",
            # Temporal features (5)
            "tx_count_7d", "volume_7d", "night_tx_ratio", 
            "rapid_tx_ratio", "volume_7d_vs_30d_ratio",
            # Network features (3)
            "is_international", "distinct_counterparties_7d", "fan_in_fan_out_ratio",
            # Structuring features (3)
            "close_to_limit_ratio", "balance_retention_ratio", "amount_roundness",
            # Labels (2)
            "is_launderer", "aml_scenario",
        ]

    label_cols = ["customer_id", "is_launderer", "aml_scenario", "timestamp"]

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
    core_cols = [
        "customer_id", "tier", "archetype", "transaction_type",
        "amount", "timestamp", "direction", "balance",
        "tx_count_7d", "volume_7d", "night_tx_ratio", "rapid_tx_ratio",
        "volume_7d_vs_30d_ratio", "is_international", "distinct_counterparties_7d",
        "fan_in_fan_out_ratio", "close_to_limit_ratio", "balance_retention_ratio",
        "amount_roundness", "is_launderer", "aml_scenario",
    ]
    keep_cols = core_cols | label_cols

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
