"""
AML Anomaly Injector for Synthetic M-PESA Data.

Injects structural anomalies into clean AMLGenerator output at a controlled
ratio (0.015 = 1.5% of total dataset). Operates directly on AMLGenerator-native
columns — no schema bridging needed.

CRITICAL: This module ONLY adds anomaly_flag and anomaly_type columns AFTER
injecting anomalies. The injector is responsible for:
1. Accepting clean data (anomaly_flag=False, anomaly_type=null)
2. Injecting anomalies at strict 0.015 ratio
3. Overwriting anomaly_flag (True for anomalous, False for clean)
4. Setting anomaly_type (POCAMLA-compatible categorical string)
"""

import logging
import numpy as np
import polars as pl
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """POCAMLA-compatible anomaly types targeting AMLGenerator-native features."""
    AMOUNT_ANOMALY = "amount_anomaly"                    # Structuring / smurfing
    VELOCITY_FUNNEL = "velocity_funnel"                  # Funnel account velocity
    MULE_ACTIVITY = "mule_activity"                      # Pass-through mule
    LAYERING = "layering"                                # Network layering hops
    CEILING_VIOLATION = "ceiling_violation"              # CBK regulatory tier breach
    HIGH_RISK_COUNTRY = "high_risk_country"              # Cross-border to risky jurisdiction
    CIRCULAR_TRADING = "circular_trading"                # Same-community circular flow
    TEMPORAL_ANOMALY = "temporal_anomaly"                # Off-hours / device churn


@dataclass
class InjectorConfig:
    """Configuration for AML-native anomaly injection.

    Each parameter controls how the injector modifies AMLGenerator columns
    during anomaly injection. Tuned for realistic M-PESA anomaly patterns.
    """
    anomaly_ratio: float = 0.015
    seed: int = 42

    # Structuring / amount anomalies
    structuring_amount_target: float = 95_000.0       # KES — just below CTR threshold
    structuring_amount_sigma: float = 5_000.0         # KES — noise around target
    structuring_roundness_threshold: float = 0.85     # High amount_roundness
    structuring_entropy_threshold: float = 0.7        # High structuring_amount_entropy

    # Velocity / funnel
    funnel_tx_count_1h: int = 18                      # tx_count_1h target
    funnel_tx_count_24h: int = 80                     # tx_count_24h target
    funnel_burst_ratio: float = 4.0                   # burst_ratio target
    funnel_score_target: float = 0.85                 # funnel_score target

    # Mule / pass-through
    mule_pass_through_ratio: float = 0.90             # pass_through_ratio
    mule_zero_balance_freq: float = 0.60              # zero_balance_frequency
    mule_retention_ratio: float = 0.10                # balance_retention_ratio (low)
    mule_depletion_rate: float = 0.85                 # balance_depletion_rate

    # Layering / network
    layering_degree_centrality: float = 0.35          # degree_centrality
    layering_reciprocity: float = 0.15                # reciprocity_ratio (low)
    layering_new_relationships: int = 7               # new_relationships_7d
    layering_behavioural_shift: float = 0.75          # behavioral_shift_score

    # Regulatory ceiling violation
    ceiling_violation_amount: float = 80_000.0        # KES — well above TIER_1 cap (10K)
    ceiling_post_tx_balance: float = 90_000.0         # KES — above TIER_1 balance cap (50K)
    ceiling_min_target_tier_encoded: int = 0          # 0 = TIER_1 (wallet_tier_encoded)

    # High risk country
    high_risk_amount: float = 150_000.0               # KES
    high_risk_country_code: str = "IR"                # High-risk jurisdiction
    counterparty_risk_flag_value: bool = True

    # Circular trading
    circular_amount_mean: float = 50_000.0            # KES
    circular_amount_sigma: float = 15_000.0
    circular_clustering: float = 0.80                 # clustering_coefficient target

    # Temporal anomaly
    temporal_anomalous_hour: int = 3                  # 3 AM — off-hours
    temporal_device_changes: int = 4                  # device_changes_7d
    temporal_location_entropy: float = 0.85           # location_entropy target

    # Equal share per anomaly type (12.5% each for 8 types → do not sum to 1.0 for compatibility)
    anomaly_type_weights: Dict[str, float] = field(default_factory=lambda: {
        "amount_anomaly": 0.20,
        "velocity_funnel": 0.15,
        "mule_activity": 0.15,
        "layering": 0.10,
        "ceiling_violation": 0.20,
        "high_risk_country": 0.10,
        "circular_trading": 0.05,
        "temporal_anomaly": 0.05,
    })


# Columns the injector may modify (all exist in AMLGenerator output)
INJECTABLE_FEATURES: List[str] = [
    "amount",
    "tx_count_1h",
    "tx_count_24h",
    "burst_ratio",
    "funnel_score",
    "pass_through_ratio",
    "zero_balance_frequency",
    "balance_retention_ratio",
    "balance_depletion_rate",
    "degree_centrality",
    "reciprocity_ratio",
    "new_relationships_7d",
    "behavioral_shift_score",
    "structuring_amount_entropy",
    "amount_roundness",
    "amount_vs_profile_avg",
    "post_tx_balance",
    "current_balance",
    "hour_of_day",
    "is_anomalous_hour",
    "device_changes_7d",
    "location_entropy",
    "clustering_coefficient",
    "community_id",
    "wallet_tier_encoded",
]

# Mapping from AnomalyType enum value to POCAMLA label
ANOMALY_TYPE_MAP: Dict[str, str] = {
    "amount_anomaly": "structuring",
    "velocity_funnel": "funnel_account",
    "mule_activity": "mule_account",
    "layering": "layering",
    "ceiling_violation": "regulatory_breach",
    "high_risk_country": "high_risk_jurisdiction",
    "circular_trading": "circular_trading",
    "temporal_anomaly": "temporal_anomaly",
}

ANOMALY_TYPE_DISTRIBUTION: Dict[AnomalyType, float] = {
    AnomalyType.AMOUNT_ANOMALY: 0.20,
    AnomalyType.VELOCITY_FUNNEL: 0.15,
    AnomalyType.MULE_ACTIVITY: 0.15,
    AnomalyType.LAYERING: 0.10,
    AnomalyType.CEILING_VIOLATION: 0.20,
    AnomalyType.HIGH_RISK_COUNTRY: 0.10,
    AnomalyType.CIRCULAR_TRADING: 0.05,
    AnomalyType.TEMPORAL_ANOMALY: 0.05,
}


class FinancialAnomalyInjector:
    """Injects AML-native anomalies into synthetic M-PESA transaction data.

    Operates directly on the AMLGenerator output schema. Accepts a clean
    DataFrame (anomaly_flag=False, anomaly_type=null), injects anomalies,
    and returns the same DataFrame with anomalies modified and labels set.

    The injector is schema-agnostic beyond INJECTABLE_FEATURES — missing
    columns are silently skipped (graceful degradation for subsets).
    """

    def __init__(self, config: Optional[InjectorConfig] = None):
        self.config = config or InjectorConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._validate_type_weights()

    def _validate_type_weights(self) -> None:
        """Warn if anomaly type weights do not sum to 1.0."""
        total = sum(self.config.anomaly_type_weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(
                "Anomaly type weights sum to %.3f (expected ~1.0).",
                total,
            )

    def inject(self, df: pl.DataFrame) -> pl.DataFrame:
        """Inject anomalies into a clean DataFrame.

        Args:
            df: Clean transaction data from AMLGenerator (66+ columns).

        Returns:
            DataFrame with anomaly_flag (bool) and anomaly_type (str) set.
        """
        result = df.with_columns([
            pl.lit(False).alias("anomaly_flag"),
            pl.lit(None, dtype=pl.Utf8).alias("anomaly_type"),
        ])

        n_total = len(result)
        n_anomalies = max(1, int(n_total * self.config.anomaly_ratio))

        logger.info(
            "Injecting %d anomalies into %d records (ratio=%.4f)",
            n_anomalies,
            n_total,
            self.config.anomaly_ratio,
        )

        # Stratified anomaly assignment
        anomaly_indices = self._rng.choice(
            n_total, size=n_anomalies, replace=False
        )
        anomaly_types = self._assign_anomaly_types(n_anomalies)

        # Group by anomaly type and inject
        type_to_indices: Dict[str, List[int]] = {}
        for idx, at in zip(anomaly_indices, anomaly_types):
            type_to_indices.setdefault(at, []).append(int(idx))

        for at_name, indices in type_to_indices.items():
            type_enum = AnomalyType(at_name)
            method = getattr(self, f"_inject_{at_name}", None)
            if method is None:
                logger.warning("No injection method for '%s', skipping.", at_name)
                continue

            pocamla_label = ANOMALY_TYPE_MAP.get(at_name, at_name)
            row_mask = pl.Series("__mask", [False] * n_total)
            for i in indices:
                row_mask[i] = True

            try:
                result = method(result, row_mask)
            except Exception:
                logger.exception("Injection failed for '%s', skipping.", at_name)
                continue

            # Set labels on injected rows (scoped to current type group only)
            result = result.with_columns([
                pl.when(row_mask)
                .then(pl.lit(pocamla_label))
                .otherwise(pl.col("anomaly_type"))
                .alias("anomaly_type"),
            ])

        logger.info(
            "Injected %d anomalies (%d types).",
            n_anomalies,
            len(type_to_indices),
        )
        return result

    def _assign_anomaly_types(self, n: int) -> List[str]:
        """Assign anomaly types to n indices using the weighted distribution."""
        types = list(self.config.anomaly_type_weights.keys())
        weights = list(self.config.anomaly_type_weights.values())
        weights = np.array(weights) / sum(weights)
        return list(self._rng.choice(types, size=n, p=weights))

    # ------------------------------------------------------------------
    # Injection methods — each targets AMLGenerator-native columns
    # ------------------------------------------------------------------

    def _inject_amount_anomaly(
        self, df: pl.DataFrame, mask: pl.Series
    ) -> pl.DataFrame:
        """Structuring: amounts just below CTR threshold with high roundness."""
        n = mask.sum()
        struct_amounts = self._rng.normal(
            self.config.structuring_amount_target,
            self.config.structuring_amount_sigma,
            size=int(n),
        ).clip(50_000, 950_000)
        N = len(df)

        amount_arr = np.full(N, np.nan, dtype=np.float64)
        amount_arr[mask.to_numpy()] = struct_amounts
        round_arr = np.full(N, np.nan, dtype=np.float64)
        round_arr[mask.to_numpy()] = self.config.structuring_roundness_threshold
        entropy_arr = np.full(N, np.nan, dtype=np.float64)
        entropy_arr[mask.to_numpy()] = self.config.structuring_entropy_threshold

        return df.with_columns([
            pl.when(mask)
            .then(pl.Series(amount_arr))
            .otherwise(pl.col("amount"))
            .alias("amount"),
            pl.when(mask)
            .then(pl.Series(round_arr))
            .otherwise(pl.col("amount_roundness"))
            .alias("amount_roundness"),
            pl.when(mask)
            .then(pl.Series(entropy_arr))
            .otherwise(pl.col("structuring_amount_entropy"))
            .alias("structuring_amount_entropy"),
            pl.when(mask)
            .then(pl.lit(True))
            .otherwise(pl.col("anomaly_flag"))
            .alias("anomaly_flag"),
        ])

    def _inject_velocity_funnel(
        self, df: pl.DataFrame, mask: pl.Series
    ) -> pl.DataFrame:
        """High velocity, bursty patterns suggesting a funnel account."""
        N = len(df)
        mask_np = mask.to_numpy()
        tx1h_arr = np.zeros(N, dtype=np.int64)
        tx1h_arr[mask_np] = self.config.funnel_tx_count_1h
        tx24h_arr = np.zeros(N, dtype=np.int64)
        tx24h_arr[mask_np] = self.config.funnel_tx_count_24h
        burst_arr = np.zeros(N, dtype=np.float64)
        burst_arr[mask_np] = self.config.funnel_burst_ratio
        funnel_arr = np.zeros(N, dtype=np.float64)
        funnel_arr[mask_np] = self.config.funnel_score_target

        return df.with_columns([
            pl.when(mask).then(pl.Series(tx1h_arr)).otherwise(pl.col("tx_count_1h")).alias("tx_count_1h"),
            pl.when(mask).then(pl.Series(tx24h_arr)).otherwise(pl.col("tx_count_24h")).alias("tx_count_24h"),
            pl.when(mask).then(pl.Series(burst_arr)).otherwise(pl.col("burst_ratio")).alias("burst_ratio"),
            pl.when(mask).then(pl.Series(funnel_arr)).otherwise(pl.col("funnel_score")).alias("funnel_score"),
            pl.when(mask).then(pl.lit(True)).otherwise(pl.col("anomaly_flag")).alias("anomaly_flag"),
        ])

    def _inject_mule_activity(
        self, df: pl.DataFrame, mask: pl.Series
    ) -> pl.DataFrame:
        """Pass-through mule: high pass-through, near-zero retention."""
        N = len(df)
        mask_np = mask.to_numpy()
        pt_arr = np.zeros(N, dtype=np.float64)
        pt_arr[mask_np] = self.config.mule_pass_through_ratio
        zbf_arr = np.zeros(N, dtype=np.float64)
        zbf_arr[mask_np] = self.config.mule_zero_balance_freq
        ret_arr = np.zeros(N, dtype=np.float64)
        ret_arr[mask_np] = self.config.mule_retention_ratio
        dep_arr = np.zeros(N, dtype=np.float64)
        dep_arr[mask_np] = self.config.mule_depletion_rate

        return df.with_columns([
            pl.when(mask).then(pl.Series(pt_arr)).otherwise(pl.col("pass_through_ratio")).alias("pass_through_ratio"),
            pl.when(mask).then(pl.Series(zbf_arr)).otherwise(pl.col("zero_balance_frequency")).alias("zero_balance_frequency"),
            pl.when(mask).then(pl.Series(ret_arr)).otherwise(pl.col("balance_retention_ratio")).alias("balance_retention_ratio"),
            pl.when(mask).then(pl.Series(dep_arr)).otherwise(pl.col("balance_depletion_rate")).alias("balance_depletion_rate"),
            pl.when(mask).then(pl.lit(True)).otherwise(pl.col("anomaly_flag")).alias("anomaly_flag"),
        ])

    def _inject_layering(
        self, df: pl.DataFrame, mask: pl.Series
    ) -> pl.DataFrame:
        """Network layering: high centrality, low reciprocity, many new relationships."""
        N = len(df)
        mask_np = mask.to_numpy()
        deg_arr = np.zeros(N, dtype=np.float64)
        deg_arr[mask_np] = self.config.layering_degree_centrality
        rec_arr = np.zeros(N, dtype=np.float64)
        rec_arr[mask_np] = self.config.layering_reciprocity
        newrel_arr = np.zeros(N, dtype=np.int64)
        newrel_arr[mask_np] = self.config.layering_new_relationships
        bshift_arr = np.zeros(N, dtype=np.float64)
        bshift_arr[mask_np] = self.config.layering_behavioural_shift

        return df.with_columns([
            pl.when(mask).then(pl.Series(deg_arr)).otherwise(pl.col("degree_centrality")).alias("degree_centrality"),
            pl.when(mask).then(pl.Series(rec_arr)).otherwise(pl.col("reciprocity_ratio")).alias("reciprocity_ratio"),
            pl.when(mask).then(pl.Series(newrel_arr)).otherwise(pl.col("new_relationships_7d")).alias("new_relationships_7d"),
            pl.when(mask).then(pl.Series(bshift_arr)).otherwise(pl.col("behavioral_shift_score")).alias("behavioral_shift_score"),
            pl.when(mask).then(pl.lit(True)).otherwise(pl.col("anomaly_flag")).alias("anomaly_flag"),
        ])

    def _inject_ceiling_violation(
        self, df: pl.DataFrame, mask: pl.Series
    ) -> pl.DataFrame:
        """Breach CBK regulatory ceiling: TIER_1 account with large tx/balance."""
        N = len(df)
        mask_np = mask.to_numpy()
        amounts = self._rng.normal(
            self.config.ceiling_violation_amount,
            10_000.0,
            size=int(mask.sum()),
        ).clip(15_000, 200_000)
        balances = self._rng.normal(
            self.config.ceiling_post_tx_balance,
            15_000.0,
            size=int(mask.sum()),
        ).clip(60_000, 500_000)

        amt_arr = np.full(N, np.nan, dtype=np.float64)
        amt_arr[mask_np] = amounts
        bal_arr = np.full(N, np.nan, dtype=np.float64)
        bal_arr[mask_np] = balances

        # Build tier-conditioned mask using numpy
        tier_mask = (df["wallet_tier_encoded"].to_numpy() == self.config.ceiling_min_target_tier_encoded)
        combined_mask = mask_np & tier_mask

        return df.with_columns([
            pl.when(pl.Series(combined_mask))
            .then(pl.Series(amt_arr))
            .otherwise(pl.col("amount"))
            .alias("amount"),
            pl.when(pl.Series(combined_mask))
            .then(pl.Series(bal_arr))
            .otherwise(pl.col("post_tx_balance"))
            .alias("post_tx_balance"),
            pl.when(pl.Series(combined_mask))
            .then(pl.Series(bal_arr))
            .otherwise(pl.col("current_balance"))
            .alias("current_balance"),
            pl.when(pl.Series(combined_mask))
            .then(pl.lit(True))
            .otherwise(pl.col("anomaly_flag"))
            .alias("anomaly_flag"),
        ])

    def _inject_high_risk_country(
        self, df: pl.DataFrame, mask: pl.Series
    ) -> pl.DataFrame:
        """Cross-border to high-risk jurisdiction with large amounts."""
        N = len(df)
        mask_np = mask.to_numpy()
        amounts = self._rng.normal(
            self.config.high_risk_amount,
            25_000.0,
            size=int(mask.sum()),
        ).clip(100_000, 500_000)

        amt_arr = np.full(N, np.nan, dtype=np.float64)
        amt_arr[mask_np] = amounts

        return df.with_columns([
            pl.when(mask)
            .then(pl.Series(amt_arr))
            .otherwise(pl.col("amount"))
            .alias("amount"),
            pl.when(mask)
            .then(pl.lit(self.config.high_risk_country_code))
            .otherwise(pl.col("receiver_county"))
            .alias("receiver_county"),
            pl.when(mask)
            .then(pl.lit(True))
            .otherwise(pl.col("anomaly_flag"))
            .alias("anomaly_flag"),
        ])

    def _inject_circular_trading(
        self, df: pl.DataFrame, mask: pl.Series
    ) -> pl.DataFrame:
        """Circular flow within same community with high clustering."""
        N = len(df)
        mask_np = mask.to_numpy()
        amounts = self._rng.normal(
            self.config.circular_amount_mean,
            self.config.circular_amount_sigma,
            size=int(mask.sum()),
        ).clip(10_000, 150_000)

        # Pick random community IDs from non-anomalous data
        existing_communities = df.filter(mask.not_()).get_column("community_id").unique().to_list()
        if not existing_communities:
            existing_communities = [0]
        community_ids = list(self._rng.choice(existing_communities, size=int(mask.sum())))

        amt_arr = np.full(N, np.nan, dtype=np.float64)
        amt_arr[mask_np] = amounts
        comm_arr = np.full(N, np.nan, dtype=np.float64)
        comm_arr[mask_np] = community_ids

        return df.with_columns([
            pl.when(mask)
            .then(pl.Series(amt_arr))
            .otherwise(pl.col("amount"))
            .alias("amount"),
            pl.when(mask)
            .then(pl.Series(comm_arr))
            .otherwise(pl.col("community_id"))
            .alias("community_id"),
            pl.when(mask)
            .then(pl.lit(self.config.circular_clustering))
            .otherwise(pl.col("degree_centrality"))
            .alias("degree_centrality"),
            pl.when(mask)
            .then(pl.lit(True))
            .otherwise(pl.col("anomaly_flag"))
            .alias("anomaly_flag"),
        ])

    def _inject_temporal_anomaly(
        self, df: pl.DataFrame, mask: pl.Series
    ) -> pl.DataFrame:
        """Off-hours activity with device churn and high location entropy."""
        N = len(df)
        mask_np = mask.to_numpy()
        hod_arr = np.zeros(N, dtype=np.int64)
        hod_arr[mask_np] = self.config.temporal_anomalous_hour
        dev_arr = np.zeros(N, dtype=np.int64)
        dev_arr[mask_np] = self.config.temporal_device_changes
        loc_arr = np.zeros(N, dtype=np.float64)
        loc_arr[mask_np] = self.config.temporal_location_entropy

        return df.with_columns([
            pl.when(mask)
            .then(pl.Series(hod_arr))
            .otherwise(pl.col("hour_of_day"))
            .alias("hour_of_day"),
            pl.when(mask)
            .then(pl.Series(dev_arr))
            .otherwise(pl.col("device_changes_7d"))
            .alias("device_changes_7d"),
            pl.when(mask)
            .then(pl.Series(loc_arr))
            .otherwise(pl.col("location_entropy"))
            .alias("location_entropy"),
            pl.when(mask)
            .then(pl.lit(True))
            .otherwise(pl.col("anomaly_flag"))
            .alias("anomaly_flag"),
            pl.when(mask)
            .then(pl.lit(True))
            .otherwise(pl.col("is_anomalous_hour"))
            .alias("is_anomalous_hour"),
        ])
