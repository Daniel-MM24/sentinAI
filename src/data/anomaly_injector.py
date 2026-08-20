"""
AML Anomaly Injector for Synthetic M-PESA Data.

This module provides two anomaly injection approaches:

1. Legacy AMLGenerator AnomalyInjector: Injects structural anomalies into clean 
   AMLGenerator output at a controlled ratio (0.015 = 1.5% of total dataset).
   Operates directly on AMLGenerator-native columns.

2. TVAE Hybrid FinancialAnomalyInjector: Injects deterministic AML scenarios
   into enriched TVAE data (19 columns) with specific scenario patterns
   (structuring, layering, integration, mule) and sets is_launderer/aml_scenario labels.
"""

import logging
import numpy as np
import polars as pl
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta, timezone

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

    TVAE Hybrid v2.0 - 21-feature schema configuration.
    Each parameter controls how the injector modifies TVAE Hybrid columns
    during anomaly injection. Tuned for realistic M-PESA anomaly patterns.
    """
    anomaly_ratio: float = 0.015
    seed: int = 42

    # Structuring / amount anomalies (21-feature schema)
    structuring_amount_target: float = 95_000.0       # KES — just below CTR threshold
    structuring_amount_sigma: float = 5_000.0         # KES — noise around target
    structuring_roundness_threshold: float = 0.85     # High amount_roundness
    structuring_close_to_limit: float = 0.90          # High close_to_limit_ratio

    # Velocity / funnel (21-feature schema)
    funnel_tx_count_7d: int = 45                      # tx_count_7d target
    funnel_volume_7d: float = 500_000.0               # volume_7d target (KES)
    funnel_rapid_tx_ratio: float = 0.40               # rapid_tx_ratio target
    funnel_burst_ratio: float = 3.5                   # volume_7d_vs_30d_ratio target

    # Mule / pass-through (21-feature schema)
    mule_fan_in_fan_out_ratio: float = 0.10          # fan_in_fan_out_ratio (low - mostly outflow)
    mule_retention_ratio: float = 0.10                # balance_retention_ratio (low)
    mule_roundness: float = 0.90                      # amount_roundness (high - structured)

    # Layering / network (21-feature schema)
    layering_distinct_counterparties: int = 15         # distinct_counterparties_7d
    layering_fan_ratio: float = 0.25                   # fan_in_fan_out_ratio
    layering_international: bool = True                # is_international

    # Regulatory ceiling violation (21-feature schema)
    ceiling_violation_amount: float = 80_000.0        # KES — well above TIER_1 cap (50K)
    ceiling_balance: float = 90_000.0                 # KES — above TIER_1 balance cap (50K)
    ceiling_tier: int = 1                             # TIER_1
    ceiling_close_to_limit: float = 0.95              # close_to_limit_ratio

    # High risk country (21-feature schema)
    high_risk_amount: float = 150_000.0               # KES
    high_risk_international: bool = True              # is_international

    # Temporal anomaly (21-feature schema)
    temporal_night_ratio: float = 0.40               # night_tx_ratio target
    temporal_rapid_ratio: float = 0.35                # rapid_tx_ratio target

    # Equal share per anomaly type for TVAE Hybrid scenarios
    anomaly_type_weights: Dict[str, float] = field(default_factory=lambda: {
        "structuring": 0.25,
        "layering": 0.20,
        "mule_activity": 0.20,
        "integration": 0.15,
        "ceiling_violation": 0.10,
        "high_risk_country": 0.10,
        "circular_trading": 0.05,
        "temporal_anomaly": 0.05,
    })


# Columns the injector may modify (TVAE Hybrid v2.0 - 21-feature schema)
INJECTABLE_FEATURES: List[str] = [
    # Core features (8)
    "amount",
    "balance",
    "tier",
    "archetype",
    "transaction_type",
    "direction",
    # Temporal features (5)
    "tx_count_7d",
    "volume_7d",
    "night_tx_ratio",
    "rapid_tx_ratio",
    "volume_7d_vs_30d_ratio",
    # Network features (3)
    "is_international",
    "distinct_counterparties_7d",
    "fan_in_fan_out_ratio",
    # Structuring features (3)
    "close_to_limit_ratio",
    "balance_retention_ratio",
    "amount_roundness",
    # Labels (2)
    "is_launderer",
    "aml_scenario",
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
        close_to_limit_arr = np.full(N, np.nan, dtype=np.float64)
        close_to_limit_arr[mask.to_numpy()] = self.config.structuring_close_to_limit

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
            .then(pl.Series(close_to_limit_arr))
            .otherwise(pl.col("close_to_limit_ratio"))
            .alias("close_to_limit_ratio"),
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
        tx_count_arr = np.zeros(N, dtype=np.int64)
        tx_count_arr[mask_np] = self.config.funnel_tx_count_7d
        volume_arr = np.zeros(N, dtype=np.float64)
        volume_arr[mask_np] = self.config.funnel_volume_7d
        rapid_arr = np.zeros(N, dtype=np.float64)
        rapid_arr[mask_np] = self.config.funnel_rapid_tx_ratio
        burst_arr = np.zeros(N, dtype=np.float64)
        burst_arr[mask_np] = self.config.funnel_burst_ratio

        return df.with_columns([
            pl.when(mask).then(pl.Series(tx_count_arr)).otherwise(pl.col("tx_count_7d")).alias("tx_count_7d"),
            pl.when(mask).then(pl.Series(volume_arr)).otherwise(pl.col("volume_7d")).alias("volume_7d"),
            pl.when(mask).then(pl.Series(rapid_arr)).otherwise(pl.col("rapid_tx_ratio")).alias("rapid_tx_ratio"),
            pl.when(mask).then(pl.Series(burst_arr)).otherwise(pl.col("volume_7d_vs_30d_ratio")).alias("volume_7d_vs_30d_ratio"),
            pl.when(mask).then(pl.lit(True)).otherwise(pl.col("anomaly_flag")).alias("anomaly_flag"),
        ])

    def _inject_mule_activity(
        self, df: pl.DataFrame, mask: pl.Series
    ) -> pl.DataFrame:
        """Pass-through mule: high pass-through, near-zero retention."""
        N = len(df)
        mask_np = mask.to_numpy()
        fan_arr = np.zeros(N, dtype=np.float64)
        fan_arr[mask_np] = self.config.mule_fan_in_fan_out_ratio
        ret_arr = np.zeros(N, dtype=np.float64)
        ret_arr[mask_np] = self.config.mule_retention_ratio
        round_arr = np.zeros(N, dtype=np.float64)
        round_arr[mask_np] = self.config.mule_roundness

        return df.with_columns([
            pl.when(mask).then(pl.Series(fan_arr)).otherwise(pl.col("fan_in_fan_out_ratio")).alias("fan_in_fan_out_ratio"),
            pl.when(mask).then(pl.Series(ret_arr)).otherwise(pl.col("balance_retention_ratio")).alias("balance_retention_ratio"),
            pl.when(mask).then(pl.Series(round_arr)).otherwise(pl.col("amount_roundness")).alias("amount_roundness"),
            pl.when(mask).then(pl.lit(True)).otherwise(pl.col("anomaly_flag")).alias("anomaly_flag"),
        ])

    def _inject_layering(
        self, df: pl.DataFrame, mask: pl.Series
    ) -> pl.DataFrame:
        """Network layering: high distinct counterparties, balanced fan-in/fan-out."""
        N = len(df)
        mask_np = mask.to_numpy()
        distinct_arr = np.zeros(N, dtype=np.int64)
        distinct_arr[mask_np] = self.config.layering_distinct_counterparties
        fan_arr = np.zeros(N, dtype=np.float64)
        fan_arr[mask_np] = self.config.layering_fan_ratio
        international_arr = np.zeros(N, dtype=np.int64)
        international_arr[mask_np] = self.config.layering_international

        return df.with_columns([
            pl.when(mask).then(pl.Series(distinct_arr)).otherwise(pl.col("distinct_counterparties_7d")).alias("distinct_counterparties_7d"),
            pl.when(mask).then(pl.Series(fan_arr)).otherwise(pl.col("fan_in_fan_out_ratio")).alias("fan_in_fan_out_ratio"),
            pl.when(mask).then(pl.Series(international_arr)).otherwise(pl.col("is_international")).alias("is_international"),
            pl.when(mask).then(pl.lit(True)).otherwise(pl.col("is_launderer")).alias("is_launderer"),
            pl.when(mask).then(pl.lit("layering")).otherwise(pl.col("aml_scenario")).alias("aml_scenario"),
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
            self.config.ceiling_balance,
            15_000.0,
            size=int(mask.sum()),
        ).clip(60_000, 500_000)

        amt_arr = np.full(N, np.nan, dtype=np.float64)
        amt_arr[mask_np] = amounts
        bal_arr = np.full(N, np.nan, dtype=np.float64)
        bal_arr[mask_np] = balances
        tier_arr = np.full(N, np.nan, dtype=np.int64)
        tier_arr[mask_np] = self.config.ceiling_tier
        close_to_limit_arr = np.full(N, np.nan, dtype=np.float64)
        close_to_limit_arr[mask_np] = self.config.ceiling_close_to_limit

        return df.with_columns([
            pl.when(mask)
            .then(pl.Series(amt_arr))
            .otherwise(pl.col("amount"))
            .alias("amount"),
            pl.when(mask)
            .then(pl.Series(bal_arr))
            .otherwise(pl.col("balance"))
            .alias("balance"),
            pl.when(mask)
            .then(pl.Series(tier_arr))
            .otherwise(pl.col("tier"))
            .alias("tier"),
            pl.when(mask)
            .then(pl.Series(close_to_limit_arr))
            .otherwise(pl.col("close_to_limit_ratio"))
            .alias("close_to_limit_ratio"),
            pl.when(mask)
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
        international_arr = np.full(N, dtype=np.int64)
        international_arr[mask_np] = self.config.high_risk_international

        return df.with_columns([
            pl.when(mask)
            .then(pl.Series(amt_arr))
            .otherwise(pl.col("amount"))
            .alias("amount"),
            pl.when(mask)
            .then(pl.Series(international_arr))
            .otherwise(pl.col("is_international"))
            .alias("is_international"),
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
            100_000.0,
            25_000.0,
            size=int(mask.sum()),
        ).clip(10_000, 150_000)

        amt_arr = np.full(N, np.nan, dtype=np.float64)
        amt_arr[mask_np] = amounts
        distinct_arr = np.zeros(N, dtype=np.int64)
        distinct_arr[mask_np] = 5  # Low distinct counterparties for circular trading
        fan_arr = np.zeros(N, dtype=np.float64)
        fan_arr[mask_np] = 1.0  # Perfectly balanced fan-in/fan-out for circular

        return df.with_columns([
            pl.when(mask)
            .then(pl.Series(amt_arr))
            .otherwise(pl.col("amount"))
            .alias("amount"),
            pl.when(mask)
            .then(pl.Series(distinct_arr))
            .otherwise(pl.col("distinct_counterparties_7d"))
            .alias("distinct_counterparties_7d"),
            pl.when(mask)
            .then(pl.Series(fan_arr))
            .otherwise(pl.col("fan_in_fan_out_ratio"))
            .alias("fan_in_fan_out_ratio"),
            pl.when(mask)
            .then(pl.lit(True))
            .otherwise(pl.col("anomaly_flag"))
            .alias("anomaly_flag"),
        ])

    def _inject_temporal_anomaly(
        self, df: pl.DataFrame, mask: pl.Series
    ) -> pl.DataFrame:
        """Off-hours activity with high rapid transaction ratio."""
        N = len(df)
        mask_np = mask.to_numpy()
        night_arr = np.zeros(N, dtype=np.float64)
        night_arr[mask_np] = self.config.temporal_night_ratio
        rapid_arr = np.zeros(N, dtype=np.float64)
        rapid_arr[mask_np] = self.config.temporal_rapid_ratio

        return df.with_columns([
            pl.when(mask)
            .then(pl.Series(night_arr))
            .otherwise(pl.col("night_tx_ratio"))
            .alias("night_tx_ratio"),
            pl.when(mask)
            .then(pl.Series(rapid_arr))
            .otherwise(pl.col("rapid_tx_ratio"))
            .alias("rapid_tx_ratio"),
            pl.when(mask)
            .then(pl.lit(True))
            .otherwise(pl.col("anomaly_flag"))
            .alias("anomaly_flag"),
        ])


# ---------------------------------------------------------------------------
# TVAE Hybrid FinancialAnomalyInjector
# ---------------------------------------------------------------------------


class TVAEAMLScenario(str, Enum):
    """AML scenarios for TVAE hybrid pipeline."""
    STRUCTURING = "structuring"
    LAYERING = "layering"
    INTEGRATION = "integration"
    MULE = "mule"
    NONE = "none"


@dataclass
class TVAEInjectorConfig:
    """Configuration for TVAE hybrid anomaly injection."""
    launderer_fraction: float = 0.015  # 1.5% of customers
    seed: int = 42
    
    # Scenario distribution (must sum to 1.0)
    scenario_shares: Dict[str, float] = field(default_factory=lambda: {
        "structuring": 0.35,
        "layering": 0.30,
        "integration": 0.20,
        "mule": 0.15,
    })
    
    # Structuring parameters
    structuring_threshold: float = 100_000.0  # KES - CTR threshold
    structuring_min_tx: int = 25
    structuring_max_counterparties: int = 12
    
    # Layering parameters
    layering_min_accounts: int = 4
    layering_window_hours: int = 24
    layering_tx_per_cycle: tuple[int, int] = (2, 4)
    
    # Integration parameters
    integration_min_amount: float = 200_000.0  # KES
    integration_international_ratio: float = 0.80
    
    # Mule parameters
    mule_withdraw_pct: float = 0.85
    mule_min_cycles: int = 5
    mule_retention_threshold: float = 0.15  # Low balance retention
    
    # Regulatory tier limits (from regulatory.yaml)
    tier_limits: Dict[int, float] = field(default_factory=lambda: {
        1: 50_000.0,
        2: 200_000.0,
        3: 1_000_000.0,
        4: 5_000_000.0,
    })


class FinancialAnomalyInjector:
    """Injects deterministic AML scenarios into enriched TVAE data.
    
    Takes enriched TVAE data (19 columns) as input and implements deterministic
    AML scenario injection with specific patterns for:
    - Structuring: Multiple transactions just below tier limits
    - Layering: Rapid transfers between multiple accounts
    - Integration: Large international transfers followed by withdrawals
    - Mule: High inflow with immediate outflow (low balance retention)
    
    Preserves regulatory constraints during injection and outputs final gold dataset
    (21 columns + labels) with is_launderer and aml_scenario labels.
    """
    
    def __init__(self, config: Optional[TVAEInjectorConfig] = None):
        self.config = config or TVAEInjectorConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._launderer_map: Dict[str, str] = {}
        self._injection_stats: Dict[str, Any] = {}
        
    def inject(
        self,
        enriched_df: pl.DataFrame,
        partition: str = "default",
        output_dir: str = "data/gold"
    ) -> pl.DataFrame:
        """Inject AML scenarios into enriched TVAE data.
        
        Args:
            enriched_df: Enriched TVAE data (19 columns)
            partition: Partition identifier for output file naming
            output_dir: Directory for output parquet file
            
        Returns:
            DataFrame with final gold dataset (21 columns + labels)
        """
        logger.info("=== Starting TVAE Hybrid Anomaly Injection ===")
        
        # Validate input schema
        self._validate_input_schema(enriched_df)
        
        # Initialize label columns
        result = enriched_df.with_columns([
            pl.lit(False).alias("is_launderer"),
            pl.lit("none", dtype=pl.Utf8).alias("aml_scenario")
        ])
        
        # Select launderers
        customer_ids = result["customer_id"].unique().to_list()
        self._launderer_map = self._select_launderers(customer_ids)
        
        # Inject scenarios per customer
        for customer_id, scenario in self._launderer_map.items():
            customer_data = result.filter(pl.col("customer_id") == customer_id)
            
            if scenario == "structuring":
                result = self._inject_structuring(result, customer_id, customer_data)
            elif scenario == "layering":
                result = self._inject_layering(result, customer_id, customer_data)
            elif scenario == "integration":
                result = self._inject_integration(result, customer_id, customer_data)
            elif scenario == "mule":
                result = self._inject_mule(result, customer_id, customer_data)
        
        # Apply labels
        result = result.with_columns([
            pl.when(pl.col("customer_id").is_in(list(self._launderer_map.keys())))
            .then(pl.lit(True))
            .otherwise(pl.col("is_launderer"))
            .alias("is_launderer"),
            pl.when(pl.col("customer_id").is_in(list(self._launderer_map.keys())))
            .then(pl.col("customer_id").map_dict(self._launderer_map, default="none"))
            .otherwise(pl.col("aml_scenario"))
            .alias("aml_scenario")
        ])
        
        # Validate regulatory constraints
        self._validate_regulatory_constraints(result)
        
        # Validate labels
        self._validate_labels(result)
        
        # Save output
        output_path = Path(output_dir) / f"tvae_hybrid_gold_{partition}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.write_parquet(output_path)
        
        # Log statistics
        self._log_injection_statistics(result)
        
        logger.info(f"Saved gold dataset to {output_path}")
        logger.info("=== TVAE Hybrid Anomaly Injection Complete ===")
        
        return result
    
    def _validate_input_schema(self, df: pl.DataFrame) -> None:
        """Validate that input has required 19 columns."""
        required_columns = [
            "customer_id", "tier", "archetype", "transaction_type",
            "amount", "timestamp", "direction", "balance",
            "tx_count_7d", "volume_7d", "night_tx_ratio", "rapid_tx_ratio",
            "volume_7d_vs_30d_ratio", "is_international", "distinct_counterparties_7d",
            "fan_in_fan_out_ratio", "close_to_limit_ratio", "balance_retention_ratio",
            "amount_roundness"
        ]
        
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        logger.info("Input schema validated: 19 columns present")
    
    def _select_launderers(self, customer_ids: List[str]) -> Dict[str, str]:
        """Select specified percentage of customers as launderers and assign scenarios."""
        total = len(customer_ids)
        n_launderers = max(1, round(total * self.config.launderer_fraction))
        selected = self._rng.choice(customer_ids, size=n_launderers, replace=False).tolist()
        
        # Assign scenarios according to configured shares
        scenarios = list(self.config.scenario_shares.keys())
        weights = [self.config.scenario_shares[s] for s in scenarios]
        weights = np.array(weights) / sum(weights)
        
        n_per_scenario = (weights * n_launderers).astype(int)
        diff = n_launderers - n_per_scenario.sum()
        if diff > 0:
            n_per_scenario[:diff] += 1
        
        assignments = []
        for scenario, count in zip(scenarios, n_per_scenario):
            assignments.extend([scenario] * count)
        assignments = assignments[:n_launderers]
        self._rng.shuffle(assignments)
        
        mapping = dict(zip(selected, assignments))
        
        logger.info(f"Selected {n_launderers} launderers ({self.config.launderer_fraction*100:.1f}% of {total})")
        logger.info(f"Scenario distribution: {dict(zip(scenarios, n_per_scenario))}")
        
        return mapping
    
    def _inject_structuring(
        self,
        df: pl.DataFrame,
        customer_id: str,
        customer_data: pl.DataFrame
    ) -> pl.DataFrame:
        """Inject structuring pattern: multiple transactions just below tier limits."""
        n_tx = self.config.structuring_min_tx + int(self._rng.exponential(15))
        tier = customer_data["tier"].unique()[0]
        tier_limit = self.config.tier_limits.get(tier, 100_000.0)
        
        # Generate amounts just below threshold (90-95% of limit)
        amounts = self._rng.uniform(
            tier_limit * 0.90,
            tier_limit * 0.95,
            size=n_tx
        )
        
        # Update customer's transactions
        customer_indices = np.where(df["customer_id"].to_numpy() == customer_id)[0]
        if len(customer_indices) > 0:
            # Modify existing transactions to show structuring pattern
            n_modify = min(n_tx, len(customer_indices))
            modify_indices = self._rng.choice(customer_indices, size=n_modify, replace=False)
            
            amount_col = df["amount"].to_numpy()
            amount_col[modify_indices] = amounts[:n_modify]
            
            roundness_col = df["amount_roundness"].to_numpy()
            roundness_col[modify_indices] = 0.85  # High roundness
            
            close_to_limit_col = df["close_to_limit_ratio"].to_numpy()
            close_to_limit_col[modify_indices] = 0.92  # Close to limit
            
            df = df.with_columns([
                pl.Series("amount", amount_col),
                pl.Series("amount_roundness", roundness_col),
                pl.Series("close_to_limit_ratio", close_to_limit_col)
            ])
        
        logger.debug(f"Structuring injected for {customer_id}: {n_tx} transactions")
        return df
    
    def _inject_layering(
        self,
        df: pl.DataFrame,
        customer_id: str,
        customer_data: pl.DataFrame
    ) -> pl.DataFrame:
        """Inject layering pattern: rapid transfers between multiple accounts."""
        customer_indices = np.where(df["customer_id"].to_numpy() == customer_id)[0]
        
        if len(customer_indices) > 0:
            # Increase distinct counterparties and velocity
            distinct_cp_col = df["distinct_counterparties_7d"].to_numpy()
            distinct_cp_col[customer_indices] = self.config.layering_min_accounts + int(self._rng.integers(0, 5))
            
            rapid_tx_col = df["rapid_tx_ratio"].to_numpy()
            rapid_tx_col[customer_indices] = 0.75  # High rapid transaction ratio
            
            volume_7d_col = df["volume_7d"].to_numpy()
            volume_7d_col[customer_indices] *= 2.5  # Increased volume
            
            fan_in_out_col = df["fan_in_fan_out_ratio"].to_numpy()
            fan_in_out_col[customer_indices] = 1.1  # Near-balanced flow
            
            df = df.with_columns([
                pl.Series("distinct_counterparties_7d", distinct_cp_col),
                pl.Series("rapid_tx_ratio", rapid_tx_col),
                pl.Series("volume_7d", volume_7d_col),
                pl.Series("fan_in_fan_out_ratio", fan_in_out_col)
            ])
        
        logger.debug(f"Layering injected for {customer_id}")
        return df
    
    def _inject_integration(
        self,
        df: pl.DataFrame,
        customer_id: str,
        customer_data: pl.DataFrame
    ) -> pl.DataFrame:
        """Inject integration pattern: large international transfers followed by withdrawals."""
        customer_indices = np.where(df["customer_id"].to_numpy() == customer_id)[0]
        
        if len(customer_indices) > 0:
            # Set large amounts and international flag
            amount_col = df["amount"].to_numpy()
            amount_col[customer_indices] = self._rng.uniform(
                self.config.integration_min_amount,
                self.config.integration_min_amount * 2.0,
                size=len(customer_indices)
            )
            
            international_col = df["is_international"].to_numpy()
            international_col[customer_indices] = True
            
            # High volume burst
            volume_7d_col = df["volume_7d"].to_numpy()
            volume_7d_col[customer_indices] *= 3.0
            
            burst_ratio_col = df["volume_7d_vs_30d_ratio"].to_numpy()
            burst_ratio_col[customer_indices] = 2.5  # High burst ratio
            
            df = df.with_columns([
                pl.Series("amount", amount_col),
                pl.Series("is_international", international_col),
                pl.Series("volume_7d", volume_7d_col),
                pl.Series("volume_7d_vs_30d_ratio", burst_ratio_col)
            ])
        
        logger.debug(f"Integration injected for {customer_id}")
        return df
    
    def _inject_mule(
        self,
        df: pl.DataFrame,
        customer_id: str,
        customer_data: pl.DataFrame
    ) -> pl.DataFrame:
        """Inject mule pattern: high inflow with immediate outflow (low balance retention)."""
        customer_indices = np.where(df["customer_id"].to_numpy() == customer_id)[0]
        
        if len(customer_indices) > 0:
            # Low balance retention (pass-through)
            retention_col = df["balance_retention_ratio"].to_numpy()
            retention_col[customer_indices] = self.config.mule_retention_threshold
            
            # High fan-in/fan-out ratio (balanced flow)
            fan_in_out_col = df["fan_in_fan_out_ratio"].to_numpy()
            fan_in_out_col[customer_indices] = 0.95  # Near 1.0 = balanced
            
            # High volume with rapid turnover
            volume_7d_col = df["volume_7d"].to_numpy()
            volume_7d_col[customer_indices] *= 2.0
            
            rapid_tx_col = df["rapid_tx_ratio"].to_numpy()
            rapid_tx_col[customer_indices] = 0.80
            
            # Low close_to_limit (not trying to evade limits)
            close_to_limit_col = df["close_to_limit_ratio"].to_numpy()
            close_to_limit_col[customer_indices] = 0.40
            
            df = df.with_columns([
                pl.Series("balance_retention_ratio", retention_col),
                pl.Series("fan_in_fan_out_ratio", fan_in_out_col),
                pl.Series("volume_7d", volume_7d_col),
                pl.Series("rapid_tx_ratio", rapid_tx_col),
                pl.Series("close_to_limit_ratio", close_to_limit_col)
            ])
        
        logger.debug(f"Mule pattern injected for {customer_id}")
        return df
    
    def _validate_regulatory_constraints(self, df: pl.DataFrame) -> None:
        """Validate that injected data respects regulatory tier limits."""
        logger.info("Validating regulatory constraints...")
        
        # Check balance caps per tier
        for tier, limit in self.config.tier_limits.items():
            tier_data = df.filter(pl.col("tier") == tier)
            if len(tier_data) > 0:
                violations = tier_data.filter(pl.col("balance") > limit)
                if len(violations) > 0:
                    logger.warning(
                        f"Found {len(violations)} balance cap violations for Tier {tier} "
                        f"(limit: {limit})"
                    )
                else:
                    logger.info(f"Tier {tier} balance cap validated: {len(tier_data)} records OK")
        
        # Check transaction limits per tier
        tier_tx_limits = {1: 10_000, 2: 50_000, 3: 150_000, 4: 500_000}
        for tier, limit in tier_tx_limits.items():
            tier_data = df.filter(pl.col("tier") == tier)
            if len(tier_data) > 0:
                violations = tier_data.filter(pl.col("amount") > limit)
                if len(violations) > 0:
                    logger.warning(
                        f"Found {len(violations)} transaction limit violations for Tier {tier} "
                        f"(limit: {limit})"
                    )
                else:
                    logger.info(f"Tier {tier} transaction limit validated: {len(tier_data)} records OK")
    
    def _validate_labels(self, df: pl.DataFrame) -> None:
        """Validate that labels are correctly applied."""
        logger.info("Validating labels...")
        
        # Check label columns exist
        assert "is_launderer" in df.columns
        assert "aml_scenario" in df.columns
        
        # Count labels
        launderer_count = df.filter(pl.col("is_launderer") == True).shape[0]
        total_count = df.shape[0]
        
        logger.info(f"Launderer count: {launderer_count} ({launderer_count/total_count*100:.2f}%)")
        
        # Check scenario distribution
        scenario_dist = (
            df.filter(pl.col("is_launderer") == True)
            .group_by("aml_scenario")
            .agg(pl.len().alias("count"))
            .sort("aml_scenario")
        )
        
        logger.info(f"Scenario distribution among launderers:\n{scenario_dist}")
        
        # Validate all launderers have non-"none" scenario
        invalid = df.filter(
            (pl.col("is_launderer") == True) & (pl.col("aml_scenario") == "none")
        )
        if len(invalid) > 0:
            raise ValueError(f"Found {len(invalid)} launderers with 'none' scenario")
        
        # Validate all non-launderers have "none" scenario
        invalid = df.filter(
            (pl.col("is_launderer") == False) & (pl.col("aml_scenario") != "none")
        )
        if len(invalid) > 0:
            raise ValueError(f"Found {len(invalid)} non-launderers with non-'none' scenario")
        
        logger.info("Label validation passed")
    
    def _log_injection_statistics(self, df: pl.DataFrame) -> None:
        """Log comprehensive injection statistics."""
        total = len(df)
        launderers = df.filter(pl.col("is_launderer") == True)
        n_launderers = launderers.shape[0]
        
        self._injection_stats = {
            "total_records": total,
            "launderer_count": n_launderers,
            "launderer_percentage": round(n_launderers / total * 100, 2),
            "scenario_distribution": {}
        }
        
        for scenario in TVAEAMLScenario:
            if scenario != TVAEAMLScenario.NONE:
                count = launderers.filter(pl.col("aml_scenario") == scenario.value).shape[0]
                self._injection_stats["scenario_distribution"][scenario.value] = count
        
        logger.info("=== Injection Statistics ===")
        logger.info(f"Total records: {self._injection_stats['total_records']}")
        logger.info(f"Launderer count: {self._injection_stats['launderer_count']}")
        logger.info(f"Launderer percentage: {self._injection_stats['launderer_percentage']}%")
        logger.info("Scenario distribution:")
        for scenario, count in self._injection_stats["scenario_distribution"].items():
            logger.info(f"  {scenario}: {count}")
    
    def get_injection_stats(self) -> Dict[str, Any]:
        """Return injection statistics."""
        return self._injection_stats
    
    def get_launderer_map(self) -> Dict[str, str]:
        """Return customer_id -> scenario mapping."""
        return dict(self._launderer_map)
