"""
Feature Engineering Pipeline for Customer Features (TVAE Hybrid v2.0)

This module computes time-varying customer-level features from raw transactions.
These features are stored in the customer_features table with proper foreign key relationships.

TVAE Hybrid Implementation (v2.0) - 10 downstream features:
- Temporal Features (5): tx_count_7d, volume_7d, night_tx_ratio, rapid_tx_ratio, volume_7d_vs_30d_ratio
- Network Features (2): distinct_counterparties_7d, fan_in_fan_out_ratio
- Structuring Features (3): close_to_limit_ratio, balance_retention_ratio, amount_roundness
"""

import logging
import polars as pl
from typing import Dict, Any, Optional
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from pathlib import Path
import random

logger = logging.getLogger(__name__)


class CustomerFeatureEngineer:
    """
    Computes customer-level features from transaction data.
    
    This class takes raw transaction data and computes the aggregate/rolling
    features that belong in the customer_features table. It maintains proper
    foreign key relationships to the customers table.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature engineer.
        
        Args:
            config: Optional configuration dictionary for feature computation parameters
        """
        self.config = config or {}
        
        # Default windows for feature computation
        self.velocity_windows = self.config.get("velocity_windows", ["1h", "24h", "7d", "30d"])
        self.balance_windows = self.config.get("balance_windows", ["7d", "30d"])
        
        logger.info("Initialized CustomerFeatureEngineer")
    
    def compute_features(
        self,
        transactions_df: pl.DataFrame,
        feature_date: Optional[datetime] = None,
        partition: str = "default",
        output_dir: str = "data"
    ) -> pl.DataFrame:
        """
        Compute customer features from transaction data.
        
        Computes the engineered temporal/behavioral features required for the Gold layer:
        - tx_count_7d
        - volume_7d
        - night_tx_ratio
        - rapid_tx_ratio
        - volume_7d_vs_30d_ratio
        - distinct_counterparties_7d
        - fan_in_fan_out_ratio
        - close_to_limit_ratio
        - balance_retention_ratio
        - amount_roundness
        
        Args:
            transactions_df: DataFrame with balance-corrected TVAE events (9 columns)
            feature_date: Date for feature computation window
            partition: Partition identifier for output file naming
            output_dir: Directory for output parquet file
        
        Returns:
            DataFrame with enriched data (19 columns: original 9 + 10 features)
        """
        customer_id_col = None
        for col in ["customer_id", "entity_id", "sender_id"]:
            if col in transactions_df.columns:
                customer_id_col = col
                break
        
        if customer_id_col is None:
            raise ValueError(f"transactions_df must contain a customer ID column. Available columns: {transactions_df.columns}")
        
        if feature_date is None:
            if "timestamp" in transactions_df.columns:
                feature_date = transactions_df["timestamp"].max()
            else:
                feature_date = datetime.now(timezone.utc)
                
        # Rename for consistency
        if customer_id_col != "customer_id":
            transactions_df = transactions_df.rename({customer_id_col: "customer_id"})
            
        if transactions_df.schema.get("timestamp") in (pl.String, pl.Utf8):
            transactions_df = transactions_df.with_columns(
                pl.col("timestamp").str.to_datetime(time_zone="UTC", strict=False).alias("timestamp")
            )
            
        df = transactions_df.sort("timestamp")
        
        # Tier limits mapping for close_to_limit_ratio
        # 1: 50k, 2: 200k, 3: 1m, 4: 5m
        tier_limits = pl.when(pl.col("tier") == 1).then(50000.0) \
                        .when(pl.col("tier") == 2).then(200000.0) \
                        .when(pl.col("tier") == 3).then(1000000.0) \
                        .otherwise(5000000.0)
                        
        df = df.with_columns(tier_limits.alias("tier_limit"))
        
        # Generate synthetic counterparty IDs if missing (TVAE doesn't generate counterparties)
        if "counterparty" not in df.columns:
            logger.info("Generating synthetic counterparty IDs for TVAE data")
            # Create synthetic counterparty IDs based on customer_id hash
            # This ensures deterministic behavior while simulating network structure
            unique_customers = df["customer_id"].unique().to_list()
            customer_to_counterparty = {cid: f"CP_{abs(hash(cid)) % 10000:04d}" for cid in unique_customers}
            
            # Map each customer to their primary counterparty
            df = df.with_columns([
                pl.col("customer_id").map_elements(
                    lambda x: customer_to_counterparty.get(x, f"CP_{abs(hash(x)) % 10000:04d}"),
                    return_dtype=pl.Utf8
                ).alias("counterparty")
            ])
            
            # Add variety: for 20% of transactions, assign to random counterparties
            # This simulates having multiple different counterparties
            random.seed(42)
            n_rows = len(df)
            random_indices = set(random.sample(range(n_rows), min(int(n_rows * 0.2), n_rows)))
            
            # Create random counterparty IDs for variety
            random_counterparties = [f"CP_R{i:04d}" for i in range(len(random_indices))]
            
            # Apply random assignments using a simpler approach
            df_dict = df.to_dict(as_series=False)
            for idx, cp in zip(random_indices, random_counterparties):
                df_dict["counterparty"][idx] = cp
            
            df = pl.DataFrame(df_dict)
        
        cp_col = "counterparty"

        features = df.group_by("customer_id").agg([
            pl.col("timestamp").filter(
                (pl.col("timestamp") >= (feature_date - timedelta(days=7))) &
                (pl.col("timestamp") <= feature_date)
            ).len().alias("tx_count_7d"),
            
            pl.col("amount").filter(
                (pl.col("timestamp") >= (feature_date - timedelta(days=7))) &
                (pl.col("timestamp") <= feature_date)
            ).sum().alias("volume_7d"),
            
            # 30 day volume for burst ratio
            pl.col("amount").filter(
                (pl.col("timestamp") >= (feature_date - timedelta(days=30))) &
                (pl.col("timestamp") <= feature_date)
            ).sum().alias("volume_30d"),
            
            (pl.col("timestamp").filter(
                (pl.col("timestamp") >= (feature_date - timedelta(days=7))) &
                (pl.col("timestamp") <= feature_date) &
                ((pl.col("timestamp").dt.hour().is_between(22, 23)) | 
                 (pl.col("timestamp").dt.hour().is_between(0, 5)))
            ).len().cast(pl.Float64) / 
            pl.col("timestamp").filter(
                (pl.col("timestamp") >= (feature_date - timedelta(days=7))) &
                (pl.col("timestamp") <= feature_date)
            ).len().cast(pl.Float64)).alias("night_tx_ratio"),
            
            (pl.col("timestamp").diff().filter(
                pl.col("timestamp").diff() <= timedelta(minutes=5)
            ).len().cast(pl.Float64) / pl.len().cast(pl.Float64)).alias("rapid_tx_ratio"),
            
            pl.col(cp_col).filter(
                (pl.col("timestamp") >= (feature_date - timedelta(days=7))) &
                (pl.col("timestamp") <= feature_date)
            ).n_unique().alias("distinct_counterparties_7d"),
            
            pl.col("amount").filter(pl.col("direction") == "inflow").sum().alias("total_inflow"),
            pl.col("amount").filter(pl.col("direction") == "outflow").sum().alias("total_outflow"),
            
            # close to limit ratio: avg of amount / tier_limit
            (pl.col("amount") / pl.col("tier_limit")).mean().alias("close_to_limit_ratio"),
            
            # amount roundness
            (pl.col("amount") % 1000 == 0).mean().alias("amount_roundness"),
            
            # balance retention (last balance / avg balance)
            (pl.col("balance").last() / pl.col("balance").mean()).alias("balance_retention_ratio")
        ])
        
        # Calculate derived ratios
        features = features.with_columns([
            (pl.col("volume_7d") / (pl.col("volume_30d") / 4.28 + 1.0)).alias("volume_7d_vs_30d_ratio"),
            (pl.col("total_inflow") / (pl.col("total_outflow") + 1.0)).alias("fan_in_fan_out_ratio"),
        ])
        
        # Cleanup and fill nulls
        features = features.select([
            "customer_id", "tx_count_7d", "volume_7d", "night_tx_ratio", 
            "rapid_tx_ratio", "volume_7d_vs_30d_ratio", "distinct_counterparties_7d",
            "fan_in_fan_out_ratio", "close_to_limit_ratio", "balance_retention_ratio",
            "amount_roundness"
        ]).with_columns([
            pl.col("tx_count_7d").fill_null(0),
            pl.col("volume_7d").fill_null(0.0),
            pl.col("night_tx_ratio").fill_null(0.0),
            pl.col("rapid_tx_ratio").fill_null(0.0),
            pl.col("volume_7d_vs_30d_ratio").fill_null(0.0),
            pl.col("distinct_counterparties_7d").fill_null(0),
            pl.col("fan_in_fan_out_ratio").fill_null(0.0),
            pl.col("close_to_limit_ratio").fill_null(0.0),
            pl.col("balance_retention_ratio").fill_null(0.0),
            pl.col("amount_roundness").fill_null(0.0),
        ])
        
        # Merge features back with original transaction data to create enriched dataset
        # Join on customer_id to get 19 columns total (9 original + 10 features)
        enriched_df = df.join(features, on="customer_id", how="left")
        
        # Log feature distribution statistics
        self._log_feature_distributions(features)
        
        # Save enriched data to parquet
        output_path = Path(output_dir) / f"tvae_enriched_{partition}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enriched_df.write_parquet(output_path)
        logger.info(f"Saved enriched data ({len(enriched_df)} rows, {len(enriched_df.columns)} columns) to {output_path}")
        
        return enriched_df
    
    def _log_feature_distributions(self, features_df: pl.DataFrame) -> None:
        """
        Log feature distribution statistics for monitoring and validation.
        
        Args:
            features_df: DataFrame with computed features
        """
        logger.info("=" * 60)
        logger.info("Feature Distribution Statistics")
        logger.info("=" * 60)
        
        feature_cols = [
            "tx_count_7d", "volume_7d", "night_tx_ratio", "rapid_tx_ratio",
            "volume_7d_vs_30d_ratio", "distinct_counterparties_7d",
            "fan_in_fan_out_ratio", "close_to_limit_ratio",
            "balance_retention_ratio", "amount_roundness"
        ]
        
        for col in feature_cols:
            if col in features_df.columns:
                col_data = features_df[col]
                logger.info(f"\n{col}:")
                logger.info(f"  Mean:   {col_data.mean():.4f}")
                logger.info(f"  Std:    {col_data.std():.4f}")
                logger.info(f"  Min:    {col_data.min():.4f}")
                logger.info(f"  25%:    {col_data.quantile(0.25):.4f}")
                logger.info(f"  50%:    {col_data.quantile(0.50):.4f}")
                logger.info(f"  75%:    {col_data.quantile(0.75):.4f}")
                logger.info(f"  Max:    {col_data.max():.4f}")
                logger.info(f"  Nulls:  {col_data.null_count()}")
        
        logger.info("=" * 60)


def compute_customer_features(
    transactions_df: pl.DataFrame,
    feature_date: Optional[datetime] = None,
    config: Optional[Dict[str, Any]] = None
) -> pl.DataFrame:
    """
    Convenience function to compute customer features.
    
    Args:
        transactions_df: Raw transaction DataFrame
        feature_date: Date for feature computation
        config: Optional configuration
        
    Returns:
        DataFrame with customer features
    """
    engineer = CustomerFeatureEngineer(config)
    return engineer.compute_features(transactions_df, feature_date)
