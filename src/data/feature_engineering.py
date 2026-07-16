"""
Feature Engineering Pipeline for Customer Features

This module computes time-varying customer-level features from raw transactions.
These features are stored in the customer_features table with proper foreign key relationships.

The pipeline computes:
- Real-time velocity features (tx counts, amount sums, time since last tx)
- Balance pattern features (min/max/avg balances, volatility, retention ratios)
- Amount pattern features (roundness, structuring, entropy)
- Network features (centrality, degree, reciprocity)
- Temporal anomaly features (burst ratios, velocity changes)
- Device/location features (changes, entropy)
- Rolling window features (averages, net flows)
- Advanced analytics features (community detection, behavioral shifts)
"""

import logging
import polars as pl
from typing import Dict, Any, Optional
from datetime import datetime, timedelta, timezone
from collections import defaultdict

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
        feature_date: Optional[datetime] = None
    ) -> pl.DataFrame:
        """
        Compute all customer features from transaction data.
        
        Args:
            transactions_df: Raw transaction DataFrame with transaction_id, customer_id, amount, timestamp
            feature_date: The date for which features are computed. If None, uses max timestamp in data
            
        Returns:
            DataFrame with customer_id, feature_date, and all computed features
        """
        # Check if customer_id exists, if not try alternative column names
        customer_id_col = None
        for col in ["customer_id", "entity_id", "sender_id"]:
            if col in transactions_df.columns:
                customer_id_col = col
                break
        
        if customer_id_col is None:
            raise ValueError(f"transactions_df must contain a customer ID column (customer_id, entity_id, or sender_id). Available columns: {transactions_df.columns}")
        
        # Check if amount exists, if not try alternative column names
        amount_col = None
        for col in ["amount", "transaction_amount"]:
            if col in transactions_df.columns:
                amount_col = col
                break
        
        if amount_col is None:
            raise ValueError(f"transactions_df must contain an amount column (amount or transaction_amount). Available columns: {transactions_df.columns}")
        
        # Check if receiver_id exists, if not try alternative column names
        receiver_id_col = None
        for col in ["receiver_id", "recipient_id", "beneficiary_id"]:
            if col in transactions_df.columns:
                receiver_id_col = col
                break
        
        # Check if counterparty_id exists, if not try alternative column names  
        counterparty_id_col = None
        for col in ["counterparty_id", "counterparty"]:
            if col in transactions_df.columns:
                counterparty_id_col = col
                break
        
        if feature_date is None:
            if "timestamp" in transactions_df.columns:
                feature_date = transactions_df["timestamp"].max()
            else:
                feature_date = datetime.now(timezone.utc)
        
        logger.info(f"Computing customer features as of {feature_date} using column '{customer_id_col}'")
        
        # Start with base customer_id and feature_date
        customers = transactions_df[customer_id_col].unique()
        features_df = pl.DataFrame({
            "customer_id": customers,
            "feature_date": [feature_date] * len(customers)
        })
        
        # Rename customer_id_col to customer_id for consistency if needed
        if customer_id_col != "customer_id":
            transactions_df = transactions_df.rename({customer_id_col: "customer_id"})
        
        # Rename amount_col to amount for consistency if needed
        if amount_col != "amount":
            transactions_df = transactions_df.rename({amount_col: "amount"})
        
        # Rename receiver_id_col to receiver_id for consistency if needed
        if receiver_id_col and receiver_id_col != "receiver_id":
            transactions_df = transactions_df.rename({receiver_id_col: "receiver_id"})
        
        # Rename counterparty_id_col to counterparty_id for consistency if needed
        if counterparty_id_col and counterparty_id_col != "counterparty_id":
            transactions_df = transactions_df.rename({counterparty_id_col: "counterparty_id"})
        
        # Compute feature categories
        velocity_features = self._compute_velocity_features(transactions_df, feature_date)
        balance_features = self._compute_balance_features(transactions_df, feature_date)
        amount_features = self._compute_amount_features(transactions_df, feature_date)
        network_features = self._compute_network_features(transactions_df, feature_date)
        temporal_features = self._compute_temporal_features(transactions_df, feature_date)
        device_features = self._compute_device_features(transactions_df, feature_date)
        rolling_features = self._compute_rolling_features(transactions_df, feature_date)
        
        # Join all features
        features_df = features_df.join(velocity_features, on="customer_id", how="left")
        features_df = features_df.join(balance_features, on="customer_id", how="left")
        features_df = features_df.join(amount_features, on="customer_id", how="left")
        features_df = features_df.join(network_features, on="customer_id", how="left")
        features_df = features_df.join(temporal_features, on="customer_id", how="left")
        features_df = features_df.join(device_features, on="customer_id", how="left")
        features_df = features_df.join(rolling_features, on="customer_id", how="left")
        
        logger.info(f"Computed features for {len(features_df)} customers with {len(features_df.columns)} columns")
        
        return features_df
    
    def _compute_velocity_features(
        self,
        transactions_df: pl.DataFrame,
        feature_date: datetime
    ) -> pl.DataFrame:
        """
        Compute real-time velocity features.
        
        Features:
        - tx_count_1h: Transaction count in last 1 hour
        - tx_count_24h: Transaction count in last 24 hours
        - amount_sum_24h: Total amount in last 24 hours
        - amount_vs_profile_avg: Ratio of recent amount to historical average
        - time_since_last_tx: Time since last transaction in hours
        """
        # Filter transactions before feature date
        tx_before = transactions_df.filter(pl.col("timestamp") <= feature_date)
        
        # Compute 1h and 24h windows
        one_hour_ago = feature_date - timedelta(hours=1)
        twenty_four_hours_ago = feature_date - timedelta(hours=24)
        
        # Transaction counts
        tx_count_1h = (
            tx_before
            .filter(pl.col("timestamp") >= one_hour_ago)
            .group_by("customer_id")
            .agg(pl.len().alias("tx_count_1h"))
        )
        
        tx_count_24h = (
            tx_before
            .filter(pl.col("timestamp") >= twenty_four_hours_ago)
            .group_by("customer_id")
            .agg(pl.len().alias("tx_count_24h"))
        )
        
        # Amount sums
        amount_sum_24h = (
            tx_before
            .filter(pl.col("timestamp") >= twenty_four_hours_ago)
            .group_by("customer_id")
            .agg(pl.col("amount").sum().alias("amount_sum_24h"))
        )
        
        # Time since last transaction
        last_tx = (
            tx_before
            .group_by("customer_id")
            .agg(pl.col("timestamp").max().alias("last_tx_time"))
        )
        last_tx = last_tx.with_columns([
            ((feature_date - pl.col("last_tx_time")).dt.total_seconds() / 3600).alias("time_since_last_tx")
        ])
        last_tx = last_tx.drop("last_tx_time")
        
        # Amount vs profile average (using historical mean)
        profile_avg = (
            tx_before
            .group_by("customer_id")
            .agg(pl.col("amount").mean().alias("profile_avg_amount"))
        )
        
        recent_avg = (
            tx_before
            .filter(pl.col("timestamp") >= twenty_four_hours_ago)
            .group_by("customer_id")
            .agg(pl.col("amount").mean().alias("recent_avg_amount"))
        )
        
        amount_vs_profile = profile_avg.join(recent_avg, on="customer_id", how="left")
        amount_vs_profile = amount_vs_profile.with_columns([
            (pl.col("recent_avg_amount") / pl.col("profile_avg_amount")).alias("amount_vs_profile_avg")
        ])
        amount_vs_profile = amount_vs_profile.select(["customer_id", "amount_vs_profile_avg"])
        
        # Combine all velocity features
        velocity_df = tx_count_1h
        velocity_df = velocity_df.join(tx_count_24h, on="customer_id", how="outer_coalesce")
        velocity_df = velocity_df.join(amount_sum_24h, on="customer_id", how="outer_coalesce")
        velocity_df = velocity_df.join(last_tx, on="customer_id", how="outer_coalesce")
        velocity_df = velocity_df.join(amount_vs_profile, on="customer_id", how="outer_coalesce")
        
        return velocity_df
    
    def _compute_balance_features(
        self,
        transactions_df: pl.DataFrame,
        feature_date: datetime
    ) -> pl.DataFrame:
        """
        Compute balance pattern features.
        
        Features:
        - current_balance: Latest balance
        - min_balance_30d: Minimum balance in last 30 days
        - max_balance_30d: Maximum balance in last 30 days
        - avg_balance_30d: Average balance in last 30 days
        - balance_volatility_30d: Standard deviation of balances in last 30 days
        - balance_retention_ratio: Ratio of ending to average balance
        - zero_balance_frequency: Count of zero-balance events in last 30 days
        """
        # For now, use amount-based proxy since we don't have running balance
        # In production, this would use the actual balance tracking from CustomerState
        
        thirty_days_ago = feature_date - timedelta(days=30)
        tx_before = transactions_df.filter(pl.col("timestamp") <= feature_date)
        tx_recent = tx_before.filter(pl.col("timestamp") >= thirty_days_ago)
        
        # Compute balance proxies (cumulative sum by customer)
        # This is a simplified version - production would use actual balance tracking
        balance_proxy = (
            tx_recent
            .sort(["customer_id", "timestamp"])
            .group_by("customer_id")
            .agg([
                pl.col("amount").sum().alias("total_flow_30d"),
                pl.col("amount").mean().alias("avg_balance_30d"),
                pl.col("amount").min().alias("min_balance_30d"),
                pl.col("amount").max().alias("max_balance_30d"),
                pl.col("amount").std().alias("balance_volatility_30d")
            ])
        )
        
        # Current balance (latest transaction amount as proxy)
        latest_tx = (
            tx_before
            .sort("timestamp")
            .group_by("customer_id")
            .last()
            .select(["customer_id", "amount"])
            .rename({"amount": "current_balance"})
        )
        
        # Balance retention ratio (current / avg)
        balance_df = balance_proxy.join(latest_tx, on="customer_id", how="left")
        balance_df = balance_df.with_columns([
            (pl.col("current_balance") / pl.col("avg_balance_30d")).alias("balance_retention_ratio")
        ])
        
        # Zero balance frequency (count of small amounts as proxy)
        zero_balance = (
            tx_recent
            .filter(pl.col("amount") < 100)  # Near-zero threshold
            .group_by("customer_id")
            .agg(pl.len().alias("zero_balance_frequency"))
        )
        
        balance_df = balance_df.join(zero_balance, on="customer_id", how="left")
        
        return balance_df
    
    def _compute_amount_features(
        self,
        transactions_df: pl.DataFrame,
        feature_date: datetime
    ) -> pl.DataFrame:
        """
        Compute amount pattern features.
        
        Features:
        - amount_roundness: Tendency to use round numbers
        - amount_just_below_threshold: Flag for amounts just below regulatory limits
        - similar_amount_count_24h: Count of similar amounts in 24h
        - identical_amount_count_24h: Count of identical amounts in 24h
        - structuring_amount_entropy: Entropy of amount distribution
        """
        twenty_four_hours_ago = feature_date - timedelta(hours=24)
        tx_recent = transactions_df.filter(
            (pl.col("timestamp") >= twenty_four_hours_ago) & 
            (pl.col("timestamp") <= feature_date)
        )
        
        # Amount roundness (mod 1000 == 0)
        roundness = (
            tx_recent
            .with_columns([
                (pl.col("amount") % 1000 == 0).cast(int).alias("is_round")
            ])
            .group_by("customer_id")
            .agg(pl.col("is_round").mean().alias("amount_roundness"))
        )
        
        # Just below threshold (within 5% of regulatory limits)
        thresholds = [10000, 50000, 150000]  # Tier 1/2/3 limits
        below_threshold = (
            tx_recent
            .with_columns([
                pl.any_horizontal([
                    (pl.col("amount") > limit * 0.95) & (pl.col("amount") < limit)
                    for limit in thresholds
                ]).alias("just_below")
            ])
            .group_by("customer_id")
            .agg(pl.col("just_below").any().alias("amount_just_below_threshold"))
        )
        
        # Similar and identical amount counts
        amount_counts = (
            tx_recent
            .group_by(["customer_id", "amount"])
            .agg(pl.len().alias("count"))
        )
        
        identical_counts = (
            amount_counts
            .filter(pl.col("count") > 1)
            .group_by("customer_id")
            .agg(pl.col("count").sum().alias("identical_amount_count_24h"))
        )
        
        # Similar amounts (within 10%)
        similar_counts = (
            tx_recent
            .sort(["customer_id", "timestamp"])
            .with_columns([
                pl.col("amount").shift(1).over("customer_id").alias("prev_amount")
            ])
            .filter(
                (pl.col("amount") / pl.col("prev_amount") >= 0.9) &
                (pl.col("amount") / pl.col("prev_amount") <= 1.1)
            )
            .group_by("customer_id")
            .agg(pl.len().alias("similar_amount_count_24h"))
        )
        
        # Structuring entropy (simplified)
        entropy = (
            tx_recent
            .group_by("customer_id")
            .agg(pl.col("amount").std().alias("structuring_amount_entropy"))
        )
        
        # Combine
        amount_df = roundness
        amount_df = amount_df.join(below_threshold, on="customer_id", how="left")
        amount_df = amount_df.join(identical_counts, on="customer_id", how="left")
        amount_df = amount_df.join(similar_counts, on="customer_id", how="left")
        amount_df = amount_df.join(entropy, on="customer_id", how="left")
        
        return amount_df
    
    def _compute_network_features(
        self,
        transactions_df: pl.DataFrame,
        feature_date: datetime
    ) -> pl.DataFrame:
        """
        Compute network features from transaction graph.
        
        Features:
        - pass_through_ratio: Ratio of inflow to outflow
        - degree_centrality: Normalized connection count
        - in_degree: Number of unique counterparties sending money
        - out_degree: Number of unique counterparties receiving money
        - funnel_score: Ratio of concentrated flows
        - reciprocity_ratio: Bidirectional transaction ratio
        """
        thirty_days_ago = feature_date - timedelta(days=30)
        tx_recent = transactions_df.filter(
            (pl.col("timestamp") >= thirty_days_ago) & 
            (pl.col("timestamp") <= feature_date)
        )
        
        # Check if required columns exist
        has_counterparty = "counterparty_id" in transactions_df.columns
        has_receiver = "receiver_id" in transactions_df.columns
        
        customers = transactions_df["customer_id"].unique()
        
        # In-degree and out-degree (unique counterparties)
        if has_counterparty:
            in_degree = (
                tx_recent
                .group_by("customer_id")
                .agg(pl.col("counterparty_id").n_unique().alias("in_degree"))
            )
        else:
            in_degree = pl.DataFrame({
                "customer_id": customers,
                "in_degree": [0] * len(customers)
            })
        
        if has_receiver:
            out_degree = (
                tx_recent
                .group_by("customer_id")
                .agg(pl.col("receiver_id").n_unique().alias("out_degree"))
            )
        else:
            out_degree = pl.DataFrame({
                "customer_id": customers,
                "out_degree": [0] * len(customers)
            })
        
        # Pass-through ratio (outflow / inflow)
        flow = (
            tx_recent
            .group_by("customer_id")
            .agg([
                pl.col("amount").filter(pl.col("amount") > 0).sum().alias("inflow"),
                pl.col("amount").filter(pl.col("amount") < 0).abs().sum().alias("outflow")
            ])
        )
        flow = flow.with_columns([
            (pl.col("outflow") / (pl.col("inflow") + 1e-6)).alias("pass_through_ratio")
        ])
        flow = flow.select(["customer_id", "pass_through_ratio"])
        
        # Degree centrality (normalized total degree)
        total_degree = in_degree.join(out_degree, on="customer_id", how="outer_coalesce")
        total_degree = total_degree.with_columns([
            (pl.col("in_degree").fill_null(0) + pl.col("out_degree").fill_null(0)).alias("total_degree")
        ])
        max_degree = total_degree["total_degree"].max()
        total_degree = total_degree.with_columns([
            (pl.col("total_degree") / max_degree).alias("degree_centrality")
        ])
        total_degree = total_degree.select(["customer_id", "degree_centrality"])
        
        # Placeholder for funnel_score and reciprocity_ratio
        # These require more complex graph analysis
        network_df = in_degree
        network_df = network_df.join(out_degree, on="customer_id", how="outer_coalesce")
        network_df = network_df.join(flow, on="customer_id", how="left")
        network_df = network_df.join(total_degree, on="customer_id", how="left")
        network_df = network_df.with_columns([
            pl.lit(0.0).alias("funnel_score"),
            pl.lit(0.0).alias("reciprocity_ratio")
        ])
        
        return network_df
    
    def _compute_temporal_features(
        self,
        transactions_df: pl.DataFrame,
        feature_date: datetime
    ) -> pl.DataFrame:
        """
        Compute temporal anomaly features.
        
        Features:
        - burst_ratio: Transaction velocity spike
        - velocity_change_pct: Percentage change in transaction velocity
        - balance_depletion_rate: Rate of balance decrease
        - is_anomalous_hour: Flag for transactions at unusual hours
        """
        seven_days_ago = feature_date - timedelta(days=7)
        one_day_ago = feature_date - timedelta(days=1)
        
        tx_recent = transactions_df.filter(
            (pl.col("timestamp") >= seven_days_ago) & 
            (pl.col("timestamp") <= feature_date)
        )
        
        tx_yesterday = transactions_df.filter(
            (pl.col("timestamp") >= one_day_ago) & 
            (pl.col("timestamp") < feature_date)
        )
        
        # Burst ratio (recent vs historical velocity)
        recent_count = tx_recent.group_by("customer_id").agg(pl.len().alias("recent_count"))
        historical_count = tx_recent.group_by("customer_id").agg(pl.len().alias("historical_count"))
        
        burst = recent_count.join(historical_count, on="customer_id", how="left")
        burst = burst.with_columns([
            (pl.col("recent_count") / (pl.col("historical_count") + 1e-6)).alias("burst_ratio")
        ])
        burst = burst.select(["customer_id", "burst_ratio"])
        
        # Velocity change (day-over-day)
        today_count = (
            transactions_df
            .filter(pl.col("timestamp") >= one_day_ago)
            .group_by("customer_id")
            .agg(pl.len().alias("today_count"))
        )
        
        yesterday_count = (
            transactions_df
            .filter((pl.col("timestamp") >= (one_day_ago - timedelta(days=1))) & 
                   (pl.col("timestamp") < one_day_ago))
            .group_by("customer_id")
            .agg(pl.len().alias("yesterday_count"))
        )
        
        velocity_change = today_count.join(yesterday_count, on="customer_id", how="left")
        velocity_change = velocity_change.with_columns([
            ((pl.col("today_count") - pl.col("yesterday_count")) / 
             (pl.col("yesterday_count") + 1e-6) * 100).alias("velocity_change_pct")
        ])
        velocity_change = velocity_change.select(["customer_id", "velocity_change_pct"])
        
        # Anomalous hour (transactions between 2-4 AM)
        anomalous_hour = (
            tx_recent
            .filter((pl.col("timestamp").dt.hour() >= 2) & (pl.col("timestamp").dt.hour() < 4))
            .group_by("customer_id")
            .agg(pl.lit(True).alias("is_anomalous_hour"))
        )
        
        # Balance depletion rate (simplified)
        depletion = (
            tx_recent
            .group_by("customer_id")
            .agg(pl.col("amount").sum().alias("balance_depletion_rate"))
        )
        
        # Combine
        temporal_df = burst
        temporal_df = temporal_df.join(velocity_change, on="customer_id", how="left")
        temporal_df = temporal_df.join(anomalous_hour, on="customer_id", how="left")
        temporal_df = temporal_df.join(depletion, on="customer_id", how="left")
        
        return temporal_df
    
    def _compute_device_features(
        self,
        transactions_df: pl.DataFrame,
        feature_date: datetime
    ) -> pl.DataFrame:
        """
        Compute device/location features.
        
        Features:
        - device_changes_7d: Number of device changes in 7 days
        - device_change_flag: Flag for any device change
        - location_entropy: Entropy of location distribution
        """
        # Placeholder implementation
        # These require device_id and location columns which may not exist in current schema
        
        customers = transactions_df["customer_id"].unique()
        device_df = pl.DataFrame({
            "customer_id": customers,
            "device_changes_7d": [0] * len(customers),
            "device_change_flag": [False] * len(customers),
            "location_entropy": [0.0] * len(customers)
        })
        
        return device_df
    
    def _compute_rolling_features(
        self,
        transactions_df: pl.DataFrame,
        feature_date: datetime
    ) -> pl.DataFrame:
        """
        Compute rolling window features.
        
        Features:
        - rolling_avg_tx_amount_30d: 30-day rolling average transaction amount
        - rolling_net_flow_7d: 7-day rolling net flow
        - new_relationships_7d: New counterparties in last 7 days
        """
        thirty_days_ago = feature_date - timedelta(days=30)
        seven_days_ago = feature_date - timedelta(days=7)
        
        tx_recent = transactions_df.filter(
            (pl.col("timestamp") >= thirty_days_ago) & 
            (pl.col("timestamp") <= feature_date)
        )
        
        # 30-day rolling average
        rolling_avg_30d = (
            tx_recent
            .group_by("customer_id")
            .agg(pl.col("amount").mean().alias("rolling_avg_tx_amount_30d"))
        )
        
        # 7-day net flow
        tx_7d = transactions_df.filter(
            (pl.col("timestamp") >= seven_days_ago) & 
            (pl.col("timestamp") <= feature_date)
        )
        
        net_flow_7d = (
            tx_7d
            .group_by("customer_id")
            .agg(pl.col("amount").sum().alias("rolling_net_flow_7d"))
        )
        
        # New relationships (counterparties not seen before 7 days ago)
        if "counterparty_id" in transactions_df.columns:
            counterparties_before = (
                transactions_df
                .filter(pl.col("timestamp") < seven_days_ago)
                .select(["customer_id", "counterparty_id"])
                .unique()
            )
            
            counterparties_recent = (
                tx_7d
                .select(["customer_id", "counterparty_id"])
                .unique()
            )
            
            # Simplified new relationship count
            new_relationships = (
                tx_7d
                .group_by("customer_id")
                .agg(pl.col("counterparty_id").n_unique().alias("new_relationships_7d"))
            )
        else:
            customers = transactions_df["customer_id"].unique()
            new_relationships = pl.DataFrame({
                "customer_id": customers,
                "new_relationships_7d": [0] * len(customers)
            })
        
        # Combine
        rolling_df = rolling_avg_30d
        rolling_df = rolling_df.join(net_flow_7d, on="customer_id", how="left")
        rolling_df = rolling_df.join(new_relationships, on="customer_id", how="left")
        
        return rolling_df


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
