"""
Financial Anomaly Injector for Synthetic Data.

This module injects structural anomalies into clean financial data at a controlled
ratio (0.015 = 1.5% of total dataset). It accepts pristine data and embeds
realistic anomaly patterns, then appends tracking columns for ground-truth labels.

CRITICAL: This module ONLY adds anomaly_flag and anomaly_type columns AFTER
injecting anomalies. The injector is responsible for:
1. Accepting clean data (no anomaly flags)
2. Injecting anomalies at strict 0.015 ratio
3. Appending anomaly_flag (1 for anomalous, 0 for clean)
4. Appending anomaly_type (categorical string describing anomaly type)

The injector creates the ground-truth targets for downstream ML verification.
"""

import logging
import numpy as np
import polars as pl
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Enumeration of financial anomaly types."""
    AMOUNT_SPIKE = "amount_spike"
    VELOCITY_SURGE = "velocity_surge"
    BALANCE_DEPLETION = "balance_depletion"
    PRICE_MANIPULATION = "price_manipulation"
    LIQUIDITY_ANOMALY = "liquidity_anomaly"
    SPREAD_ABNORMALITY = "spread_abnormality"
    COUNTERPARTY_RISK = "counterparty_risk"
    TEMPORAL_PATTERN = "temporal_pattern"


@dataclass
class InjectorConfig:
    """Configuration for the FinancialAnomalyInjector."""
    anomaly_ratio: float = 0.015  # 1.5% of records should be anomalous
    seed: int = 42
    amount_spike_multiplier: float = 10.0  # Multiplier for amount spikes
    velocity_threshold: float = 5.0  # Std devs above mean for velocity
    balance_depletion_threshold: float = 0.9  # 90% balance depletion
    price_impact_threshold: float = 0.08  # 8% price impact
    liquidity_threshold: float = 0.1  # Below 10% liquidity
    spread_multiplier: float = 5.0  # Multiplier for spread anomalies


class FinancialAnomalyInjector:
    """
    Injects structural anomalies into clean financial data.
    
    This class accepts pristine financial data and embeds realistic anomaly
    patterns at a strict ratio of 0.015 (1.5% of total dataset). After injection,
    it appends tracking columns (anomaly_flag, anomaly_type) for ground-truth labels.
    
    The injector implements various anomaly types:
    - Amount spikes: Transactions with abnormally high amounts
    - Velocity surges: Unusual transaction frequency patterns
    - Balance depletion: Rapid account balance depletion
    - Price manipulation: Abnormal price impact patterns
    - Liquidity anomalies: Unusual liquidity scores
    - Spread abnormalities: Abnormal bid-ask spreads
    - Counterparty risk: High-risk counterparty patterns
    - Temporal patterns: Unusual timing patterns
    
    All operations use vectorized Polars expressions for performance.
    """
    
    def __init__(self, config: Optional[InjectorConfig] = None):
        """
        Initialize the FinancialAnomalyInjector.
        
        Args:
            config: InjectorConfig instance with injection parameters.
                   If None, uses default configuration.
        """
        self.config = config or InjectorConfig()
        self._rng = np.random.default_rng(self.config.seed)
        
        logger.info(
            f"Initialized FinancialAnomalyInjector with anomaly_ratio={self.config.anomaly_ratio}, "
            f"seed={self.config.seed}"
        )
    
    def _inject_amount_spikes(
        self, 
        df: pl.DataFrame, 
        anomaly_indices: np.ndarray
    ) -> pl.DataFrame:
        """
        Inject amount spike anomalies.
        
        Args:
            df: Input DataFrame
            anomaly_indices: Indices where to inject anomalies
            
        Returns:
            DataFrame with amount spike anomalies injected
        """
        n_anomalies = len(anomaly_indices)
        
        # Apply multiplier to transaction amounts
        spike_amounts = df["transaction_amount"].to_numpy()[anomaly_indices]
        spiked_amounts = spike_amounts * self.config.amount_spike_multiplier
        
        # Update the DataFrame using vectorized operations
        df = df.with_columns([
            pl.when(pl.col("transaction_amount").is_in(spike_amounts))
            .then(pl.col("transaction_amount") * self.config.amount_spike_multiplier)
            .otherwise(pl.col("transaction_amount"))
            .alias("transaction_amount")
        ])
        
        # More direct approach: create a mask and update
        mask = np.zeros(len(df), dtype=bool)
        mask[anomaly_indices] = True
        
        amounts = df["transaction_amount"].to_numpy().copy()
        amounts[mask] = amounts[mask] * self.config.amount_spike_multiplier
        
        df = df.with_columns([
            pl.Series("transaction_amount", amounts)
        ])
        
        logger.info(f"Injected {n_anomalies} amount spike anomalies")
        return df
    
    def _inject_velocity_surges(
        self, 
        df: pl.DataFrame, 
        anomaly_indices: np.ndarray
    ) -> pl.DataFrame:
        """
        Inject velocity surge anomalies.
        
        Args:
            df: Input DataFrame
            anomaly_indices: Indices where to inject anomalies
            
        Returns:
            DataFrame with velocity surge anomalies injected
        """
        n_anomalies = len(anomaly_indices)
        
        # Calculate mean and std of transaction counts
        mean_count = df["transaction_count"].mean()
        std_count = df["transaction_count"].std()
        
        # Set transaction counts to extreme values
        mask = np.zeros(len(df), dtype=bool)
        mask[anomaly_indices] = True
        
        transaction_counts = df["transaction_count"].to_numpy().copy()
        transaction_counts[mask] = mean_count + self.config.velocity_threshold * std_count
        
        df = df.with_columns([
            pl.Series("transaction_count", transaction_counts.astype(int))
        ])
        
        logger.info(f"Injected {n_anomalies} velocity surge anomalies")
        return df
    
    def _inject_balance_depletion(
        self, 
        df: pl.DataFrame, 
        anomaly_indices: np.ndarray
    ) -> pl.DataFrame:
        """
        Inject balance depletion anomalies.
        
        Args:
            df: Input DataFrame
            anomaly_indices: Indices where to inject anomalies
            
        Returns:
            DataFrame with balance depletion anomalies injected
        """
        n_anomalies = len(anomaly_indices)
        
        # Set balances to very low values (near depletion)
        mask = np.zeros(len(df), dtype=bool)
        mask[anomaly_indices] = True
        
        balances = df["account_balance"].to_numpy().copy()
        balances[mask] = balances[mask] * (1 - self.config.balance_depletion_threshold)
        
        df = df.with_columns([
            pl.Series("account_balance", balances)
        ])
        
        logger.info(f"Injected {n_anomalies} balance depletion anomalies")
        return df
    
    def _inject_price_manipulation(
        self, 
        df: pl.DataFrame, 
        anomaly_indices: np.ndarray
    ) -> pl.DataFrame:
        """
        Inject price manipulation anomalies.
        
        Args:
            df: Input DataFrame
            anomaly_indices: Indices where to inject anomalies
            
        Returns:
            DataFrame with price manipulation anomalies injected
        """
        n_anomalies = len(anomaly_indices)
        
        # Set price impact to extreme values
        mask = np.zeros(len(df), dtype=bool)
        mask[anomaly_indices] = True
        
        price_impacts = df["price_impact"].to_numpy().copy()
        # Alternate between positive and negative extreme impacts
        signs = self._rng.choice([-1, 1], size=n_anomalies)
        price_impacts[mask] = signs * self.config.price_impact_threshold
        
        df = df.with_columns([
            pl.Series("price_impact", price_impacts)
        ])
        
        logger.info(f"Injected {n_anomalies} price manipulation anomalies")
        return df
    
    def _inject_liquidity_anomalies(
        self, 
        df: pl.DataFrame, 
        anomaly_indices: np.ndarray
    ) -> pl.DataFrame:
        """
        Inject liquidity anomalies.
        
        Args:
            df: Input DataFrame
            anomaly_indices: Indices where to inject anomalies
            
        Returns:
            DataFrame with liquidity anomalies injected
        """
        n_anomalies = len(anomaly_indices)
        
        # Set liquidity scores to very low values
        mask = np.zeros(len(df), dtype=bool)
        mask[anomaly_indices] = True
        
        liquidity_scores = df["liquidity_score"].to_numpy().copy()
        liquidity_scores[mask] = self._rng.uniform(
            0.0, 
            self.config.liquidity_threshold, 
            size=n_anomalies
        )
        
        df = df.with_columns([
            pl.Series("liquidity_score", liquidity_scores)
        ])
        
        logger.info(f"Injected {n_anomalies} liquidity anomalies")
        return df
    
    def _inject_spread_abnormalities(
        self, 
        df: pl.DataFrame, 
        anomaly_indices: np.ndarray
    ) -> pl.DataFrame:
        """
        Inject spread abnormality anomalies.
        
        Args:
            df: Input DataFrame
            anomaly_indices: Indices where to inject anomalies
            
        Returns:
            DataFrame with spread abnormality anomalies injected
        """
        n_anomalies = len(anomaly_indices)
        
        # Multiply spreads by extreme factor
        mask = np.zeros(len(df), dtype=bool)
        mask[anomaly_indices] = True
        
        spreads = df["bid_ask_spread"].to_numpy().copy()
        spreads[mask] = spreads[mask] * self.config.spread_multiplier
        
        df = df.with_columns([
            pl.Series("bid_ask_spread", spreads)
        ])
        
        logger.info(f"Injected {n_anomalies} spread abnormality anomalies")
        return df
    
    def _inject_counterparty_risk(
        self, 
        df: pl.DataFrame, 
        anomaly_indices: np.ndarray
    ) -> pl.DataFrame:
        """
        Inject counterparty risk anomalies.
        
        Args:
            df: Input DataFrame
            anomaly_indices: Indices where to inject anomalies
            
        Returns:
            DataFrame with counterparty risk anomalies injected
        """
        n_anomalies = len(anomaly_indices)
        
        # Set counterparty risk tier to High
        mask = np.zeros(len(df), dtype=bool)
        mask[anomaly_indices] = True
        
        risk_tiers = df["counterparty_risk_tier"].to_numpy().copy()
        risk_tiers[mask] = "High"
        
        df = df.with_columns([
            pl.Series("counterparty_risk_tier", risk_tiers)
        ])
        
        logger.info(f"Injected {n_anomalies} counterparty risk anomalies")
        return df
    
    def _inject_temporal_patterns(
        self, 
        df: pl.DataFrame, 
        anomaly_indices: np.ndarray
    ) -> pl.DataFrame:
        """
        Inject temporal pattern anomalies (unusual timing).
        
        Args:
            df: Input DataFrame
            anomaly_indices: Indices where to inject anomalies
            
        Returns:
            DataFrame with temporal pattern anomalies injected
        """
        n_anomalies = len(anomaly_indices)
        
        # Shift timestamps to unusual hours (e.g., 2 AM - 4 AM)
        mask = np.zeros(len(df), dtype=bool)
        mask[anomaly_indices] = True
        
        timestamps = df["timestamp"].to_numpy().copy()
        # Shift by 6-8 hours to unusual times
        shift_hours = self._rng.integers(6, 9, size=n_anomalies)
        
        for i, idx in enumerate(anomaly_indices):
            timestamps[idx] = timestamps[idx] + np.timedelta64(shift_hours[i], 'h')
        
        df = df.with_columns([
            pl.Series("timestamp", timestamps)
        ])
        
        logger.info(f"Injected {n_anomalies} temporal pattern anomalies")
        return df
    
    def inject(self, clean_df: pl.DataFrame) -> pl.DataFrame:
        """
        Inject anomalies into clean financial data.
        
        This method:
        1. Calculates the number of anomalies to inject based on ratio
        2. Randomly selects indices for anomaly injection
        3. Applies various anomaly types to selected records
        4. Appends anomaly_flag and anomaly_type columns
        
        Args:
            clean_df: Polars DataFrame with clean financial data (no anomaly flags)
            
        Returns:
            Polars DataFrame with anomalies injected and tracking columns added:
            - All original columns (modified where anomalies were injected)
            - anomaly_flag: 1 for anomalous, 0 for clean (int32)
            - anomaly_type: Categorical string describing anomaly type
            
        Raises:
            ValueError: If input DataFrame is empty or ratio is invalid
        """
        logger.info("Starting anomaly injection process")
        
        # Validate input
        if len(clean_df) == 0:
            raise ValueError("Input DataFrame cannot be empty")
        
        if not (0 < self.config.anomaly_ratio <= 1):
            raise ValueError("anomaly_ratio must be between 0 and 1")
        
        # Calculate number of anomalies to inject
        n_total = len(clean_df)
        n_anomalies = int(n_total * self.config.anomaly_ratio)
        
        if n_anomalies == 0:
            logger.warning("Anomaly ratio too small for dataset size, no anomalies injected")
            # Still add tracking columns (all clean)
            df = clean_df.clone()
            df = df.with_columns([
                pl.lit(0, dtype=pl.Int32).alias("anomaly_flag"),
                pl.lit(None, dtype=pl.String).alias("anomaly_type")
            ])
            return df
        
        logger.info(f"Injecting {n_anomalies} anomalies ({n_anomalies/n_total:.2%} of dataset)")
        
        # Randomly select indices for anomaly injection
        anomaly_indices = self._rng.choice(
            n_total, 
            size=n_anomalies, 
            replace=False
        )
        anomaly_indices = np.sort(anomaly_indices)
        
        # Clone the DataFrame to avoid modifying the original
        df = clean_df.clone()
        
        # Distribute anomalies across different types
        anomaly_types = [
            AnomalyType.AMOUNT_SPIKE,
            AnomalyType.VELOCITY_SURGE,
            AnomalyType.BALANCE_DEPLETION,
            AnomalyType.PRICE_MANIPULATION,
            AnomalyType.LIQUIDITY_ANOMALY,
            AnomalyType.SPREAD_ABNORMALITY,
            AnomalyType.COUNTERPARTY_RISK,
            AnomalyType.TEMPORAL_PATTERN
        ]
        
        # Assign anomaly types to indices
        assigned_types = self._rng.choice(anomaly_types, size=n_anomalies)
        
        # Create arrays to track which records get which anomaly type
        anomaly_type_array = np.array([None] * n_total, dtype=object)
        
        # Inject anomalies by type
        for anomaly_type in anomaly_types:
            type_mask = assigned_types == anomaly_type
            type_indices = anomaly_indices[type_mask]
            
            if len(type_indices) == 0:
                continue
            
            # Mark the anomaly type
            for idx in type_indices:
                anomaly_type_array[idx] = anomaly_type.value
            
            # Inject the specific anomaly
            if anomaly_type == AnomalyType.AMOUNT_SPIKE:
                df = self._inject_amount_spikes(df, type_indices)
            elif anomaly_type == AnomalyType.VELOCITY_SURGE:
                df = self._inject_velocity_surges(df, type_indices)
            elif anomaly_type == AnomalyType.BALANCE_DEPLETION:
                df = self._inject_balance_depletion(df, type_indices)
            elif anomaly_type == AnomalyType.PRICE_MANIPULATION:
                df = self._inject_price_manipulation(df, type_indices)
            elif anomaly_type == AnomalyType.LIQUIDITY_ANOMALY:
                df = self._inject_liquidity_anomalies(df, type_indices)
            elif anomaly_type == AnomalyType.SPREAD_ABNORMALITY:
                df = self._inject_spread_abnormalities(df, type_indices)
            elif anomaly_type == AnomalyType.COUNTERPARTY_RISK:
                df = self._inject_counterparty_risk(df, type_indices)
            elif anomaly_type == AnomalyType.TEMPORAL_PATTERN:
                df = self._inject_temporal_patterns(df, type_indices)
        
        # Create anomaly_flag array
        anomaly_flags = np.zeros(n_total, dtype=np.int32)
        anomaly_flags[anomaly_indices] = 1
        
        # Convert anomaly_type_array to proper string type with null handling
        anomaly_type_list = [str(x) if x is not None else None for x in anomaly_type_array]
        
        # Append tracking columns
        df = df.with_columns([
            pl.Series("anomaly_flag", anomaly_flags, dtype=pl.Int32),
            pl.Series("anomaly_type", anomaly_type_list, dtype=pl.String)
        ])
        
        logger.info(
            f"Successfully injected {n_anomalies} anomalies. "
            f"Anomaly distribution: {dict(zip(*np.unique(anomaly_type_array[anomaly_indices], return_counts=True)))}"
        )
        
        return df
    
    def get_anomaly_summary(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for the injected anomalies.
        
        Args:
            df: Polars DataFrame with anomalies injected
            
        Returns:
            Dictionary containing anomaly summary statistics
        """
        n_total = len(df)
        n_anomalies = df.filter(pl.col("anomaly_flag") == 1).shape[0]
        
        anomaly_types_dist = (
            df.filter(pl.col("anomaly_flag") == 1)
            .group_by("anomaly_type")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
        
        summary = {
            "total_records": n_total,
            "total_anomalies": n_anomalies,
            "anomaly_ratio": n_anomalies / n_total if n_total > 0 else 0,
            "anomaly_type_distribution": anomaly_types_dist.to_dict(as_series=False)
        }
        
        return summary


def main():
    """Example usage of the FinancialAnomalyInjector."""
    logging.basicConfig(level=logging.INFO)
    
    # Import the clean data generator
    from src.data.synthetic_generator import CleanDataGenerator, GeneratorConfig
    
    # Generate clean data
    config = GeneratorConfig(num_records=10000, num_entities=500, seed=42)
    generator = CleanDataGenerator(config)
    clean_df = generator.generate()
    
    print("\n=== Clean Data ===")
    print(f"Shape: {clean_df.shape}")
    print(f"Columns: {clean_df.columns}")
    
    # Inject anomalies
    injector_config = InjectorConfig(anomaly_ratio=0.015, seed=42)
    injector = FinancialAnomalyInjector(injector_config)
    anomalous_df = injector.inject(clean_df)
    
    print("\n=== Anomalous Data ===")
    print(f"Shape: {anomalous_df.shape}")
    print(f"Columns: {anomalous_df.columns}")
    
    # Display anomaly summary
    summary = injector.get_anomaly_summary(anomalous_df)
    print(f"\n=== Anomaly Summary ===")
    print(f"Total Records: {summary['total_records']}")
    print(f"Total Anomalies: {summary['total_anomalies']}")
    print(f"Anomaly Ratio: {summary['anomaly_ratio']:.2%}")
    print(f"Anomaly Type Distribution: {summary['anomaly_type_distribution']}")
    
    # Display sample of anomalous records
    print(f"\n=== Sample Anomalous Records ===")
    print(anomalous_df.filter(pl.col("anomaly_flag") == 1).head())


if __name__ == "__main__":
    main()
