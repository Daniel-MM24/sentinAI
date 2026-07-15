"""
Stratified Customer Profile Generation for AML Simulation.

This module implements comprehensive customer profile generation with:
- KYC tier assignment with regulatory limits
- Archetype-based behavioral profiling for MRM downscaling

MRM Compliance:
- Deterministic seed configuration for reproducibility
- Statistical distributions matching Kenyan demographic data
- Regulatory tier compliance (CBK guidelines)
"""

import logging
import numpy as np
import polars as pl
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class CustomerArchetype(str, Enum):
    """Customer behavioral archetypes with transaction velocity patterns (MRM downscaling parameters)."""
    RETAIL_HEAVY = "retail_heavy"  # 15%: High transaction velocity, 850-950 tx/year
    RETAIL_STANDARD = "retail_standard"  # 70%: Moderate activity, 300-500 tx/year
    MICRO_MERCHANT = "micro_merchant"  # 12%: Business patterns, 600-800 tx/year
    CORPORATE_SME = "corporate_sme"  # 3%: Corporate/SME patterns, high-value tx


class KYCTier(str, Enum):
    """KYC tiers with regulatory transaction and balance limits (CBK PG/43 guidelines)."""
    TIER_1 = "tier_1"  # Basic: 60% - Tx ≤ KES 10,000, Balance ≤ KES 50,000
    TIER_2 = "tier_2"  # Interim: 20% - Tx ≤ KES 50,000, Balance ≤ KES 200,000
    TIER_3 = "tier_3"  # Full KYC: 15% - Tx ≤ KES 150,000, Balance ≤ KES 1,000,000
    TIER_4 = "tier_4"  # EDD: 5% - Tx ≤ KES 500,000, Balance ≤ KES 5,000,000


@dataclass
class CustomerProfile:
    """Complete customer profile with tier, risk, and archetype data."""
    customer_id: str
    kyc_tier: KYCTier
    max_transaction_limit: float  # KES
    max_balance_limit: float  # KES
    account_age_days: int
    initial_balance: float
    archetype: CustomerArchetype


@dataclass
class StratifiedProfileConfig:
    """Configuration for stratified customer profile generation."""
    num_customers: int = 1000
    seed: int = 42

    # KYC tier distribution
    tier_1_pct: float = 0.60  # 60% Tier 1
    tier_2_pct: float = 0.20  # 20% Tier 2
    tier_3_pct: float = 0.15  # 15% Tier 3
    tier_4_pct: float = 0.05  # 5% Tier 4 (EDD)

    # Archetype distribution (MRM downscaling parameters)
    archetype_weights: tuple = (0.15, 0.70, 0.12, 0.03)  # retail_heavy, retail_standard, micro_merchant, corporate_sme


class StratifiedProfileGenerator:
    """
    Generates stratified customer profiles with KYC tier assignment and
    archetype-based behavioral profiling.
    """
    
    def __init__(self, config: Optional[StratifiedProfileConfig] = None):
        """
        Initialize the stratified profile generator.
        
        Args:
            config: StratifiedProfileConfig instance with generation parameters.
                   If None, uses default configuration.
        """
        self.config = config or StratifiedProfileConfig()
        self._rng = np.random.default_rng(self.config.seed)

        logger.info(
            f"Initialized StratifiedProfileGenerator with {self.config.num_customers} customers, "
            f"seed={self.config.seed}"
        )
    
    def _assign_kyc_tier(self, n: int) -> Tuple[List[KYCTier], List[float], List[float]]:
        """
        Assign KYC tiers with regulatory limits.
        
        Args:
            n: Number of customers to assign tiers to
            
        Returns:
            Tuple of (kyc_tiers, max_transaction_limits, max_balance_limits)
        """
        tier_probs = [
            self.config.tier_1_pct,
            self.config.tier_2_pct,
            self.config.tier_3_pct,
            self.config.tier_4_pct,
        ]

        tier_indices = self._rng.choice(
            [0, 1, 2, 3],  # Indices for TIER_1, TIER_2, TIER_3, TIER_4
            size=n,
            p=tier_probs,
        )

        kyc_tiers = [
            KYCTier.TIER_1 if idx == 0 else
            KYCTier.TIER_2 if idx == 1 else
            KYCTier.TIER_3 if idx == 2 else
            KYCTier.TIER_4
            for idx in tier_indices
        ]

        # Assign limits based on tier index (CBK PG/43 guidelines)
        tx_limits = [
            10_000.0 if idx == 0 else
            50_000.0 if idx == 1 else
            150_000.0 if idx == 2 else
            500_000.0  # TIER_4 (EDD)
            for idx in tier_indices
        ]

        balance_limits = [
            50_000.0 if idx == 0 else
            200_000.0 if idx == 1 else
            1_000_000.0 if idx == 2 else
            5_000_000.0  # TIER_4 (EDD)
            for idx in tier_indices
        ]

        return kyc_tiers, tx_limits, balance_limits

    def _generate_account_age(self, n: int) -> List[int]:
        """
        Generate account age in days (30 to 365 days).
        
        Args:
            n: Number of samples to generate
            
        Returns:
            List of account age values
        """
        return [int(self._rng.integers(30, 366)) for _ in range(n)]
    
    def _generate_initial_balance(self, n: int, balance_limits: List[float]) -> List[float]:
        """
        Generate initial balance within tier limits.
        
        Args:
            n: Number of samples to generate
            balance_limits: List of maximum balance limits per customer
            
        Returns:
            List of initial balance values
        """
        # Generate balances as percentage of limit (log-normal distribution)
        balance_pct = self._rng.lognormal(mean=-1.0, sigma=0.5, size=n)
        balance_pct = np.clip(balance_pct, 0.01, 0.95)  # 1% to 95% of limit
        
        initial_balances = [float(balance_pct[i] * balance_limits[i]) for i in range(n)]
        return initial_balances
    
    def generate_profiles(self) -> List[CustomerProfile]:
        """
        Generate complete stratified customer profiles.
        
        Returns:
            List of CustomerProfile objects
        """
        logger.info(f"Generating {self.config.num_customers} stratified customer profiles")
        
        n = self.config.num_customers

        # Assign KYC tiers with limits
        kyc_tiers, tx_limits, balance_limits = self._assign_kyc_tier(n)

        # Generate account attributes
        account_ages = self._generate_account_age(n)
        initial_balances = self._generate_initial_balance(n, balance_limits)
        
        # Assign archetypes based on MRM downscaling parameters
        archetype_enum_list = list(CustomerArchetype)
        archetype_weights = self.config.archetype_weights
        # Use numpy random choice for reproducibility via self._rng
        indices = self._rng.choice(len(archetype_enum_list), size=n, p=archetype_weights)
        archetypes = [archetype_enum_list[idx] for idx in indices]

        # Create customer profiles
        profiles = []
        for i in range(n):
            profile = CustomerProfile(
                customer_id=f"CUST_{i:06d}",
                kyc_tier=kyc_tiers[i],
                max_transaction_limit=tx_limits[i],
                max_balance_limit=balance_limits[i],
                account_age_days=account_ages[i],
                initial_balance=initial_balances[i],
                archetype=archetypes[i]
            )
            profiles.append(profile)

        logger.info(
            f"Generated {len(profiles)} customer profiles. "
            f"KYC tiers: T1={np.mean([p.kyc_tier == KYCTier.TIER_1 for p in profiles]):.2%}, "
            f"T2={np.mean([p.kyc_tier == KYCTier.TIER_2 for p in profiles]):.2%}, "
            f"T3={np.mean([p.kyc_tier == KYCTier.TIER_3 for p in profiles]):.2%}, "
            f"T4={np.mean([p.kyc_tier == KYCTier.TIER_4 for p in profiles]):.2%}. "
            f"Archetypes: RH={np.mean([p.archetype == CustomerArchetype.RETAIL_HEAVY for p in profiles]):.2%}, "
            f"RS={np.mean([p.archetype == CustomerArchetype.RETAIL_STANDARD for p in profiles]):.2%}, "
            f"MM={np.mean([p.archetype == CustomerArchetype.MICRO_MERCHANT for p in profiles]):.2%}, "
            f"CS={np.mean([p.archetype == CustomerArchetype.CORPORATE_SME for p in profiles]):.2%}."
        )
        
        return profiles
    
    def profiles_to_dataframe(self, profiles: List[CustomerProfile]) -> pl.DataFrame:
        """
        Convert customer profiles to Polars DataFrame.
        
        Args:
            profiles: List of CustomerProfile objects
            
        Returns:
            Polars DataFrame with customer profile data
        """
        data = {
            "customer_id": [p.customer_id for p in profiles],
            "kyc_tier": [p.kyc_tier.value if isinstance(p.kyc_tier, KYCTier) else str(p.kyc_tier) for p in profiles],
            "max_transaction_limit_kes": [p.max_transaction_limit for p in profiles],
            "max_balance_limit_kes": [p.max_balance_limit for p in profiles],
            "account_age_days": [p.account_age_days for p in profiles],
            "initial_balance_kes": [p.initial_balance for p in profiles],
            "archetype": [p.archetype.value if isinstance(p.archetype, CustomerArchetype) else str(p.archetype) for p in profiles],
        }

        df = pl.DataFrame(data)

        # Set proper data types
        schema = {
            "customer_id": pl.String,
            "kyc_tier": pl.String,
            "max_transaction_limit_kes": pl.Float64,
            "max_balance_limit_kes": pl.Float64,
            "account_age_days": pl.Int32,
            "initial_balance_kes": pl.Float64,
            "archetype": pl.String,
        }
        
        df = df.cast(schema)
        
        return df
    
    def save_to_csv(self, df: pl.DataFrame, output_path: Path) -> None:
        """
        Save customer profiles DataFrame to CSV.

        Args:
            df: Polars DataFrame with customer profiles
            output_path: Path to save the CSV file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.write_csv(output_path)
        logger.info(f"Saved customer profiles to {output_path}")


def main():
    """Example usage of the StratifiedProfileGenerator."""
    logging.basicConfig(level=logging.INFO)
    
    # Create generator with custom configuration
    config = StratifiedProfileConfig(
        num_customers=1000,
        seed=42
    )
    
    generator = StratifiedProfileGenerator(config)
    
    # Generate profiles
    profiles = generator.generate_profiles()
    
    # Convert to DataFrame
    df = generator.profiles_to_dataframe(profiles)
    
    # Display summary
    print("\n=== Customer Profiles Generated ===")
    print(f"Shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    print(f"\n=== Distribution Summary ===")
    print(f"\nKYC tier distribution:\n{df['kyc_tier'].value_counts()}")
    print(f"\nArchetype distribution:\n{df['archetype'].value_counts()}")
    
    # Save to CSV
    output_path = Path("/home/dan/project/sentinAI/data/bronze/customers/customer_profiles_complete.csv")
    generator.save_to_csv(df, output_path)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
