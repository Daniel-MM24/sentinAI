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
    """Essential customer profile with tier and archetype data."""
    customer_id: str
    kyc_tier: KYCTier
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
    
    def _assign_kyc_tier(self, n: int) -> List[KYCTier]:
        """
        Assign KYC tiers.
        
        Args:
            n: Number of customers to assign tiers to
            
        Returns:
            List of KYC tiers
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

        return kyc_tiers

    
    def generate_profiles(self) -> List[CustomerProfile]:
        """
        Generate complete stratified customer profiles.
        
        Returns:
            List of CustomerProfile objects
        """
        logger.info(f"Generating {self.config.num_customers} stratified customer profiles")
        
        n = self.config.num_customers

        # Assign KYC tiers
        kyc_tiers = self._assign_kyc_tier(n)
        
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
            "tier": [p.kyc_tier.value for p in profiles],
            "archetype": [p.archetype.value for p in profiles],
        }

        df = pl.DataFrame(data)

        # Set proper data types
        schema = {
            "customer_id": pl.String,
            "tier": pl.String,
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
    print(f"\nKYC tier distribution:\n{df['tier'].value_counts()}")
    print(f"\nArchetype distribution:\n{df['archetype'].value_counts()}")
    
    # Save to CSV
    output_path = Path("/home/dan/project/sentinAI/data/bronze/customers/customer_profiles.csv")
    generator.save_to_csv(df, output_path)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
