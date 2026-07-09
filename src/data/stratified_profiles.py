"""
Stratified Customer Profile Generation for AML Simulation.

This module implements comprehensive customer profile generation with:
- Demographic mixing based on FinAccess Survey 2024
- KYC tier assignment with regulatory limits
- High-risk entity assignment for transaction flagging

MRM Compliance:
- Deterministic seed configuration for reproducibility
- Statistical distributions matching Kenyan demographic data
- Regulatory tier compliance (CBK guidelines)
"""

import logging
import numpy as np
import polars as pl
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class Gender(str, Enum):
    """Gender distribution matching FinAccess Survey 2024."""
    MALE = "Male"
    FEMALE = "Female"


class KYCTier(str, Enum):
    """KYC tiers with regulatory transaction and balance limits (CBK guidelines)."""
    TIER_1 = "tier_1"  # Basic: 60% - Tx ≤ KES 70,000, Balance ≤ KES 300,000
    TIER_2 = "tier_2"  # Enhanced: 30% - Tx ≤ KES 150,000, Balance ≤ KES 500,000
    TIER_3 = "tier_3"  # Full KYC: 10% - Tx ≤ KES 250,000, Balance ≤ KES 500,000


class UrbanRuralClassification(str, Enum):
    """Urban/Rural classification for Kenyan counties."""
    URBAN = "urban"
    SEMI_URBAN = "semi_urban"
    RURAL = "rural"


@dataclass
class CountyProfile:
    """County profile with population weights and classification."""
    name: str
    population_weight: float
    classification: UrbanRuralClassification


@dataclass
class CustomerProfile:
    """Complete customer profile with demographic, tier, and risk data."""
    customer_id: str
    gender: Gender
    age: int
    county: str
    urban_rural_classification: UrbanRuralClassification
    kyc_tier: KYCTier
    max_transaction_limit: float  # KES
    max_balance_limit: float  # KES
    betting_platform_flag: bool
    international_transaction_flag: bool
    account_age_days: int
    initial_balance: float


@dataclass
class StratifiedProfileConfig:
    """Configuration for stratified customer profile generation."""
    num_customers: int = 1000
    seed: int = 42
    
    # Demographic parameters
    gender_male_pct: float = 0.49  # 49% Male, 51% Female
    age_mean: float = 32.0  # Normal distribution mean
    age_std: float = 10.0  # Normal distribution std
    age_min: int = 18  # Minimum age cap
    age_max: int = 80  # Maximum age cap
    
    # KYC tier distribution
    tier_1_pct: float = 0.60  # 60% Tier 1
    tier_2_pct: float = 0.30  # 30% Tier 2
    tier_3_pct: float = 0.10  # 10% Tier 3
    
    # High-risk entity flags
    betting_platform_pct: float = 0.02  # 2% betting platform users
    international_tx_pct: float = 0.03  # 3% international transaction users


class StratifiedProfileGenerator:
    """
    Generates stratified customer profiles with demographic mixing,
    KYC tier assignment, and high-risk entity flags.
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
        
        # Initialize county profiles based on Kenyan demographic data
        self._county_profiles = self._initialize_county_profiles()
        
        logger.info(
            f"Initialized StratifiedProfileGenerator with {self.config.num_customers} customers, "
            f"seed={self.config.seed}"
        )
    
    def _initialize_county_profiles(self) -> List[CountyProfile]:
        """
        Initialize county profiles with population weights and urban/rural classification.
        
        Based on Kenyan demographic data and urbanization patterns.
        """
        # Urban centers (major cities)
        urban_centers = [
            CountyProfile("Nairobi", 0.20, UrbanRuralClassification.URBAN),
            CountyProfile("Mombasa", 0.07, UrbanRuralClassification.URBAN),
            CountyProfile("Kisumu", 0.06, UrbanRuralClassification.URBAN),
        ]
        
        # Semi-urban centers
        semi_urban = [
            CountyProfile("Nakuru", 0.08, UrbanRuralClassification.SEMI_URBAN),
            CountyProfile("Kiambu", 0.12, UrbanRuralClassification.SEMI_URBAN),
            CountyProfile("Machakos", 0.05, UrbanRuralClassification.SEMI_URBAN),
        ]
        
        # Rural counties (remaining 44 counties weighted by approximate population)
        # Using simplified weights for demonstration
        rural_counties = [
            "Baringo", "Bomet", "Bungoma", "Busia", "Elgeyo Marakwet",
            "Embu", "Garissa", "Homa Bay", "Isiolo", "Kajiado",
            "Kakamega", "Kericho", "Kilifi", "Kirinyaga", "Kisii",
            "Kitui", "Kwale", "Laikipia", "Lamu", "Makueni",
            "Mandera", "Marsabit", "Meru", "Migori", "Murang'a",
            "Nairobi", "Nakuru", "Nandi", "Narok", "Nyamira",
            "Nyandarua", "Nyeri", "Samburu", "Siaya", "Taita Taveta",
            "Tana River", "Tharaka Nithi", "Trans Nzoia", "Turkana",
            "Uasin Gishu", "Vihiga", "Wajir", "West Pokot"
        ]
        
        # Remove counties already defined in urban/semi-urban
        rural_counties = [c for c in rural_counties if c not in ["Nairobi", "Nakuru"]]
        
        # Distribute remaining weight (42%) among rural counties
        remaining_weight = 1.0 - sum(c.population_weight for c in urban_centers + semi_urban)
        rural_weight_per_county = remaining_weight / len(rural_counties)
        
        rural = [
            CountyProfile(county, rural_weight_per_county, UrbanRuralClassification.RURAL)
            for county in rural_counties
        ]
        
        return urban_centers + semi_urban + rural
    
    def _sample_gender(self, n: int) -> List[Gender]:
        """
        Sample gender distribution based on FinAccess Survey 2024.
        
        Args:
            n: Number of samples to generate
            
        Returns:
            List of Gender enum values
        """
        gender_indices = self._rng.choice(
            [0, 1],  # 0 for Male, 1 for Female
            size=n,
            p=[self.config.gender_male_pct, 1.0 - self.config.gender_male_pct]
        )
        genders = [Gender.MALE if idx == 0 else Gender.FEMALE for idx in gender_indices]
        return genders
    
    def _sample_age(self, n: int) -> np.ndarray:
        """
        Sample age distribution: Normal(μ=32, σ=10) capped at [18, 80].
        
        Args:
            n: Number of samples to generate
            
        Returns:
            Array of age values
        """
        ages = self._rng.normal(
            loc=self.config.age_mean,
            scale=self.config.age_std,
            size=n
        )
        # Cap to [18, 80]
        ages = np.clip(ages, self.config.age_min, self.config.age_max)
        return ages.astype(int)
    
    def _sample_county(self, n: int) -> Tuple[List[str], List[UrbanRuralClassification]]:
        """
        Sample counties weighted by population with urban/rural classification.
        
        Args:
            n: Number of samples to generate
            
        Returns:
            Tuple of (county_names, urban_rural_classifications)
        """
        county_names = [c.name for c in self._county_profiles]
        population_weights = [c.population_weight for c in self._county_profiles]
        
        sampled_indices = self._rng.choice(
            len(self._county_profiles),
            size=n,
            p=population_weights
        )
        
        sampled_counties = [county_names[i] for i in sampled_indices]
        sampled_classifications = [self._county_profiles[i].classification for i in sampled_indices]
        
        return sampled_counties, sampled_classifications
    
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
            self.config.tier_3_pct
        ]
        
        tier_indices = self._rng.choice(
            [0, 1, 2],  # Indices for TIER_1, TIER_2, TIER_3
            size=n,
            p=tier_probs
        )
        
        kyc_tiers = [
            KYCTier.TIER_1 if idx == 0 else
            KYCTier.TIER_2 if idx == 1 else
            KYCTier.TIER_3
            for idx in tier_indices
        ]
        
        # Assign limits based on tier index
        tx_limits = [
            70000.0 if idx == 0 else
            150000.0 if idx == 1 else
            250000.0  # TIER_3
            for idx in tier_indices
        ]
        
        balance_limits = [
            300000.0 if idx == 0 else
            500000.0  # TIER_2 and TIER_3 have same balance limit
            for idx in tier_indices
        ]
        
        return kyc_tiers, tx_limits, balance_limits
    
    def _assign_risk_flags(self, n: int) -> Tuple[List[bool], List[bool]]:
        """
        Assign high-risk entity flags for transaction flagging.
        
        Args:
            n: Number of customers to assign flags to
            
        Returns:
            Tup le of (betting_platform_flags, international_tx_flags)
        """
        betting_flags = [self._rng.random() < self.config.betting_platform_pct for _ in range(n)]
        international_flags = [self._rng.random() < self.config.international_tx_pct for _ in range(n)]
        
        return betting_flags, international_flags
    
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
        
        # Generate demographic attributes
        genders = self._sample_gender(n)
        ages = self._sample_age(n)
        counties, urban_rural = self._sample_county(n)
        
        # Assign KYC tiers with limits
        kyc_tiers, tx_limits, balance_limits = self._assign_kyc_tier(n)
        
        # Assign high-risk flags
        betting_flags, international_flags = self._assign_risk_flags(n)
        
        # Generate account attributes
        account_ages = self._generate_account_age(n)
        initial_balances = self._generate_initial_balance(n, balance_limits)
        
        # Create customer profiles
        profiles = []
        for i in range(n):
            profile = CustomerProfile(
                customer_id=f"CUST_{i:06d}",
                gender=genders[i],
                age=int(ages[i]),
                county=counties[i],
                urban_rural_classification=urban_rural[i],
                kyc_tier=kyc_tiers[i],
                max_transaction_limit=tx_limits[i],
                max_balance_limit=balance_limits[i],
                betting_platform_flag=betting_flags[i],
                international_transaction_flag=international_flags[i],
                account_age_days=account_ages[i],
                initial_balance=initial_balances[i]
            )
            profiles.append(profile)
        
        logger.info(
            f"Generated {len(profiles)} customer profiles. "
            f"Gender distribution: {np.mean([p.gender == Gender.MALE for p in profiles]):.2%} Male. "
            f"Age mean: {np.mean([p.age for p in profiles]):.1f} years. "
            f"KYC tiers: T1={np.mean([p.kyc_tier == KYCTier.TIER_1 for p in profiles]):.2%}, "
            f"T2={np.mean([p.kyc_tier == KYCTier.TIER_2 for p in profiles]):.2%}, "
            f"T3={np.mean([p.kyc_tier == KYCTier.TIER_3 for p in profiles]):.2%}. "
            f"Risk flags: Betting={np.mean([p.betting_platform_flag for p in profiles]):.2%}, "
            f"International={np.mean([p.international_transaction_flag for p in profiles]):.2%}."
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
            "gender": [p.gender.value if isinstance(p.gender, Gender) else str(p.gender) for p in profiles],
            "age": [p.age for p in profiles],
            "county": [p.county for p in profiles],
            "urban_rural_classification": [p.urban_rural_classification.value if isinstance(p.urban_rural_classification, UrbanRuralClassification) else str(p.urban_rural_classification) for p in profiles],
            "kyc_tier": [p.kyc_tier.value if isinstance(p.kyc_tier, KYCTier) else str(p.kyc_tier) for p in profiles],
            "max_transaction_limit_kes": [p.max_transaction_limit for p in profiles],
            "max_balance_limit_kes": [p.max_balance_limit for p in profiles],
            "betting_platform_flag": [bool(p.betting_platform_flag) for p in profiles],
            "international_transaction_flag": [bool(p.international_transaction_flag) for p in profiles],
            "account_age_days": [p.account_age_days for p in profiles],
            "initial_balance_kes": [p.initial_balance for p in profiles],
        }
        
        df = pl.DataFrame(data)
        
        # Set proper data types
        schema = {
            "customer_id": pl.String,
            "gender": pl.String,
            "age": pl.Int32,
            "county": pl.String,
            "urban_rural_classification": pl.String,
            "kyc_tier": pl.String,
            "max_transaction_limit_kes": pl.Float64,
            "max_balance_limit_kes": pl.Float64,
            "betting_platform_flag": pl.Boolean,
            "international_transaction_flag": pl.Boolean,
            "account_age_days": pl.Int32,
            "initial_balance_kes": pl.Float64,
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
        # Create parent directory if it doesn't exist
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
    print(f"Gender distribution:\n{df['gender'].value_counts()}")
    print(f"\nAge statistics: Mean={df['age'].mean():.1f}, Std={df['age'].std():.1f}")
    print(f"\nCounty distribution (top 10):\n{df['county'].value_counts().head(10)}")
    print(f"\nKYC tier distribution:\n{df['kyc_tier'].value_counts()}")
    print(f"\nUrban/Rural distribution:\n{df['urban_rural_classification'].value_counts()}")
    print(f"\nRisk flags:")
    print(f"Betting platform users: {df['betting_platform_flag'].sum()} ({df['betting_platform_flag'].mean():.2%})")
    print(f"International transaction users: {df['international_transaction_flag'].sum()} ({df['international_transaction_flag'].mean():.2%})")
    
    # Save to CSV
    output_path = Path("/home/dan/project/sentinAI/data/bronze/customers/customer_profiles_complete.csv")
    generator.save_to_csv(df, output_path)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
