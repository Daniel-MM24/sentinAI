"""
Stateful Synthetic Generator for AML Transaction Simulation (TVAE Hybrid v2.0).

This module implements a comprehensive stateful customer data and transactional
simulation engine that generates synthetic cohorts of customers and tracks their
complete account lifecycles day-by-day—simulating real-time incoming and outgoing
transactions while maintaining rolling balance states that mimic an M-Pesa account.

TVAE Hybrid Implementation (v2.0) - 21-feature schema:
- Core Features (8): customer_id, tier, archetype, transaction_type, amount, timestamp, direction, balance
- Temporal Features (5): tx_count_7d, volume_7d, night_tx_ratio, rapid_tx_ratio, volume_7d_vs_30d_ratio
- Network Features (3): is_international, distinct_counterparties_7d, fan_in_fan_out_ratio
- Structuring Features (3): close_to_limit_ratio, balance_retention_ratio, amount_roundness
- Labels (2): is_launderer, aml_scenario

CRITICAL: This module maintains mathematical balance integrity and generates
stateful transaction histories with rolling window feature computation.

MRM Compliance Features:
- Deterministic seed configuration for reproducibility
- Customer archetype generation with conditional probabilities
- SHA-256 hash tokenization for PII elimination
- Regulatory tier compliance (Tier 1/2/3 value caps)
- Customer metadata export for audit trails
"""

import hashlib
import logging
import numpy as np
import polars as pl
from scipy import stats
from scipy.stats import gaussian_kde
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
from pathlib import Path
from typing import Optional

from src.data.anomaly_injector import FinancialAnomalyInjector, InjectorConfig

logger = logging.getLogger(__name__)


class CustomerArchetype(Enum):
    """Customer behavioral archetypes with transaction velocity patterns (MRM downscaling parameters)."""
    RETAIL_HEAVY = "retail_heavy"  # 15%: High transaction velocity, 850-950 tx/year
    RETAIL_STANDARD = "retail_standard"  # 70%: Moderate activity, 300-500 tx/year
    MICRO_MERCHANT = "micro_merchant"  # 12%: Business patterns, 600-800 tx/year
    CORPORATE_SME = "corporate_sme"  # 3%: High-value, low-frequency, 100-200 tx/year


class WalletTier(Enum):
    """Regulatory wallet tiers with transaction and balance limits (CBK PG/43 guidelines)."""
    TIER_1 = "tier_1"  # Basic: Max single: KES 10,000, Max balance: KES 50,000
    TIER_2 = "tier_2"  # Interim: Max single: KES 50,000, Max balance: KES 200,000
    TIER_3 = "tier_3"  # Full KYC: Max single: KES 150,000, Max balance: KES 1,000,000
    TIER_4 = "tier_4"  # EDD: Max single: KES 500,000, Max balance: KES 5,000,000


@dataclass
class CustomerProfile:
    """Stateful customer profile with behavioral characteristics."""
    customer_id: str
    archetype: CustomerArchetype  # Updated to use new archetypes
    initial_balance: float
    current_balance: float
    avg_transaction_amount: float
    transaction_frequency: float  # transactions per day
    balance_volatility: float
    device_id: str
    primary_location: str
    kyc_level: int
    account_age_days: int
    risk_score: float = 0.0
    
    # Customer identity fields for Silver/Gold schemas
    customer_name: str = ""
    email: str = ""
    tax_id: str = ""
    
    # Device and wallet attributes for Silver/Gold schemas
    device_age_days: int = 365  # Default device age
    sim_match_status: bool = True
    wallet_tier: WalletTier = WalletTier.TIER_1  # Updated to use WalletTier enum
    prev_fraud_flag_count_90d: int = 0

    # Encoded variants for Silver/Gold schemas
    wallet_tier_encoded: int = 0
    kyc_level_encoded: int = 0
    
    # Rolling window state
    daily_balances: deque = field(default_factory=lambda: deque(maxlen=30))
    recent_transactions: deque = field(default_factory=lambda: deque(maxlen=100))
    counterparties: set = field(default_factory=set)
    device_history: deque = field(default_factory=lambda: deque(maxlen=10))
    location_history: deque = field(default_factory=lambda: deque(maxlen=20))


@dataclass
class AMLGeneratorConfig:
    """Configuration for the AML-focused synthetic generator."""
    num_customers: int = 1000
    num_days: int = 90
    target_transactions: Optional[int] = None
    seed: int = 42
    
    # Transaction parameters
    amount_mean: float = 7.0  # Log-normal mean
    amount_std: float = 1.2   # Log-normal std
    velocity_lambda: float = 15.0  # Exponential inter-arrival rate
    
    # Feature computation windows
    velocity_windows: List[str] = field(default_factory=lambda: ["1min", "5min", "1h", "24h", "7d", "30d"])
    balance_windows: List[str] = field(default_factory=lambda: ["7d", "30d"])
    
    # AML thresholds
    structuring_threshold: float = 900000.0  # KES
    ctr_threshold: float = 1000000.0  # KES (Cash Transaction Report)
    high_risk_hours: List[int] = field(default_factory=lambda: [2, 3, 4])
    
    # Network parameters
    network_density: float = 0.1
    community_detection: bool = True
    
    # MRM Compliance parameters
    enable_pii_hashing: bool = True  # Enable SHA-256 hash tokenization
    enforce_regulatory_caps: bool = True  # Enforce tier transaction/balance limits
    export_customer_metadata: bool = True  # Export customers_metadata.csv


@dataclass
class GeneratorConfig:
    """Configuration for the CleanDataGenerator (legacy)."""
    num_records: int = 10000
    num_entities: int = 1000
    seed: int = 42
    amount_mean: float = 7.0  # Log-normal mean
    amount_std: float = 1.2   # Log-normal std
    velocity_lambda: float = 15.0  # Exponential inter-arrival rate
    correlation_strength: float = 0.7  # Strength of feature correlations


class CleanDataGenerator:
    """
    Generates pristine, mathematically clean baseline financial data.
    
    This class creates synthetic financial records using statistical 
    distributions and copulas to ensure realistic feature correlations
    without any knowledge of downstream anomaly injection or fraud detection.
    
    The generator produces clean transaction/market data including:
    - Transaction amounts (log-normal distribution)
    - Temporal patterns (exponential inter-arrival times)
    - Entity relationships (correlated features)
    - Market microstructure (volume, price impact)
    
    All data is generated using vectorized Polars expressions for performance.
    """
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        """
        Initialize the CleanDataGenerator.
        
        Args:
            config: GeneratorConfig instance with generation parameters.
                   If None, uses default configuration.
        """
        self.config = config or GeneratorConfig()
        self._rng = np.random.default_rng(self.config.seed)
        
        logger.info(
            f"Initialized CleanDataGenerator with {self.config.num_records} records, "
            f"{self.config.num_entities} entities, seed={self.config.seed}"
        )
    
    def _generate_correlated_features(
        self, 
        n_samples: int,
        means: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Generate correlated features using multivariate normal distribution.
        
        Args:
            n_samples: Number of samples to generate
            means: Mean vector for the multivariate distribution
            cov_matrix: Covariance matrix defining feature correlations
            
        Returns:
            Array of shape (n_samples, n_features) with correlated values
        """
        # Generate from multivariate normal
        correlated = self._rng.multivariate_normal(
            mean=means,
            cov=cov_matrix,
            size=n_samples
        )
        
        # Apply transformations to ensure realistic ranges
        # Feature 0: Transaction amount (log-normal)
        correlated[:, 0] = np.exp(correlated[:, 0])
        
        # Feature 1: Volume (exponential)
        correlated[:, 1] = np.exp(correlated[:, 1] / 2.0)
        
        # Feature 2: Price impact (bounded)
        correlated[:, 2] = np.clip(correlated[:, 2], -0.1, 0.1)
        
        # Feature 3: Account balance (log-normal with higher mean)
        correlated[:, 3] = np.exp(correlated[:, 3] + 8)
        
        # Feature 4: Transaction count (Poisson-like)
        correlated[:, 4] = np.clip(np.round(np.exp(correlated[:, 4])), 1, 100)
        
        return correlated
    
    def _generate_temporal_patterns(self, n_records: int) -> pl.Series:
        """
        Generate realistic temporal patterns using exponential inter-arrival times.
        
        Simulates business hours patterns with diurnal spikes while maintaining
        realistic transaction velocity.
        
        Args:
            n_records: Number of timestamp records to generate
            
        Returns:
            Polars Series of datetime values
        """
        # Generate inter-arrival times (exponential distribution)
        inter_arrival = self._rng.exponential(
            scale=1.0 / self.config.velocity_lambda,
            size=n_records
        )
        
        # Convert to cumulative seconds
        cumulative_seconds = np.cumsum(inter_arrival)
        
        # Start from beginning of FY 2025
        start_time = datetime(2025, 1, 1, 0, 0, 0)
        
        # Generate timestamps
        timestamps = [
            start_time + timedelta(seconds=float(sec))
            for sec in cumulative_seconds
        ]
        
        return pl.Series("timestamp", timestamps).cast(pl.Datetime("us", "UTC"))
    
    def _generate_entity_features(self, n_records: int) -> Dict[str, pl.Series]:
        """
        Generate entity-level features with realistic correlations.
        
        Args:
            n_records: Number of records to generate
            
        Returns:
            Dictionary of feature name to Polars Series
        """
        # Generate entity IDs
        entity_indices = self._rng.integers(
            0, self.config.num_entities, size=n_records
        )
        entity_ids = pl.Series(
            "entity_id", 
            [f"entity_{idx}" for idx in entity_indices]
        )
        
        # Generate correlated features
        n_features = 5
        means = np.array([
            self.config.amount_mean,  # Amount
            3.0,                       # Volume
            0.0,                       # Price impact
            10.0,                      # Balance
            2.0                        # Transaction count
        ])
        
        # Create correlation matrix with realistic financial correlations
        # Amount correlates with volume and balance
        cov_matrix = self.config.correlation_strength * np.ones((n_features, n_features))
        np.fill_diagonal(cov_matrix, 1.0)
        
        # Add some specific correlation structure (symmetric)
        cov_matrix[0, 1] = 0.8  # Amount-Volume correlation
        cov_matrix[1, 0] = 0.8  # Volume-Amount correlation
        cov_matrix[0, 3] = 0.6  # Amount-Balance correlation
        cov_matrix[3, 0] = 0.6  # Balance-Amount correlation
        cov_matrix[1, 2] = 0.4  # Volume-Price impact correlation
        cov_matrix[2, 1] = 0.4  # Price impact-Volume correlation
        
        # Ensure positive semidefinite by computing nearest correlation matrix
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        eigvals = np.maximum(eigvals, 1e-8)  # Ensure positive eigenvalues
        cov_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        correlated_features = self._generate_correlated_features(
            n_records, means, cov_matrix
        )
        
        features = {
            "entity_id": entity_ids,
            "transaction_amount": pl.Series(
                "transaction_amount", 
                correlated_features[:, 0]
            ),
            "volume": pl.Series("volume", correlated_features[:, 1]),
            "price_impact": pl.Series("price_impact", correlated_features[:, 2]),
            "account_balance": pl.Series(
                "account_balance", 
                correlated_features[:, 3]
            ),
            "transaction_count": pl.Series(
                "transaction_count", 
                correlated_features[:, 4].astype(int)
            )
        }
        
        return features
    
    def _generate_market_microstructure(self, n_records: int) -> Dict[str, pl.Series]:
        """
        Generate market microstructure features.
        
        Args:
            n_records: Number of records to generate
            
        Returns:
            Dictionary of feature name to Polars Series
        """
        # Bid-ask spread (log-normal)
        spread = self._rng.lognormal(
            mean=0.5, sigma=0.3, size=n_records
        )
        
        # Market depth (exponential)
        depth = self._rng.exponential(scale=1000.0, size=n_records)
        
        # Volatility (gamma distribution)
        volatility = self._rng.gamma(shape=2.0, scale=0.01, size=n_records)
        
        # Liquidity score (beta distribution bounded [0,1])
        liquidity = self._rng.beta(a=2, b=2, size=n_records)
        
        return {
            "bid_ask_spread": pl.Series("bid_ask_spread", spread),
            "market_depth": pl.Series("market_depth", depth),
            "volatility": pl.Series("volatility", volatility),
            "liquidity_score": pl.Series("liquidity_score", liquidity)
        }
    
    def _generate_categorical_features(self, n_records: int) -> Dict[str, pl.Series]:
        """
        Generate categorical features with realistic distributions.
        
        Args:
            n_records: Number of records to generate
            
        Returns:
            Dictionary of feature name to Polars Series
        """
        # Transaction type (categorical)
        tx_types = ["P2P", "C2B", "B2C", "B2B"]
        tx_type_probs = [0.5, 0.25, 0.15, 0.1]
        transaction_types = pl.Series(
            "transaction_type",
            self._rng.choice(tx_types, p=tx_type_probs, size=n_records)
        )
        
        # Channel (categorical)
        channels = ["USSD", "App", "STK_Push", "Web"]
        channel_probs = [0.4, 0.35, 0.2, 0.05]
        channels = pl.Series(
            "channel",
            self._rng.choice(channels, p=channel_probs, size=n_records)
        )
        
        # Region (categorical)
        regions = ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret"]
        region_probs = [0.4, 0.2, 0.15, 0.15, 0.1]
        regions = pl.Series(
            "region",
            self._rng.choice(regions, p=region_probs, size=n_records)
        )
        
        # Counterparty risk tier (categorical)
        risk_tiers = ["Low", "Medium", "High"]
        tier_probs = [0.7, 0.25, 0.05]
        risk_tiers = pl.Series(
            "counterparty_risk_tier",
            self._rng.choice(risk_tiers, p=tier_probs, size=n_records)
        )
        
        return {
            "transaction_type": transaction_types,
            "channel": channels,
            "region": regions,
            "counterparty_risk_tier": risk_tiers
        }
    
    def generate(self) -> pl.DataFrame:
        """
        Generate a complete clean financial dataset.
        
        This method orchestrates the generation of all features and combines
        them into a single Polars DataFrame. The output contains NO anomaly
        flags or fraud indicators - only clean, realistic financial data.
        
        Returns:
            Polars DataFrame containing pristine financial records with columns:
            - entity_id: Unique entity identifier
            - transaction_amount: Transaction value (log-normal)
            - volume: Trading volume (exponential)
            - price_impact: Market price impact (bounded)
            - account_balance: Account balance (log-normal)
            - transaction_count: Historical transaction count (Poisson-like)
            - bid_ask_spread: Market spread (log-normal)
            - market_depth: Market depth (exponential)
            - volatility: Price volatility (gamma)
            - liquidity_score: Liquidity metric (beta)
            - transaction_type: Transaction category
            - channel: Transaction channel
            - region: Geographic region
            - counterparty_risk_tier: Risk classification
            - timestamp: Transaction timestamp
            
        Raises:
            ValueError: If configuration parameters are invalid
        """
        logger.info(f"Generating {self.config.num_records} clean financial records")
        
        # Validate configuration
        if self.config.num_records <= 0:
            raise ValueError("num_records must be positive")
        if self.config.num_entities <= 0:
            raise ValueError("num_entities must be positive")
        
        # Generate feature groups
        entity_features = self._generate_entity_features(self.config.num_records)
        market_features = self._generate_market_microstructure(self.config.num_records)
        categorical_features = self._generate_categorical_features(self.config.num_records)
        timestamps = self._generate_temporal_patterns(self.config.num_records)
        
        # Combine all features into a single DataFrame
        all_features = {
            **entity_features,
            **market_features,
            **categorical_features,
            "timestamp": timestamps
        }
        
        df = pl.DataFrame(all_features)
        
        # Ensure proper data types
        schema = {
            "entity_id": pl.String,
            "transaction_amount": pl.Float64,
            "volume": pl.Float64,
            "price_impact": pl.Float64,
            "account_balance": pl.Float64,
            "transaction_count": pl.Int64,
            "bid_ask_spread": pl.Float64,
            "market_depth": pl.Float64,
            "volatility": pl.Float64,
            "liquidity_score": pl.Float64,
            "transaction_type": pl.String,
            "channel": pl.String,
            "region": pl.String,
            "counterparty_risk_tier": pl.String,
            "timestamp": pl.Datetime("us", "UTC")
        }
        
        df = df.cast(schema)
        
        # Sort by timestamp for chronological ordering
        df = df.sort("timestamp")
        
        logger.info(
            f"Successfully generated {len(df)} clean records with "
            f"{len(df.columns)} features. "
            f"Amount range: [{df['transaction_amount'].min():.2f}, "
            f"{df['transaction_amount'].max():.2f}]"
        )
        
        return df
    
    def generate_summary_statistics(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for the clean dataset.
        
        Args:
            df: Polars DataFrame generated by this class
            
        Returns:
            Dictionary containing summary statistics
        """
        numeric_cols = df.select(pl.col(pl.Float64, pl.Int64)).columns
        
        stats_dict = {
            "num_records": len(df),
            "num_entities": df["entity_id"].n_unique(),
            "date_range": {
                "start": df["timestamp"].min().isoformat(),
                "end": df["timestamp"].max().isoformat()
            },
            "feature_statistics": {}
        }
        
        for col in numeric_cols:
            col_stats = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "median": float(df[col].median())
            }
            stats_dict["feature_statistics"][col] = col_stats
        
        return stats_dict


def main():
    """Example usage of the CleanDataGenerator."""
    logging.basicConfig(level=logging.INFO)
    
    # Create generator with custom configuration
    config = GeneratorConfig(
        num_records=5000,
        num_entities=500,
        seed=42
    )
    
    generator = CleanDataGenerator(config)
    
    # Generate clean data
    clean_df = generator.generate()
    
    # Display summary
    print("\n=== Clean Data Generated ===")
    print(f"Shape: {clean_df.shape}")
    print(f"\nFirst 5 rows:")
    print(clean_df.head())
    
    # Generate and display statistics
    stats = generator.generate_summary_statistics(clean_df)
    print(f"\n=== Summary Statistics ===")
    print(f"Records: {stats['num_records']}")
    print(f"Unique Entities: {stats['num_entities']}")
    print(f"Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}")


if __name__ == "__main__":
    main()


class AMLGenerator:
    """
    Stateful AML-focused synthetic transaction generator.
    
    This class implements a comprehensive stateful customer data and transactional
    simulation engine that generates synthetic cohorts of customers and tracks their
    complete account lifecycles day-by-day—simulating real-time incoming and outgoing
    transactions while maintaining rolling balance states that mimic an M-Pesa account.
    
    TVAE Hybrid v2.0 - Core features generated (9 columns):
    - customer_id, tier, archetype, transaction_type, amount, timestamp, direction, balance, counterparty
    - Additional 10 features computed by CustomerFeatureEngineer:
      tx_count_7d, volume_7d, night_tx_ratio, rapid_tx_ratio, volume_7d_vs_30d_ratio,
      distinct_counterparties_7d, fan_in_fan_out_ratio, close_to_limit_ratio,
      balance_retention_ratio, amount_roundness
    - Labels: is_launderer, aml_scenario
    """
    
    def __init__(self, config: Optional[AMLGeneratorConfig] = None):
        """
        Initialize the AMLGenerator.
        
        Args:
            config: AMLGeneratorConfig instance with generation parameters.
                   If None, uses default configuration.
        """
        self.config = config or AMLGeneratorConfig()
        self._rng = np.random.default_rng(self.config.seed)
        
        # Stateful customer storage
        self.customers: Dict[str, CustomerProfile] = {}
        
        # Transaction graph state
        self.transaction_graph: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.transaction_amounts: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        
        # Global transaction counter for unique IDs
        self._transaction_counter = 0
        
        # Feature computation caches
        self._velocity_cache: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: {window: deque() for window in self.config.velocity_windows}
        )
        
        logger.info(
            f"Initialized AMLGenerator with {self.config.num_customers} customers, "
            f"{self.config.num_days} days simulation, seed={self.config.seed}"
        )
    
    def _generate_customer_profiles(self) -> None:
        """
        Generate synthetic customer profiles with behavioral archetypes.
        
        Creates diverse customer archetypes with realistic behavioral parameters
        including transaction patterns, balance management, and device/location profiles.
        Uses MRM downscaling parameters for archetype distribution.
        """
        # MRM downscaling parameters: 15%, 70%, 12%, 3%
        archetypes_list = list(CustomerArchetype)
        archetype_weights = [0.15, 0.70, 0.12, 0.03]  # Retail Heavy, Standard, Micro-Merchant, Corporate/SME
        
        locations = ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret", "Kiambu"]
        
        for i in range(self.config.num_customers):
            # Generate phone number and hash it for PII elimination
            phone_number = f"2547{self._rng.integers(0, 100_000_000):08d}"
            if self.config.enable_pii_hashing:
                customer_id = hashlib.sha256(phone_number.encode()).hexdigest()
            else:
                customer_id = f"CUST_{i:06d}"
            
            # Assign archetype based on MRM conditional probabilities
            archetype = self._rng.choice(archetypes_list, p=archetype_weights)
            
            # Assign wallet tier based on archetype
            wallet_tier = self._assign_wallet_tier(archetype)
            
            # Generate archetype-specific parameters based on MRM specifications
            if archetype == CustomerArchetype.RETAIL_HEAVY:
                # 15%: High transaction velocity, 850-950 tx/year
                tx_per_year = self._rng.integers(850, 951)
                frequency = tx_per_year / 365.0  # transactions per day
                avg_amount = np.exp(self._rng.normal(7.0, 0.5))
                initial_balance = np.exp(self._rng.normal(8.0, 1.0))
                volatility = self._rng.uniform(0.2, 0.4)
            elif archetype == CustomerArchetype.RETAIL_STANDARD:
                # 70%: Moderate activity, 300-500 tx/year
                tx_per_year = self._rng.integers(300, 501)
                frequency = tx_per_year / 365.0
                avg_amount = np.exp(self._rng.normal(7.0, 0.6))
                initial_balance = np.exp(self._rng.normal(8.0, 1.2))
                volatility = self._rng.uniform(0.1, 0.3)
            elif archetype == CustomerArchetype.MICRO_MERCHANT:
                # 12%: Business patterns, 600-800 tx/year
                tx_per_year = self._rng.integers(600, 801)
                frequency = tx_per_year / 365.0
                avg_amount = np.exp(self._rng.normal(8.0, 0.8))
                initial_balance = np.exp(self._rng.normal(9.5, 1.5))
                volatility = self._rng.uniform(0.3, 0.5)
            else:  # CORPORATE_SME
                # 3%: High-value, low-frequency, 100-200 tx/year
                tx_per_year = self._rng.integers(100, 201)
                frequency = tx_per_year / 365.0
                avg_amount = np.exp(self._rng.normal(9.5, 1.0))
                initial_balance = np.exp(self._rng.normal(11.0, 2.0))
                volatility = self._rng.uniform(0.4, 0.6)
            
            # Enforce regulatory tier caps on initial balance
            if self.config.enforce_regulatory_caps:
                initial_balance = self._enforce_balance_cap(initial_balance, wallet_tier)
            
            # Generate device and location
            device_id = f"DEV_{self._rng.integers(100000, 999999)}"
            primary_location = self._rng.choice(locations)
            kyc_level = self._rng.integers(1, 5)
            account_age = self._rng.integers(30, 365)
            
            # Generate customer identity fields (with PII hashing if enabled)
            customer_name = f"Customer_{i}"
            if self.config.enable_pii_hashing:
                email_hash = hashlib.sha256(f"customer{i}@example.com".encode()).hexdigest()
                email = f"{email_hash[:16]}@sentinai.synthetic"
                tax_id = hashlib.sha256(f"TAX_{self._rng.integers(10000000, 99999999)}".encode()).hexdigest()[:16]
            else:
                email = f"customer{i}@example.com"
                tax_id = f"TAX_{self._rng.integers(10000000, 99999999)}"
            
            # Generate device and wallet attributes
            device_age = self._rng.integers(30, 1825)  # 30 days to 5 years
            sim_match = self._rng.random() < 0.95  # 95% match rate
            prev_fraud_count = self._rng.integers(0, 3)  # Most have 0-2 flags
            
            # Create customer profile
            profile = CustomerProfile(
                customer_id=customer_id,
                archetype=archetype,
                initial_balance=initial_balance,
                current_balance=initial_balance,
                avg_transaction_amount=avg_amount,
                transaction_frequency=frequency,
                balance_volatility=volatility,
                device_id=device_id,
                primary_location=primary_location,
                kyc_level=kyc_level,
                account_age_days=account_age,
                customer_name=customer_name,
                email=email,
                tax_id=tax_id,
                device_age_days=device_age,
                sim_match_status=sim_match,
                wallet_tier=wallet_tier,
                prev_fraud_flag_count_90d=prev_fraud_count
            )
            
            self.customers[customer_id] = profile
        
        logger.info(f"Generated {len(self.customers)} customer profiles with MRM archetypes")
    
    def _assign_wallet_tier(self, archetype: CustomerArchetype) -> WalletTier:
        """
        Assign regulatory wallet tier based on customer archetype.

        Args:
            archetype: Customer archetype

        Returns:
            WalletTier enum value
        """
        # Tier assignment logic based on archetype patterns (CBK PG/43 alignment)
        if archetype == CustomerArchetype.CORPORATE_SME:
            # Corporate/SMEs typically get Tier 3 or 4 for high-value transactions
            tier_probs = [0.05, 0.15, 0.50, 0.30]
        elif archetype == CustomerArchetype.MICRO_MERCHANT:
            # Micro-merchants typically Tier 2-3
            tier_probs = [0.10, 0.55, 0.30, 0.05]
        elif archetype == CustomerArchetype.RETAIL_HEAVY:
            # Heavy retail users mixed Tier 1/2
            tier_probs = [0.35, 0.50, 0.12, 0.03]
        else:  # RETAIL_STANDARD
            # Standard retail mostly Tier 1
            tier_probs = [0.70, 0.20, 0.08, 0.02]

        tiers = list(WalletTier)
        tier = self._rng.choice(tiers, p=tier_probs)
        return tier
    
    def _enforce_balance_cap(self, balance: float, tier: WalletTier) -> float:
        """
        Enforce regulatory balance caps based on wallet tier.
        
        Args:
            balance: Proposed balance amount
            tier: Wallet tier
            
        Returns:
            Balance clipped to tier limit
        """
        tier_limits = {
            WalletTier.TIER_1: 50_000.0,     # CBK Tier 1: max wallet KES 50,000
            WalletTier.TIER_2: 200_000.0,    # CBK Tier 2: max wallet KES 200,000
            WalletTier.TIER_3: 1_000_000.0,  # CBK Tier 3: max wallet KES 1,000,000
            WalletTier.TIER_4: 5_000_000.0,  # CBK Tier 4 (EDD): no cap — soft limit
        }

        max_balance = tier_limits.get(tier, 50_000.0)
        return min(balance, max_balance)
    
    def _enforce_transaction_cap(self, amount: float, tier: WalletTier) -> float:
        """
        Enforce regulatory transaction caps based on wallet tier.
        
        Args:
            amount: Proposed transaction amount
            tier: Wallet tier
            
        Returns:
            Amount clipped to tier limit
        """
        tier_limits = {
            WalletTier.TIER_1: 10_000.0,     # CBK Tier 1: max single tx KES 10,000
            WalletTier.TIER_2: 50_000.0,     # CBK Tier 2: max single tx KES 50,000
            WalletTier.TIER_3: 150_000.0,    # CBK Tier 3: max single tx KES 150,000
            WalletTier.TIER_4: 500_000.0,    # CBK Tier 4 (EDD): max single tx KES 500,000
        }

        max_transaction = tier_limits.get(tier, 10_000.0)
        return min(amount, max_transaction)
    
    def _get_velocity_features(
        self, 
        customer_id: str, 
        current_time: datetime
    ) -> Dict[str, float]:
        """
        Compute real-time velocity features for a customer.
        
        Args:
            customer_id: Customer identifier
            current_time: Current transaction timestamp
            
        Returns:
            Dictionary of velocity features
        """
        profile = self.customers[customer_id]
        velocity_features = {}
        
        # Get recent transactions from profile state
        recent_txs = [
            tx for tx in profile.recent_transactions 
            if tx['timestamp'] > current_time - timedelta(days=30)
        ]
        
        # Compute velocity for each window
        for window in self.config.velocity_windows:
            if window == "1min":
                cutoff = current_time - timedelta(minutes=1)
            elif window == "5min":
                cutoff = current_time - timedelta(minutes=5)
            elif window == "1h":
                cutoff = current_time - timedelta(hours=1)
            elif window == "24h":
                cutoff = current_time - timedelta(days=1)
            elif window == "7d":
                cutoff = current_time - timedelta(days=7)
            else:  # 30d
                cutoff = current_time - timedelta(days=30)
            
            window_txs = [tx for tx in recent_txs if tx['timestamp'] > cutoff]
            
            # Count and sum features
            velocity_features[f'tx_count_{window}'] = len(window_txs)
            velocity_features[f'amount_sum_{window}'] = sum(tx['amount'] for tx in window_txs)
        
        # TVAE Hybrid v2.0 - No legacy velocity features computed here
        # These are now computed by CustomerFeatureEngineer
        
        return velocity_features
    
    def _compute_balance_features(
        self, 
        customer_id: str, 
        current_time: datetime
    ) -> Dict[str, float]:
        """
        Compute balance-based features for 21-feature schema.
        
        Args:
            customer_id: Customer identifier
            current_time: Current transaction timestamp
            
        Returns:
            Dictionary of balance features (only balance_retention_ratio for 21-feature schema)
        """
        profile = self.customers[customer_id]
        balance_features = {}
        
        # Current balance
        balance_features['balance'] = profile.current_balance
        
        # Balance retention ratio (simplified) - only balance feature in 21-feature schema
        recent_inflows = sum(
            tx['amount'] for tx in list(profile.recent_transactions)[-20:]
            if tx['type'] == 'deposit' and tx['timestamp'] > current_time - timedelta(days=1)
        )
        if recent_inflows > 0:
            balance_features['balance_retention_ratio'] = min(profile.current_balance / recent_inflows, 1.0)
        else:
            balance_features['balance_retention_ratio'] = 0.0
        
        return balance_features
    
    def _get_amount_pattern_features(
        self, 
        customer_id: str, 
        amount: float,
        current_time: datetime
    ) -> Dict[str, float]:
        """
        Compute amount pattern features for AML detection.
        
        Args:
            customer_id: Customer identifier
            amount: Current transaction amount
            current_time: Current transaction timestamp
            
        Returns:
            Dictionary of amount pattern features
        """
        profile = self.customers[customer_id]
        amount_features = {}
        
        # Amount vs profile average
        amount_features['amount_vs_profile_avg'] = amount / (profile.avg_transaction_amount + 1e-6)
        
        # Amount roundness (how round is the number?)
        amount_features['amount_roundness'] = self._calculate_roundness(amount)
        
        # Amount just below threshold (structuring indicator)
        amount_features['amount_just_below_threshold'] = 1.0 if (
            self.config.structuring_threshold * 0.9 <= amount < self.config.structuring_threshold
        ) else 0.0
        
        # Similar amount count in 24h (±5%)
        recent_txs = [
            tx for tx in profile.recent_transactions 
            if tx['timestamp'] > current_time - timedelta(days=1)
        ]
        similar_count = sum(
            1 for tx in recent_txs 
            if abs(tx['amount'] - amount) / (amount + 1e-6) < 0.05
        )
        amount_features['similar_amount_count_24h'] = similar_count
        
        # Identical amount count in 24h
        identical_count = sum(1 for tx in recent_txs if abs(tx['amount'] - amount) < 1.0)
        amount_features['identical_amount_count_24h'] = identical_count
        
        # Structuring amount entropy
        if len(recent_txs) > 1:
            recent_amounts = [tx['amount'] for tx in recent_txs]
            amount_features['structuring_amount_entropy'] = self._calculate_amount_entropy(recent_amounts)
        else:
            amount_features['structuring_amount_entropy'] = 0.0
        
        return amount_features
    
    def _calculate_roundness(self, amount: float) -> float:
        """
        Calculate how round an amount is (AML structuring indicator).
        
        Round amounts like 1000, 10000, 100000 are suspicious for structuring.
        
        Args:
            amount: Transaction amount
            
        Returns:
            Roundness score (0.0 to 1.0, higher = more round)
        """
        if amount < 1.0:
            return 0.0
        
        # Check if amount is close to a round number
        log_amount = np.log10(amount)
        nearest_power = 10 ** np.round(log_amount)
        
        # Calculate proximity to nearest round number
        roundness = 1.0 - min(abs(amount - nearest_power) / amount, 1.0)
        
        return roundness
    
    def _calculate_amount_entropy(self, amounts: List[float]) -> float:
        """
        Calculate entropy of transaction amounts (structuring detection).
        
        Low entropy indicates similar amounts (potential structuring).
        
        Args:
            amounts: List of transaction amounts
            
        Returns:
            Entropy value
        """
        if len(amounts) <= 1:
            return 0.0
        
        # Bin amounts and calculate entropy
        hist, _ = np.histogram(amounts, bins=10, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        
        return entropy
    
    def _get_network_features(self, customer_id: str) -> Dict[str, float]:
        """
        Compute network-based features for AML detection.
        
        Args:
            customer_id: Customer identifier
            
        Returns:
            Dictionary of network features
        """
        profile = self.customers[customer_id]
        network_features = {}
        
        # TVAE Hybrid v2.0 - Network features computed by CustomerFeatureEngineer
        # Only basic counterparty tracking needed for core generation
        network_features['distinct_counterparties_7d'] = len(profile.counterparties)
        
        return network_features
    
    def _get_temporal_features(
        self, 
        customer_id: str, 
        current_time: datetime
    ) -> Dict[str, float]:
        """
        Compute temporal anomaly features (TVAE Hybrid v2.0).
        
        Args:
            customer_id: Customer identifier
            current_time: Current transaction timestamp
            
        Returns:
            Dictionary of temporal features
        """
        profile = self.customers[customer_id]
        temporal_features = {}
        
        # TVAE Hybrid v2.0 - Temporal features computed by CustomerFeatureEngineer
        # Only basic timestamp needed for core generation
        
        # Is anomalous hour (2-5 AM)
        temporal_features['is_anomalous_hour'] = 1.0 if current_time.hour in self.config.high_risk_hours else 0.0
        
        # Time since last transaction
        if profile.recent_transactions:
            last_tx_time = list(profile.recent_transactions)[-1]['timestamp']
            temporal_features['time_since_last_tx'] = (current_time - last_tx_time).total_seconds()
        else:
            temporal_features['time_since_last_tx'] = float('inf')
        
        # Balance depletion rate
        if len(profile.daily_balances) >= 2:
            recent_balances = list(profile.daily_balances)[-7:]
            depletion = (recent_balances[0] - recent_balances[-1]) / (recent_balances[0] + 1e-6)
            temporal_features['balance_depletion_rate'] = max(0.0, depletion)
        else:
            temporal_features['balance_depletion_rate'] = 0.0
        
        return temporal_features
    
    def _get_device_location_features(
        self, 
        customer_id: str, 
        current_time: datetime
    ) -> Dict[str, float]:
        """
        Compute device and location intelligence features.
        
        Args:
            customer_id: Customer identifier
            current_time: Current transaction timestamp
            
        Returns:
            Dictionary of device/location features
        """
        profile = self.customers[customer_id]
        device_location_features = {}
        
        # Device changes in 7 days (simplified)
        unique_devices = set(profile.device_history)
        device_location_features['device_changes_7d'] = len(unique_devices)
        
        # Location entropy
        unique_locations = set(profile.location_history)
        if len(unique_locations) > 1:
            # Calculate entropy of location distribution
            location_counts = defaultdict(int)
            for loc in profile.location_history:
                location_counts[loc] += 1
            total = sum(location_counts.values())
            probs = [count / total for count in location_counts.values()]
            entropy = -sum(p * np.log(p + 1e-10) for p in probs)
            device_location_features['location_entropy'] = entropy
        else:
            device_location_features['location_entropy'] = 0.0
        
        # Device change flag
        if len(profile.device_history) > 1:
            device_history_list = list(profile.device_history)
            device_location_features['device_change_flag'] = 1.0 if device_history_list[-1] != device_history_list[-2] else 0.0
        else:
            device_location_features['device_change_flag'] = 0.0
        
        return device_location_features
    
    def _generate_transaction_amount(self, customer_id: str) -> float:
        """
        Generate a realistic transaction amount based on customer profile.
        
        Args:
            customer_id: Customer identifier
            
        Returns:
            Transaction amount
        """
        profile = self.customers[customer_id]
        
        # Sample from log-normal distribution centered on profile average
        amount = self._rng.lognormal(
            mean=np.log(profile.avg_transaction_amount),
            sigma=0.5
        )
        
        return amount
    
    # 168-hour weekly intensity vector indexed by (dow * 24 + hour)
    # Matches temporal_model.py WEEKLY_INTENSITY — kept locally for self-containment.
    _WEEKLY_INTENSITY: list[float] = [
        0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.40, 0.40, 0.40, 0.85, 0.85, 0.85,
        0.85, 0.75, 0.75, 0.75, 1.00, 1.00, 1.00, 1.00, 0.50, 0.50, 0.50, 0.50,
        0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.35, 0.35, 0.35, 0.80, 0.80, 0.80,
        0.80, 0.70, 0.70, 0.70, 0.95, 0.95, 0.95, 0.95, 0.45, 0.45, 0.45, 0.45,
        0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.35, 0.35, 0.35, 0.80, 0.80, 0.80,
        0.80, 0.70, 0.70, 0.70, 0.95, 0.95, 0.95, 0.95, 0.45, 0.45, 0.45, 0.45,
        0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.35, 0.35, 0.35, 0.80, 0.80, 0.80,
        0.80, 0.70, 0.70, 0.70, 0.95, 0.95, 0.95, 0.95, 0.50, 0.50, 0.50, 0.50,
        0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.35, 0.35, 0.35, 0.75, 0.75, 0.75,
        0.75, 0.65, 0.65, 0.65, 0.90, 0.90, 0.90, 0.90, 0.60, 0.60, 0.60, 0.60,
        0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.15, 0.15, 0.15, 0.50, 0.50, 0.50,
        0.50, 0.55, 0.55, 0.55, 0.50, 0.50, 0.50, 0.50, 0.35, 0.35, 0.35, 0.35,
        0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.30, 0.30, 0.30,
        0.30, 0.35, 0.35, 0.35, 0.30, 0.30, 0.30, 0.30, 0.15, 0.15, 0.15, 0.15,
    ]

    @staticmethod
    def _week_intensity(ts: datetime) -> float:
        """Return [0, 1] intensity for the hour-of-week at *ts*."""
        idx = ts.weekday() * 24 + ts.hour
        return AMLGenerator._WEEKLY_INTENSITY[idx]

    def _generate_transaction_time(
        self,
        customer_id: str,
        last_time: datetime,
    ) -> datetime:
        """
        Generate next transaction time using **thinning** of a homogeneous
        Poisson process with the temporal_model WEEKLY_INTENSITY vector.

        The base rate λ_base = *transaction_frequency* / 24  (events / hour).
        The max rate λ_max = λ_base × 1.0 (peak weekday intensity).
        Candidate times are drawn from Exp(λ_max) and accepted with
        probability intensity(t) / 1.0.

        This naturally produces:
        - Peak-hour clustering (morning 6-9, evening 17-20)
        - Off-peak lulls (late night intensity ~0.02 → ~98 % rejection)
        - Weekday/weekend variation
        """
        profile = self.customers[customer_id]
        # Base hourly rate from daily frequency
        lambda_base = profile.transaction_frequency / 24.0  # events / hour
        lambda_max = lambda_base * 1.0  # peak intensity = 1.0

        # Thinning loop
        while True:
            # Draw candidate inter-arrival from homogeneous Exp(λ_max)
            dt_hours = self._rng.exponential(1.0 / lambda_max)
            candidate = last_time + timedelta(hours=dt_hours)

            # Accept / reject based on intensity at candidate time
            intensity = self._week_intensity(candidate)
            if intensity <= 0.0:
                # Off-peak floor — always advance at least a little
                continue
            if self._rng.random() < intensity:
                return candidate

            # Rejected: advance last_time to candidate and retry from there
            last_time = candidate
    
    def _select_counterparty(self, customer_id: str, exclude: set = None) -> str:
        """
        Select a counterparty for transaction.
        
        Args:
            customer_id: Customer identifier
            exclude: Set of customer IDs to exclude from selection
            
        Returns:
            Counterparty customer ID
        """
        if exclude is None:
            exclude = self.customers.keys()
        
        # Prefer existing counterparties (network effect)
        profile = self.customers[customer_id]
        if profile.counterparties and self._rng.random() < 0.7:
            counterparty = self._rng.choice(list(profile.counterparties))
        else:
            # Select new counterparty
            available = [cid for cid in self.customers.keys() if cid != customer_id and cid not in (exclude or set())]
            if available:
                counterparty = self._rng.choice(available)
            else:
                counterparty = self._rng.choice([cid for cid in self.customers.keys() if cid != customer_id])
        
        return counterparty
    
    def _update_transaction_graph(
        self, 
        sender_id: str, 
        receiver_id: str, 
        amount: float
    ) -> None:
        """
        Update transaction graph state.
        
        Args:
            sender_id: Sender customer ID
            receiver_id: Receiver customer ID
            amount: Transaction amount
        """
        # Update edge counts
        self.transaction_graph[sender_id][receiver_id] += 1
        
        # Update amount history
        self.transaction_amounts[sender_id][receiver_id].append(amount)
        
        # Update counterparty sets
        self.customers[sender_id].counterparties.add(receiver_id)
        self.customers[receiver_id].counterparties.add(sender_id)
    
    def _simulate_day(
        self,
        day: int,
        start_date: datetime,
        max_transactions: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Simulate transactions using a continuous-time Monte Carlo ledger.

        Instead of batching per-customer-per-day with random hours, this uses
        non-homogeneous Poisson inter-arrival times across all customers via a
        global event queue. Each customer has an independent exponential clock
        whose rate depends on their transaction frequency. Transaction amounts
        are conditioned on current balance, and outflows are capped so balance
        never goes below zero. Tier balance caps are enforced post-update.

        Args:
            day: Day number in simulation
            start_date: Simulation start date
            max_transactions: Optional cap on transactions generated this day

        Returns:
            List of transaction records with AML features
        """
        day_start = start_date + timedelta(days=day)
        day_end = day_start + timedelta(days=1)
        daily_transactions = []

        # --- 1. Build per-customer event queue for this day ---
        # Each customer gets their next tx time sampled from their exponential clock.
        # We interleave all customers globally so transactions are temporally mixed.
        if not hasattr(self, "_next_tx_time"):
            self._next_tx_time = {}

        event_queue: list[tuple[datetime, str]] = []  # (tx_time, customer_id)

        for customer_id, profile in self.customers.items():
            # Determine first tx of the day from the inter-arrival clock
            if customer_id in self._next_tx_time:
                last_tx = self._next_tx_time[customer_id]
            else:
                last_tx = day_start

            # Generate tx times throughout the day via renewal process
            t = last_tx
            while True:
                t = self._generate_transaction_time(customer_id, t)
                if t >= day_end:
                    break
                if t >= day_start or last_tx < day_start:
                    event_queue.append((t, customer_id))
            # Save the residual clock for the next day
            self._next_tx_time[customer_id] = t

        # Sort globally — this is the key step that creates temporal mixing
        event_queue.sort(key=lambda x: x[0])

        # --- 2. Process events in global temporal order ---
        for tx_time, customer_id in event_queue:
            if max_transactions is not None and len(daily_transactions) >= max_transactions:
                break

            profile = self.customers[customer_id]

            # Determine transaction type with balance-dependent directionality
            balance_ratio = profile.current_balance / max(profile.avg_transaction_amount, 1.0)
            # Low balance → higher inflow probability; high balance → higher outflow probability
            if balance_ratio < 2.0:
                tx_type = self._rng.choice(
                    ['deposit', 'withdrawal', 'transfer'],
                    p=[0.55, 0.15, 0.30]
                )
            elif balance_ratio > 10.0:
                tx_type = self._rng.choice(
                    ['deposit', 'withdrawal', 'transfer'],
                    p=[0.15, 0.35, 0.50]
                )
            else:
                tx_type = self._rng.choice(
                    ['deposit', 'withdrawal', 'transfer'],
                    p=[0.30, 0.25, 0.45]
                )

            # Generate amount — capped by tier and constrained by balance for outflows
            amount = self._generate_transaction_amount(customer_id)

            # Enforce regulatory per-transaction cap
            amount = self._enforce_transaction_cap(amount, profile.wallet_tier)

            # Balance constraint: outflows cannot exceed current balance
            if tx_type in ('withdrawal', 'transfer'):
                if amount > profile.current_balance:
                    amount = profile.current_balance * self._rng.uniform(0.3, 0.9)
                    if amount < 10.0:
                        continue  # skip micro outflows that would be noise

            # Select counterparty for transfers
            counterparty = None
            if tx_type == 'transfer':
                counterparty = self._select_counterparty(customer_id)

            # Update balance
            pre_balance = profile.current_balance
            if tx_type == 'deposit':
                profile.current_balance += amount
            elif tx_type in ('withdrawal', 'transfer'):
                profile.current_balance -= amount

            # Enforce tier balance cap
            profile.current_balance = self._enforce_balance_cap(
                profile.current_balance, profile.wallet_tier
            )

            # Update transaction graph for transfers
            if tx_type == 'transfer' and counterparty:
                self._update_transaction_graph(customer_id, counterparty, amount)
                counterparty_profile = self.customers.get(counterparty)
                if counterparty_profile:
                    counterparty_profile.current_balance += amount
                    counterparty_profile.current_balance = self._enforce_balance_cap(
                        counterparty_profile.current_balance, counterparty_profile.wallet_tier
                    )

            # Update device/location state
            if self._rng.random() < 0.05:
                profile.device_id = f"DEV_{self._rng.integers(100000, 999999)}"
            profile.device_history.append(profile.device_id)

            if self._rng.random() < 0.1:
                locations = ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret", "Kiambu"]
                profile.primary_location = self._rng.choice(locations)
            profile.location_history.append(profile.primary_location)

            # Store transaction in profile state
            tx_record = {
                'timestamp': tx_time,
                'type': tx_type,
                'amount': amount,
                'counterparty': counterparty,
            }
            profile.recent_transactions.append(tx_record)

            # Compute all AML features
            features = self._compute_all_features(customer_id, amount, tx_time, tx_type)

            # Create complete transaction record
            transaction_record = {
                'transaction_id': f"TXN_{self._transaction_counter:010d}",
                'customer_id': customer_id,
                'counterparty_id': counterparty or '',
                'timestamp': tx_time,
                'transaction_type': tx_type,
                'amount': amount,
                'post_tx_balance': profile.current_balance,
                'customer_name': profile.customer_name,
                'email': profile.email,
                'tax_id': profile.tax_id,
                'device_age_days': profile.device_age_days,
                'sim_match_status': profile.sim_match_status,
                'wallet_tier_encoded': profile.wallet_tier_encoded,
                'kyc_level_encoded': profile.kyc_level,
                'prev_fraud_flag_count_90d': profile.prev_fraud_flag_count_90d,
                'receiver_id': counterparty or '',
                'sender_county': profile.primary_location,
                'receiver_county': self.customers[counterparty].primary_location
                if counterparty and counterparty in self.customers
                else '',
                'anomaly_flag': False,
                'anomaly_type': None,
                'tier': profile.wallet_tier.value,
                'archetype': profile.archetype.value,
                'direction': 'inflow' if tx_type == 'deposit' else 'outflow',
                **features,
            }

            daily_transactions.append(transaction_record)
            self._transaction_counter += 1

        # Update daily balances for all customers
        for customer_id, profile in self.customers.items():
            profile.daily_balances.append(profile.current_balance)

        return daily_transactions
    
    def _compute_all_features(
        self, 
        customer_id: str, 
        amount: float,
        current_time: datetime,
        tx_type: str
    ) -> Dict[str, Any]:
        """
        Compute all AML features for a transaction.
        
        Args:
            customer_id: Customer identifier
            amount: Transaction amount
            current_time: Transaction timestamp
            tx_type: Transaction type
            
        Returns:
            Dictionary of all AML features
        """
        features = {}
        
        # Tier 1: Real-time velocity features
        velocity_features = self._get_velocity_features(customer_id, current_time)
        features.update(velocity_features)
        
        # Tier 1: Balance features
        balance_features = self._compute_balance_features(customer_id, current_time)
        features.update(balance_features)
        
        # Tier 1: Amount pattern features
        amount_features = self._get_amount_pattern_features(customer_id, amount, current_time)
        features.update(amount_features)
        
        # TVAE Hybrid v2.0 - pass_through_ratio removed (not in 21-feature schema)
        # Mule detection now uses fan_in_fan_out_ratio and balance_retention_ratio
        
        # Tier 1: Network features
        network_features = self._get_network_features(customer_id)
        features.update(network_features)
        
        # Tier 2: Temporal features
        temporal_features = self._get_temporal_features(customer_id, current_time)
        features.update(temporal_features)
        
        # TVAE Hybrid v2.0 - Legacy tier 2/3 features removed
        # These are now computed by CustomerFeatureEngineer
        
        return features
    
    def _get_rolling_avg_amount(self, customer_id: str, days: int) -> float:
        """Calculate rolling average transaction amount."""
        profile = self.customers[customer_id]
        cutoff = datetime.now() - timedelta(days=days)  # Simplified
        recent_amounts = [
            tx['amount'] for tx in profile.recent_transactions
            if len(profile.recent_transactions) > 0
        ]
        return np.mean(recent_amounts) if recent_amounts else profile.avg_transaction_amount
    
    def _get_rolling_net_flow(self, customer_id: str, days: int) -> float:
        """Calculate rolling net flow (inflows - outflows)."""
        profile = self.customers[customer_id]
        recent_txs = list(profile.recent_transactions)[-50:]  # Simplified
        inflows = sum(tx['amount'] for tx in recent_txs if tx['type'] == 'deposit')
        outflows = sum(tx['amount'] for tx in recent_txs if tx['type'] in ['withdrawal', 'transfer'])
        return inflows - outflows
    
    
    def _get_behavioral_shift(self, customer_id: str) -> float:
        """Calculate behavioral shift score from baseline."""
        profile = self.customers[customer_id]
        # Simplified: compare recent activity to persona baseline
        recent_tx_count = len(profile.recent_transactions)
        expected_count = profile.transaction_frequency * 7  # Weekly expected
        shift = abs(recent_tx_count - expected_count) / (expected_count + 1e-6)
        return min(shift, 1.0)
    
    def generate(self) -> pl.DataFrame:
        """
        Generate complete stateful synthetic transaction dataset with AML features.
        
        This method orchestrates the entire simulation:
        1. Generates customer profiles with behavioral personas
        2. Simulates day-by-day transactions for the configured period
        3. Maintains rolling balance states and transaction graph
        4. Computes Tier 1, Tier 2, and Tier 3 AML features in real-time
        
        Returns:
            Polars DataFrame containing transaction records with comprehensive AML features
        """
        target = self.config.target_transactions
        if target is not None:
            logger.info(
                "Starting stateful simulation for up to %s days (target=%s transactions)",
                self.config.num_days,
                target,
            )
        else:
            logger.info(f"Starting stateful simulation for {self.config.num_days} days")
        
        # Generate customer profiles
        self._generate_customer_profiles()
        
        # Initialize simulation
        start_date = datetime(2025, 1, 1)
        all_transactions = []
        
        # Simulate each day
        for day in range(self.config.num_days):
            if target is not None and len(all_transactions) >= target:
                break
            remaining = None if target is None else target - len(all_transactions)
            daily_txs = self._simulate_day(day, start_date, max_transactions=remaining)
            all_transactions.extend(daily_txs)
            
            if (day + 1) % 10 == 0:
                logger.info(f"Completed day {day + 1}/{self.config.num_days}, total transactions: {len(all_transactions)}")
        
        if target is not None and len(all_transactions) > target:
            all_transactions = all_transactions[:target]
        
        # Convert to Polars DataFrame
        df = pl.DataFrame(all_transactions)
        
        # Sort by timestamp
        df = df.sort("timestamp")
        
        logger.info(
            f"Generated {len(df)} transactions with {len(df.columns)} features "
            f"for {len(self.customers)} customers over {self.config.num_days} days"
        )
        
        return df
    
    def generate_normalized(
        self,
        anomaly_ratio: Optional[float] = None,
        anomaly_seed: Optional[int] = None,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Generate normalized customer and transaction DataFrames with proper primary/foreign keys.

        Optionally injects anomalies on the combined dataset before splitting,
        ensuring all INJECTABLE_FEATURES (aggregate columns) are available to
        the anomaly injector. Aggregate columns are stripped from the returned
        transaction DataFrame regardless.

        Args:
            anomaly_ratio: If set, inject anomalies at this ratio (0.0-1.0).
                Uses FinancialAnomalyInjector with the combined DataFrame.
            anomaly_seed: RNG seed for anomaly injection (defaults to 42).

        Returns:
            Tuple of (customers_df, transactions_df) Polars DataFrames
        """
        # Generate the combined dataset first
        logger.info("Generating combined AML dataset …")
        combined_df = self.generate()

        # --- Anomaly injection on the combined DataFrame ------------------
        if anomaly_ratio is not None and anomaly_ratio > 0:
            logger.info(
                "Injecting anomalies at ratio=%.4f on combined dataset …",
                anomaly_ratio,
            )
            seed = anomaly_seed or 42
            injector = FinancialAnomalyInjector(
                InjectorConfig(anomaly_ratio=anomaly_ratio, seed=seed)
            )
            combined_df = injector.inject(combined_df)
            actual_ratio = (
                combined_df["anomaly_flag"].cast(pl.Float64).mean()
                if "anomaly_flag" in combined_df.columns
                else 0.0
            )
            logger.info("Anomaly injection complete: actual_ratio=%.4f", actual_ratio)
        # -----------------------------------------------------------------

        # Extract unique customer records
        customer_cols = [
            "customer_id",
            "customer_name",
            "email",
            "tax_id",
            "kyc_level_encoded",
            "wallet_tier_encoded",
            "device_age_days",
            "sim_match_status",
            "prev_fraud_flag_count_90d"
        ]
        
        # Add currency if it exists, otherwise add default
        if "currency" in combined_df.columns:
            customer_cols.append("currency")
        
        # Only select columns that exist
        available_customer_cols = [col for col in customer_cols if col in combined_df.columns]
        customers_df = combined_df.select(available_customer_cols).unique(subset=["customer_id"])
        
        # Add currency with default value if missing
        if "currency" not in customers_df.columns:
            customers_df = customers_df.with_columns([
                pl.lit("KES").alias("currency")
            ])
        
        # Add registration_date and customer_tier (derived from wallet_tier_encoded)
        customers_df = customers_df.with_columns([
            pl.lit(datetime(2024, 1, 1)).alias("registration_date"),
            pl.col("wallet_tier_encoded").alias("customer_tier")
        ])
        
        # Extract transaction records (remove customer-specific static columns)
        customer_static_cols = {
            "customer_name", "email", "tax_id", "currency",
            "kyc_level_encoded", "wallet_tier_encoded", "device_age_days",
            "sim_match_status", "prev_fraud_flag_count_90d"
        }
        
        # TVAE Hybrid v2.0 - Remove legacy aggregate/rolling features
        # These are now computed by CustomerFeatureEngineer
        aggregate_cols = {
            "tx_count_7d", "volume_7d", "night_tx_ratio", "rapid_tx_ratio", 
            "volume_7d_vs_30d_ratio", "distinct_counterparties_7d", 
            "fan_in_fan_out_ratio", "close_to_limit_ratio", 
            "balance_retention_ratio", "amount_roundness"
        }
        
        # Keep only core transaction columns
        transaction_cols = [col for col in combined_df.columns 
                          if col not in customer_static_cols and col not in aggregate_cols]
        
        transactions_df = combined_df.select(transaction_cols)
        
        # Ensure transaction_type column exists (derive from existing columns if needed)
        if "transaction_type" not in transactions_df.columns:
            # Derive transaction_type from amount patterns or use default
            transactions_df = transactions_df.with_columns([
                pl.lit("Send Money").alias("transaction_type")
            ])
        
        # Ensure required columns exist
        required_cols = ["transaction_id", "counterparty_id"]
        for col in required_cols:
            if col not in transactions_df.columns:
                if col == "transaction_id":
                    transactions_df = transactions_df.with_columns([
                        pl.concat_str(["TXN_", pl.col("customer_id").rank().cast(pl.String)])
                        .alias("transaction_id")
                    ])
                elif col == "counterparty_id":
                    transactions_df = transactions_df.with_columns([
                        pl.lit("COUNTERPARTY_").alias("counterparty_id")
                    ])
        
        logger.info(
            f"Generated normalized schema: {len(customers_df)} customers, "
            f"{len(transactions_df)} transactions"
        )
        
        return customers_df, transactions_df
    
    def generate_summary_statistics(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for the AML dataset.
        
        Args:
            df: Polars DataFrame generated by this class
            
        Returns:
            Dictionary containing summary statistics
        """
        numeric_cols = df.select(pl.col(pl.Float64, pl.Int64)).columns
        
        stats_dict = {
            "num_transactions": len(df),
            "num_customers": df["customer_id"].n_unique(),
            "date_range": {
                "start": df["timestamp"].min().isoformat(),
                "end": df["timestamp"].max().isoformat()
            },
            "feature_statistics": {}
        }
        
        for col in numeric_cols:
            col_stats = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "median": float(df[col].median())
            }
            stats_dict["feature_statistics"][col] = col_stats
        
        return stats_dict
    
    def generate_and_ingest_to_bronze(
        self, 
        bronze_layer: Optional['BronzeLayer'] = None,
        partition_key: Optional[str] = None
    ) -> Tuple[pl.DataFrame, str]:
        """
        Generate AML synthetic data and ingest to Bronze layer with lineage tracking.
        
        This method integrates with the existing Bronze layer architecture, ensuring
        that AML feature-rich synthetic data is properly tagged with synthetic_flag=True
        and follows the same lineage tracking as real data.
        
        Args:
            bronze_layer: BronzeLayer instance for ingestion. If None, creates default.
            partition_key: Optional partition key for Bronze layer storage
            
        Returns:
            Tuple of (synthetic_dataframe, bronze_parquet_path)
        """
        from src.data.bronze import BronzeLayer
        
        logger.info("Generating AML synthetic data and ingesting to Bronze layer")
        
        # Generate AML synthetic data
        aml_df = self.generate()
        
        # Initialize Bronze layer if not provided
        if bronze_layer is None:
            bronze_layer = BronzeLayer()
        
        # Map AML generator columns to Bronze schema expectations
        # Bronze expects: customer_id, customer_name, email, tax_id, currency, amount, timestamp
        bronze_df = aml_df.rename({
            "customer_id": "customer_id",
            "counterparty_id": "customer_name",
            "transaction_type": "currency",  # Using transaction_type as currency placeholder
            "amount": "amount",
            "timestamp": "timestamp"
        })
        
        # Add missing Bronze schema columns
        bronze_df = bronze_df.with_columns([
            pl.lit(None, dtype=pl.String).alias("email"),
            pl.lit(None, dtype=pl.String).alias("tax_id"),
            pl.lit("KES", dtype=pl.String).alias("currency")  # Override with actual currency
        ])
        
        # Ensure timestamp is in string format for Bronze compatibility
        bronze_df = bronze_df.with_columns([
            pl.col("timestamp").dt.strftime("%Y-%m-%dT%H:%M:%S%.6f").alias("timestamp")
        ])
        
        # Ingest to Bronze layer
        bronze_path = bronze_layer.ingest_synthetic_data(
            bronze_df,
            source_table="aml_synthetic_transactions",
            partition_key=partition_key
        )
        
        logger.info(f"AML synthetic data ingested to Bronze layer at {bronze_path}")
        return aml_df, bronze_path
    
    def export_customer_metadata(
        self, 
        output_path: str = "data/customers_metadata.csv"
    ) -> pl.DataFrame:
        """
        Export customer metadata to CSV for MRM audit trails.
        
        This method exports customer metadata including user_id (hashed), archetype,
        registration date, tier, and baseline parameters as specified in MRM requirements.
        
        Args:
            output_path: Path to output CSV file
            
        Returns:
            Polars DataFrame with customer metadata
        """
        if not self.customers:
            logger.warning("No customer profiles generated. Run generate() first.")
            return pl.DataFrame()
        
        # Generate registration dates (simulate realistic registration over past 5 years)
        registration_dates = []
        for customer_id in self.customers.keys():
            days_ago = int(self._rng.integers(0, 1825))  # 0 to 5 years ago (convert to int)
            reg_date = datetime.now() - timedelta(days=days_ago)
            registration_dates.append(reg_date)
        
        # Build customer metadata dataframe
        metadata_data = []
        for i, (customer_id, profile) in enumerate(self.customers.items()):
            metadata = {
                "user_id": customer_id,  # Already hashed if PII hashing enabled
                "archetype": profile.archetype.value,
                "registration_date": registration_dates[i],
                "tier": profile.wallet_tier.value,
                "baseline_tx_per_year": int(profile.transaction_frequency * 365),
                "baseline_avg_amount": profile.avg_transaction_amount,
                "initial_balance": profile.initial_balance,
            }
            metadata_data.append(metadata)
        
        df = pl.DataFrame(metadata_data)
        df = df.sort("registration_date")
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Export to CSV
        df.write_csv(output_path)
        logger.info(f"Exported customer metadata to {output_path}")
        
        # Log summary statistics
        self._log_customer_metadata_summary(df)
        
        return df
    
    def _log_customer_metadata_summary(self, df: pl.DataFrame) -> None:
        """Log summary statistics for customer metadata."""
        logger.info("=== Customer Metadata Summary ===")
        logger.info(f"Total customers: {len(df)}")
        
        # Archetype distribution
        archetype_counts = df.group_by("archetype").count().sort("count", descending=True)
        logger.info("Archetype distribution:")
        for row in archetype_counts.iter_rows(named=True):
            logger.info(f"  {row['archetype']}: {row['count']} ({row['count']/len(df)*100:.1f}%)")
        
        # Tier distribution
        tier_counts = df.group_by("tier").count().sort("count", descending=True)
        logger.info("Tier distribution:")
        for row in tier_counts.iter_rows(named=True):
            logger.info(f"  {row['tier']}: {row['count']} ({row['count']/len(df)*100:.1f}%)")
        
        # Transaction statistics
        logger.info(f"Transaction statistics:")
        logger.info(f"  Total baseline tx/year: {df['baseline_tx_per_year'].sum():,}")
        logger.info(f"  Avg tx/year per customer: {df['baseline_tx_per_year'].mean():.1f}")
        logger.info(f"  Avg transaction amount: KES {df['baseline_avg_amount'].mean():.2f}")
        logger.info(f"  Total initial balance: KES {df['initial_balance'].sum():,.2f}")
        
        # Registration date range
        logger.info(f"Registration date range:")
        logger.info(f"  Earliest: {df['registration_date'].min()}")
        logger.info(f"  Latest: {df['registration_date'].max()}")
    
    def validate_balance_integrity(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Validate mathematical balance integrity across all transactions.
        
        This critical validation ensures that the stateful simulation maintains
        mathematically perfect balance constraints - no negative balances and
        proper balance transitions.
        
        Args:
            df: Polars DataFrame generated by this class
            
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            "validation_status": "PASS",
            "violations": [],
            "statistics": {}
        }
        
        # Check for negative balances
        negative_balances = df.filter(pl.col("post_tx_balance") < 0)
        if len(negative_balances) > 0:
            validation_results["validation_status"] = "FAIL"
            validation_results["violations"].append({
                "type": "negative_balance",
                "count": len(negative_balances),
                "examples": negative_balances.select(["customer_id", "amount", "post_tx_balance"]).head(5).to_dicts()
            })
        
        # Check balance continuity for each customer
        balance_violations = []
        for customer_id in df["customer_id"].unique():
            customer_txs = df.filter(pl.col("customer_id") == customer_id).sort("timestamp")
            balances = customer_txs["post_tx_balance"].to_list()
            
            # Check for balance jumps that don't match transaction amounts
            for i in range(1, len(balances)):
                prev_balance = balances[i-1]
                curr_balance = balances[i]
                tx_amount = customer_txs["amount"][i]
                tx_type = customer_txs["transaction_type"][i]
                
                # Expected balance based on transaction type
                if tx_type == "deposit":
                    expected = prev_balance + tx_amount
                elif tx_type in ["withdrawal", "transfer"]:
                    expected = prev_balance - tx_amount
                else:
                    expected = prev_balance
                
                # Allow small floating point tolerance
                if abs(curr_balance - expected) > 0.01:
                    balance_violations.append({
                        "customer_id": customer_id,
                        "transaction_index": i,
                        "expected_balance": expected,
                        "actual_balance": curr_balance,
                        "difference": curr_balance - expected
                    })
        
        if balance_violations:
            validation_results["validation_status"] = "FAIL"
            validation_results["violations"].append({
                "type": "balance_continuity",
                "count": len(balance_violations),
                "examples": balance_violations[:5]
            })
        
        # Compute balance statistics
        validation_results["statistics"] = {
            "total_transactions": len(df),
            "unique_customers": df["customer_id"].n_unique(),
            "min_balance": float(df["post_tx_balance"].min()),
            "max_balance": float(df["post_tx_balance"].max()),
            "avg_balance": float(df["post_tx_balance"].mean()),
            "negative_balance_count": len(negative_balances),
            "balance_continuity_violations": len(balance_violations)
        }
        
        logger.info(
            f"Balance integrity validation: {validation_results['validation_status']}, "
            f"violations: {len(validation_results['violations'])}"
        )
        
        return validation_results
