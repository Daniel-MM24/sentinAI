import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame, Series
from typing import Annotated
from datetime import datetime


class CustomerSchema(pa.DataFrameModel):
    """
    Customer dimension table - customer_id as primary key.
    
    Contains static customer profile information that doesn't change per transaction.
    This table serves as the customer master record with customer_id as the unique identifier.
    """
    customer_id: str = pa.Field(nullable=False)
    customer_name: str = pa.Field(nullable=False)
    email: str = pa.Field(nullable=False)
    tax_id: str = pa.Field(nullable=False)
    currency: str = pa.Field(nullable=False)
    kyc_level_encoded: int = pa.Field(nullable=False, ge=1, le=4)
    wallet_tier_encoded: int = pa.Field(nullable=False, ge=1, le=3)
    device_age_days: int = pa.Field(nullable=False, ge=0)
    sim_match_status: bool = pa.Field(nullable=False)
    prev_fraud_flag_count_90d: int = pa.Field(nullable=False, ge=0)
    registration_date: pl.Datetime("us", "UTC") = pa.Field(nullable=False)
    customer_tier: int = pa.Field(nullable=False, ge=1, le=4)
    
    class Config:
        strict = True
        coerce = True


class TransactionSchema(pa.DataFrameModel):
    """
    Transaction fact table - transaction_id as primary key, customer_id as foreign key.
    
    Contains raw transaction events. Each row represents a single transaction.
    customer_id links to CustomerSchema.customer_id (foreign key relationship).
    """
    transaction_id: str = pa.Field(nullable=False)
    customer_id: str = pa.Field(nullable=False)
    counterparty_id: str = pa.Field(nullable=False)
    amount: float = pa.Field(nullable=False, ge=0)
    timestamp: pl.Datetime("us", "UTC") = pa.Field(nullable=False)
    receiver_id: str = pa.Field(nullable=False)
    sender_county: str = pa.Field(nullable=False)
    receiver_county: str = pa.Field(nullable=False)
    transaction_type: str = pa.Field(nullable=False)
    anomaly_flag: bool = pa.Field(nullable=False)
    anomaly_type: str = pa.Field(nullable=True)
    anomaly_case_id: str = pa.Field(nullable=True)
    
    class Config:
        strict = True
        coerce = True


class CustomerFeaturesSchema(pa.DataFrameModel):
    """
    Customer features table - composite key (customer_id, feature_date), customer_id as foreign key.
    
    TVAE Hybrid Implementation (v2.0) - 10 downstream features computed by CustomerFeatureEngineer.
    
    Temporal Features (5): tx_count_7d, volume_7d, night_tx_ratio, rapid_tx_ratio, volume_7d_vs_30d_ratio
    Network Features (2): distinct_counterparties_7d, fan_in_fan_out_ratio
    Structuring Features (3): close_to_limit_ratio, balance_retention_ratio, amount_roundness
    """
    customer_id: str = pa.Field(nullable=False)
    feature_date: pl.Datetime("us", "UTC") = pa.Field(nullable=False)
    
    # Temporal Features (5)
    tx_count_7d: int = pa.Field(nullable=True, ge=0)
    volume_7d: float = pa.Field(nullable=True, ge=0)
    night_tx_ratio: float = pa.Field(nullable=True, ge=0, le=1)
    rapid_tx_ratio: float = pa.Field(nullable=True, ge=0, le=1)
    volume_7d_vs_30d_ratio: float = pa.Field(nullable=True)
    
    # Network Features (2)
    distinct_counterparties_7d: int = pa.Field(nullable=True, ge=0)
    fan_in_fan_out_ratio: float = pa.Field(nullable=True, ge=0)
    
    # Structuring Features (3)
    close_to_limit_ratio: float = pa.Field(nullable=True, ge=0, le=1)
    balance_retention_ratio: float = pa.Field(nullable=True, ge=0, le=1)
    amount_roundness: float = pa.Field(nullable=True, ge=0, le=1)

    class Config:
        strict = False  # allow other columns to pass through
        coerce = True


class SilverRecordSchema(pa.DataFrameModel):
    """
    Pandera schema to enforce strict data types and constraints
    for the Silver layer records.

    TVAE Hybrid Implementation (v2.0) - 21-feature schema optimized for tree-based AML models.
    
    Core Features (8): customer_id, tier, archetype, transaction_type, amount, timestamp, direction, balance
    Temporal Features (5): tx_count_7d, volume_7d, night_tx_ratio, rapid_tx_ratio, volume_7d_vs_30d_ratio
    Network Features (3): is_international, distinct_counterparties_7d, fan_in_fan_out_ratio
    Structuring Features (3): close_to_limit_ratio, balance_retention_ratio, amount_roundness
    Labels (2): is_launderer, aml_scenario
    """
    # Core Features (8)
    customer_id: str = pa.Field(nullable=False)
    tier: int = pa.Field(nullable=True)
    archetype: str = pa.Field(nullable=True)
    transaction_type: str = pa.Field(nullable=True)
    amount: float = pa.Field(nullable=True, ge=0)
    timestamp: pl.Datetime("us", "UTC") = pa.Field(nullable=True)
    direction: str = pa.Field(nullable=True)
    balance: float = pa.Field(nullable=True)
    
    # Temporal Features (5)
    tx_count_7d: int = pa.Field(nullable=True, ge=0)
    volume_7d: float = pa.Field(nullable=True, ge=0)
    night_tx_ratio: float = pa.Field(nullable=True, ge=0, le=1)
    rapid_tx_ratio: float = pa.Field(nullable=True, ge=0, le=1)
    volume_7d_vs_30d_ratio: float = pa.Field(nullable=True)
    
    # Network Features (3)
    is_international: bool = pa.Field(nullable=True)
    distinct_counterparties_7d: int = pa.Field(nullable=True, ge=0)
    fan_in_fan_out_ratio: float = pa.Field(nullable=True, ge=0)
    
    # Structuring Features (3)
    close_to_limit_ratio: float = pa.Field(nullable=True, ge=0, le=1)
    balance_retention_ratio: float = pa.Field(nullable=True, ge=0, le=1)
    amount_roundness: float = pa.Field(nullable=True, ge=0, le=1)
    
    # Labels (2)
    is_launderer: bool = pa.Field(nullable=True)
    aml_scenario: str = pa.Field(nullable=True)


class GoldFeatureSchema(pa.DataFrameModel):
    """
    Pandera schema to enforce strict data types and constraints
    for the Gold Feature Store.

    TVAE Hybrid Implementation (v2.0) - 21-feature schema optimized for tree-based AML models.
    
    Core Features (8): customer_id, tier, archetype, transaction_type, amount, timestamp, direction, balance
    Temporal Features (5): tx_count_7d, volume_7d, night_tx_ratio, rapid_tx_ratio, volume_7d_vs_30d_ratio
    Network Features (3): is_international, distinct_counterparties_7d, fan_in_fan_out_ratio
    Structuring Features (3): close_to_limit_ratio, balance_retention_ratio, amount_roundness
    Labels (2): is_launderer, aml_scenario
    """
    # Core Features (8)
    customer_id: str = pa.Field(nullable=False)
    tier: int = pa.Field(nullable=True)
    archetype: str = pa.Field(nullable=True)
    transaction_type: str = pa.Field(nullable=True)
    amount: float = pa.Field(nullable=True, ge=0)
    timestamp: pl.Datetime("us", "UTC") = pa.Field(nullable=True)
    direction: str = pa.Field(nullable=True)
    balance: float = pa.Field(nullable=True)
    
    # Temporal Features (5)
    tx_count_7d: int = pa.Field(nullable=True, ge=0)
    volume_7d: float = pa.Field(nullable=True, ge=0)
    night_tx_ratio: float = pa.Field(nullable=True, ge=0, le=1)
    rapid_tx_ratio: float = pa.Field(nullable=True, ge=0, le=1)
    volume_7d_vs_30d_ratio: float = pa.Field(nullable=True)
    
    # Network Features (3)
    is_international: bool = pa.Field(nullable=True)
    distinct_counterparties_7d: int = pa.Field(nullable=True, ge=0)
    fan_in_fan_out_ratio: float = pa.Field(nullable=True, ge=0)
    
    # Structuring Features (3)
    close_to_limit_ratio: float = pa.Field(nullable=True, ge=0, le=1)
    balance_retention_ratio: float = pa.Field(nullable=True, ge=0, le=1)
    amount_roundness: float = pa.Field(nullable=True, ge=0, le=1)
    
    # Labels (2)
    is_launderer: bool = pa.Field(nullable=True)
    aml_scenario: str = pa.Field(nullable=True)

    class Config:
        strict = False  # allow other columns to pass through
        coerce = True


# SilverCompactSchema removed - legacy schema no longer needed in TVAE Hybrid v2.0
