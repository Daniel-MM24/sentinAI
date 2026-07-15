import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame, Series
from typing import Annotated

class SilverRecordSchema(pa.DataFrameModel):
    """
    Pandera schema to enforce strict data types and constraints
    for the Silver layer records.

    Includes AML-focused features from the new stateful generator:
    - Tier 1: Real-time velocity, balance patterns, amount patterns, network features
    - Tier 2: Temporal anomalies, device/location intelligence, rolling windows
    - Tier 3: Community detection, behavioral shifts, advanced analytics
    """
    customer_id: str = pa.Field(nullable=False)
    customer_name: str = pa.Field(nullable=True)
    email: str = pa.Field(nullable=True)
    tax_id: str = pa.Field(nullable=True)
    currency: str = pa.Field(nullable=True)
    amount: float = pa.Field(nullable=True, ge=0)
    timestamp: pl.Datetime("us", "UTC") = pa.Field(nullable=True)
    receiver_id: str = pa.Field(nullable=True)
    sender_county: str = pa.Field(nullable=True)
    receiver_county: str = pa.Field(nullable=True)
    device_age_days: int = pa.Field(nullable=True)
    sim_match_status: bool = pa.Field(nullable=True)
    wallet_tier_encoded: int = pa.Field(nullable=True)
    kyc_level_encoded: int = pa.Field(nullable=True)
    prev_fraud_flag_count_90d: int = pa.Field(nullable=True)
    anomaly_flag: bool = pa.Field(nullable=True)
    anomaly_type: str = pa.Field(nullable=True)

    # AML Tier 1 Features - Real-time velocity
    tx_count_1h: int = pa.Field(nullable=True)
    tx_count_24h: int = pa.Field(nullable=True)
    amount_sum_24h: float = pa.Field(nullable=True)
    amount_vs_profile_avg: float = pa.Field(nullable=True)
    time_since_last_tx: float = pa.Field(nullable=True)

    # AML Tier 1 Features - Balance patterns
    current_balance: float = pa.Field(nullable=True)
    min_balance_30d: float = pa.Field(nullable=True)
    max_balance_30d: float = pa.Field(nullable=True)
    avg_balance_30d: float = pa.Field(nullable=True)
    balance_volatility_30d: float = pa.Field(nullable=True)
    balance_retention_ratio: float = pa.Field(nullable=True)
    zero_balance_frequency: int = pa.Field(nullable=True)

    # AML Tier 2 Features - Amount patterns
    amount_roundness: int = pa.Field(nullable=True)
    amount_just_below_threshold: bool = pa.Field(nullable=True)
    similar_amount_count_24h: int = pa.Field(nullable=True)
    identical_amount_count_24h: int = pa.Field(nullable=True)
    structuring_amount_entropy: float = pa.Field(nullable=True)

    # AML Tier 1 Features - Network features
    pass_through_ratio: float = pa.Field(nullable=True)
    degree_centrality: float = pa.Field(nullable=True, ge=0, le=1)
    in_degree: int = pa.Field(nullable=True)
    out_degree: int = pa.Field(nullable=True)
    funnel_score: float = pa.Field(nullable=True)
    reciprocity_ratio: float = pa.Field(nullable=True)

    # AML Tier 2 Features - Temporal anomalies
    burst_ratio: float = pa.Field(nullable=True)
    velocity_change_pct: float = pa.Field(nullable=True)
    hour_of_day: int = pa.Field(nullable=True, ge=0, le=23)
    day_of_week: int = pa.Field(nullable=True, ge=0, le=6)
    is_anomalous_hour: bool = pa.Field(nullable=True)

    # AML Tier 2 Features - Business intelligence
    new_relationships_7d: int = pa.Field(nullable=True)
    balance_depletion_rate: float = pa.Field(nullable=True)
    device_changes_7d: int = pa.Field(nullable=True)
    location_entropy: float = pa.Field(nullable=True)
    device_change_flag: bool = pa.Field(nullable=True)

    # AML Tier 3 Features - Advanced analytics
    rolling_avg_tx_amount_30d: float = pa.Field(nullable=True)
    rolling_net_flow_7d: float = pa.Field(nullable=True)
    community_id: int = pa.Field(nullable=True)
    behavioral_shift_score: float = pa.Field(nullable=True)

    # Legacy fields for migration compatibility
    transaction_amount: float = pa.Field(nullable=True)
    account_balance: float = pa.Field(nullable=True)
    transaction_count: int = pa.Field(nullable=True)
    price_impact: float = pa.Field(nullable=True)
    liquidity_score: float = pa.Field(nullable=True)
    bid_ask_spread: float = pa.Field(nullable=True)

    # Categorical/regulatory fields
    counterparty_risk_tier: str = pa.Field(nullable=True)
    source_table: str = pa.Field(nullable=True)
    ingestion_date: str = pa.Field(nullable=True)
    source_type: str = pa.Field(nullable=True)
    synthetic_flag: bool = pa.Field(nullable=True)
    regulatory_report_status: str = pa.Field(nullable=True)
    data_provenance_hash: str = pa.Field(nullable=True)


class GoldFeatureSchema(pa.DataFrameModel):
    """
    Pandera schema to enforce strict data types and constraints
    for the Gold Feature Store.

    Includes AML-focused features from the new stateful generator:
    - Tier 1: Real-time velocity, balance patterns, amount patterns, network features
    - Tier 2: Temporal anomalies, device/location intelligence, rolling windows
    - Tier 3: Community detection, behavioral shifts, advanced analytics
    """
    customer_id: str = pa.Field(nullable=False)
    amount: float = pa.Field(nullable=True)
    timestamp: pl.Datetime("us", "UTC") = pa.Field(nullable=True)
    partition_date: str = pa.Field(nullable=True)
    anomaly_case_id: str = pa.Field(nullable=True)

    # Base fields from model card
    log_amount: float = pa.Field(nullable=True)
    hour_of_day: int = pa.Field(nullable=True)
    day_of_week: int = pa.Field(nullable=True)
    is_weekend: bool = pa.Field(nullable=True)
    device_age_days: int = pa.Field(nullable=True)
    sim_match_status: bool = pa.Field(nullable=True)
    wallet_tier_encoded: int = pa.Field(nullable=True)
    kyc_level_encoded: int = pa.Field(nullable=True)
    prev_fraud_flag_count_90d: int = pa.Field(nullable=True)

    # Feature Engineering Outputs
    transaction_velocity: float = pa.Field(nullable=True)
    mean_transaction_amount: float = pa.Field(nullable=True)
    z_score_deviation: float = pa.Field(nullable=True)
    amount_near_threshold: int = pa.Field(nullable=True)
    is_round_number_100k: int = pa.Field(nullable=True)
    is_stk_push: int = pa.Field(nullable=True)
    is_b2c: int = pa.Field(nullable=True)

    # AML Tier 1 Features - Real-time velocity
    tx_count_1h: int = pa.Field(nullable=True)
    tx_count_24h: int = pa.Field(nullable=True)
    amount_sum_24h: float = pa.Field(nullable=True)
    amount_vs_profile_avg: float = pa.Field(nullable=True)
    time_since_last_tx: float = pa.Field(nullable=True)

    # AML Tier 1 Features - Balance patterns
    avg_balance_30d: float = pa.Field(nullable=True)
    balance_volatility_30d: float = pa.Field(nullable=True)
    current_balance: float = pa.Field(nullable=True)
    min_balance_30d: float = pa.Field(nullable=True)
    max_balance_30d: float = pa.Field(nullable=True)
    balance_retention_ratio: float = pa.Field(nullable=True)
    zero_balance_frequency: float = pa.Field(nullable=True)

    # AML Tier 1 Features - Amount patterns
    amount_roundness: float = pa.Field(nullable=True)
    amount_just_below_threshold: float = pa.Field(nullable=True)
    similar_amount_count_24h: int = pa.Field(nullable=True)
    identical_amount_count_24h: int = pa.Field(nullable=True)
    structuring_amount_entropy: float = pa.Field(nullable=True)

    # AML Tier 1 Features - Network features
    pass_through_ratio: float = pa.Field(nullable=True)
    degree_centrality: float = pa.Field(nullable=True)
    in_degree: int = pa.Field(nullable=True)
    out_degree: int = pa.Field(nullable=True)
    funnel_score: float = pa.Field(nullable=True)
    reciprocity_ratio: float = pa.Field(nullable=True)

    # AML Tier 2 Features - Temporal anomalies
    burst_ratio: float = pa.Field(nullable=True)
    velocity_change_pct: float = pa.Field(nullable=True)
    balance_depletion_rate: float = pa.Field(nullable=True)
    is_anomalous_hour: float = pa.Field(nullable=True)

    # AML Tier 2 Features - Device/location intelligence
    device_changes_7d: int = pa.Field(nullable=True)
    device_change_flag: float = pa.Field(nullable=True)
    location_entropy: float = pa.Field(nullable=True)

    # AML Tier 2 Features - Rolling windows
    rolling_avg_tx_amount_30d: float = pa.Field(nullable=True)
    rolling_net_flow_7d: float = pa.Field(nullable=True)
    new_relationships_7d: int = pa.Field(nullable=True)

    # AML Tier 3 Features - Advanced analytics
    community_id: float = pa.Field(nullable=True)
    behavioral_shift_score: float = pa.Field(nullable=True)

    # Additional AML generator fields
    transaction_id: str = pa.Field(nullable=True)
    counterparty_id: str = pa.Field(nullable=True)
    post_tx_balance: float = pa.Field(nullable=True)

    class Config:
        strict = False  # allow other columns to pass through


class SilverCompactSchema(SilverRecordSchema):
    """Minimal Silver schema for efficiency — drops nullable feature columns."""

    # Override: keep only mandatory regulatory + core fields
    customer_name: str = pa.Field(nullable=False)
    email: str = pa.Field(nullable=False)
    tax_id: str = pa.Field(nullable=False)
    currency: str = pa.Field(nullable=False)
    amount: float = pa.Field(nullable=False, ge=0)
    timestamp: pl.Datetime("us", "UTC") = pa.Field(nullable=False)
    receiver_id: str = pa.Field(nullable=False)
    sender_county: str = pa.Field(nullable=False)
    receiver_county: str = pa.Field(nullable=False)
    device_age_days: int = pa.Field(nullable=False)
    sim_match_status: bool = pa.Field(nullable=False)
    wallet_tier_encoded: int = pa.Field(nullable=False)
    kyc_level_encoded: int = pa.Field(nullable=False)
    prev_fraud_flag_count_90d: int = pa.Field(nullable=False)

    # AML Tier 1 Features - Real-time velocity
    tx_count_1h: int = pa.Field(nullable=False)
    tx_count_24h: int = pa.Field(nullable=False)
    amount_sum_24h: float = pa.Field(nullable=False)
    amount_vs_profile_avg: float = pa.Field(nullable=False)
    time_since_last_tx: float = pa.Field(nullable=False)

    # AML Tier 1 Features - Balance patterns
    current_balance: float = pa.Field(nullable=False)
    min_balance_30d: float = pa.Field(nullable=False)
    max_balance_30d: float = pa.Field(nullable=False)
    avg_balance_30d: float = pa.Field(nullable=False)
    balance_volatility_30d: float = pa.Field(nullable=False)
    balance_retention_ratio: float = pa.Field(nullable=False)
    zero_balance_frequency: int = pa.Field(nullable=False)

    # AML Tier 2 Features - Amount patterns
    amount_roundness: int = pa.Field(nullable=False)
    amount_just_below_threshold: bool = pa.Field(nullable=False)
    similar_amount_count_24h: int = pa.Field(nullable=False)
    identical_amount_count_24h: int = pa.Field(nullable=False)
    structuring_amount_entropy: float = pa.Field(nullable=False)

    # AML Tier 1 Features - Network features
    pass_through_ratio: float = pa.Field(nullable=False)
    degree_centrality: float = pa.Field(nullable=False, ge=0, le=1)
    in_degree: int = pa.Field(nullable=False)
    out_degree: int = pa.Field(nullable=False)
    funnel_score: float = pa.Field(nullable=False)
    reciprocity_ratio: float = pa.Field(nullable=False)

    # AML Tier 2 Features - Temporal anomalies
    burst_ratio: float = pa.Field(nullable=False)
    velocity_change_pct: float = pa.Field(nullable=False)
    hour_of_day: int = pa.Field(nullable=False, ge=0, le=23)
    day_of_week: int = pa.Field(nullable=False, ge=0, le=6)
    is_anomalous_hour: bool = pa.Field(nullable=False)

    # AML Tier 2 Features - Business intelligence
    new_relationships_7d: int = pa.Field(nullable=False)
    balance_depletion_rate: float = pa.Field(nullable=False)
    device_changes_7d: int = pa.Field(nullable=False)
    location_entropy: float = pa.Field(nullable=False)
    device_change_flag: bool = pa.Field(nullable=False)

    # AML Tier 3 Features - Advanced analytics
    rolling_avg_tx_amount_30d: float = pa.Field(nullable=False)
    rolling_net_flow_7d: float = pa.Field(nullable=False)
    community_id: int = pa.Field(nullable=False)
    behavioral_shift_score: float = pa.Field(nullable=False)

    # Legacy fields for migration compatibility
    transaction_amount: float = pa.Field(nullable=False)
    account_balance: float = pa.Field(nullable=False)
    transaction_count: int = pa.Field(nullable=False)
    price_impact: float = pa.Field(nullable=False)
    liquidity_score: float = pa.Field(nullable=False)
    bid_ask_spread: float = pa.Field(nullable=False)

    # Categorical/regulatory fields
    counterparty_risk_tier: str = pa.Field(nullable=False)
    source_table: str = pa.Field(nullable=False)
    ingestion_date: str = pa.Field(nullable=False)
    source_type: str = pa.Field(nullable=False)
    synthetic_flag: bool = pa.Field(nullable=False)
    regulatory_report_status: str = pa.Field(nullable=False)
    data_provenance_hash: str = pa.Field(nullable=False)
