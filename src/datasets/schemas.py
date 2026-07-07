import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame, Series
from typing import Annotated

class SilverRecordSchema(pa.DataFrameModel):
    """
    Pandera schema to enforce strict data types and constraints
    for the Silver layer records.
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

    class Config:
        strict = False  # allow other columns to pass through


class GoldFeatureSchema(pa.DataFrameModel):
    """
    Pandera schema to enforce strict data types and constraints
    for the Gold Feature Store.
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

    class Config:
        strict = False  # allow other columns to pass through
