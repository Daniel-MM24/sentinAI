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
    
    # Feature Engineering Outputs
    day_of_month: int = pa.Field(nullable=True)
    day_of_week: int = pa.Field(nullable=True)
    hour_of_day: int = pa.Field(nullable=True)
    transaction_velocity: float = pa.Field(nullable=True)
    clv: float = pa.Field(nullable=True)
    mean_transaction_amount: float = pa.Field(nullable=True)
    high_risk_amount: int = pa.Field(nullable=True)
    z_score_deviation: float = pa.Field(nullable=True)

    class Config:
        strict = False  # allow other columns to pass through
