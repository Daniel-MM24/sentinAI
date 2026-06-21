"""
Pydantic Schema Definitions for Medallion Architecture

This module defines type-safe Pydantic models for all data layers in the
Medallion architecture (Bronze/Silver/Gold). These schemas enforce strict
type constraints and validation rules for MRM compliance.

MRM Compliance: All data transformations must use these Pydantic models
to ensure type-safety and auditability.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, validator
from enum import Enum


class DataSourceType(str, Enum):
    """Enumeration of data source types for lineage tracking."""
    POSTGRESQL = "postgresql"
    SYNTHETIC = "synthetic"
    EXTERNAL_API = "external_api"


class BronzeTransactionRecord(BaseModel):
    """
    Pydantic model for Bronze layer raw transaction records.

    This represents the immutable raw logs ingested from PostgreSQL.
    All fields are optional at this layer since data is raw and unvalidated.
    """
    customer_id: Optional[str] = Field(None, description="Customer identifier")
    customer_name: Optional[str] = Field(None, description="Customer full name")
    email: Optional[str] = Field(None, description="Customer email address")
    tax_id: Optional[str] = Field(None, description="Tax identification number")
    currency: Optional[str] = Field(None, description="Transaction currency code")
    amount: Optional[float] = Field(None, description="Transaction amount")
    timestamp: Optional[datetime] = Field(None, description="Transaction timestamp")
    source_table: Optional[str] = Field(None, description="Source PostgreSQL table")
    ingestion_date: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Date of ingestion")
    source_type: Optional[DataSourceType] = Field(default=DataSourceType.POSTGRESQL, description="Data source type")
    synthetic_flag: bool = Field(default=False, description="Flag indicating synthetic data")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SilverRecord(BaseModel):
    """
    Pydantic model for Silver layer validated records.

    This represents cleaned, deduplicated, and validated data with
    strict type enforcement and business rules.
    """
    customer_id: str = Field(..., description="Customer identifier (required)")
    customer_name: Optional[str] = Field(None, description="Customer full name")
    email: Optional[str] = Field(None, description="Customer email address")
    tax_id: Optional[str] = Field(None, description="Tax identification number")
    currency: str = Field(default="USD", description="Transaction currency code")
    amount: float = Field(..., ge=0, description="Transaction amount (non-negative)")
    timestamp: datetime = Field(..., description="Transaction timestamp")
    golden_record_id: str = Field(..., description="Deterministic golden record ID")
    partition_date: str = Field(..., description="Date partition for storage")
    synthetic_flag: bool = Field(default=False, description="Flag indicating synthetic data")

    @validator('email')
    def validate_email(cls, v):
        """Basic email validation."""
        if v and '@' not in v:
            raise ValueError('Invalid email format')
        return v

    @validator('currency')
    def validate_currency(cls, v):
        """Currency code validation (ISO 4217)."""
        if v and len(v) != 3:
            raise ValueError('Currency must be 3-character ISO code')
        return v.upper()

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class GoldFeatureRecord(BaseModel):
    """
    Pydantic model for Gold layer feature records.

    This represents feature-rich, denormalized data for AI consumption
    with high-performance aggregations and engineered features.
    """
    customer_id: str = Field(..., description="Customer identifier (required)")
    amount: float = Field(..., ge=0, description="Transaction amount")
    timestamp: datetime = Field(..., description="Transaction timestamp")
    partition_date: str = Field(..., description="Date partition for storage")

    # Feature Engineering Outputs
    day_of_month: int = Field(..., ge=1, le=31, description="Day of month (1-31)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0-6)")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    transaction_velocity: float = Field(..., ge=0, description="Transaction velocity metric")
    clv: float = Field(..., ge=0, description="Customer lifetime value")
    mean_transaction_amount: float = Field(..., ge=0, description="Mean transaction amount")
    high_risk_amount: int = Field(..., ge=0, description="High risk amount flag")
    z_score_deviation: float = Field(..., description="Z-score deviation from mean")
    synthetic_flag: bool = Field(default=False, description="Flag indicating synthetic data")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DataQualityMetrics(BaseModel):
    """
    Pydantic model for data quality metrics.

    Captures quality metrics for MRM compliance and audit trails.
    """
    total_rows: int = Field(..., ge=0, description="Total number of rows")
    null_customer_id: int = Field(..., ge=0, description="Count of null customer_id")
    null_tax_id: int = Field(..., ge=0, description="Count of null tax_id")
    negative_amounts: int = Field(..., ge=0, description="Count of negative amounts")
    duplicate_records: int = Field(..., ge=0, description="Count of duplicate records")
    validation_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of validation")
    synthetic_flag: bool = Field(default=False, description="Flag indicating synthetic data")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TransformationMetadata(BaseModel):
    """
    Pydantic model for transformation metadata.

    Captures transformation details for lineage tracking and auditability.
    """
    job_name: str = Field(..., description="Name of the transformation job")
    run_id: str = Field(..., description="Unique run identifier")
    input_datasets: List[str] = Field(..., description="List of input dataset names")
    output_datasets: List[str] = Field(..., description="List of output dataset names")
    transformation_type: str = Field(..., description="Type of transformation")
    start_time: datetime = Field(default_factory=datetime.utcnow, description="Job start time")
    end_time: Optional[datetime] = Field(None, description="Job end time")
    status: str = Field(default="START", description="Job status (START/COMPLETE/FAIL)")
    input_rows: Optional[int] = Field(None, description="Number of input rows")
    output_rows: Optional[int] = Field(None, description="Number of output rows")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    synthetic_flag: bool = Field(default=False, description="Flag indicating synthetic data")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
