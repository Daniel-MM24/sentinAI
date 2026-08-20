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

    TVAE Hybrid v2.0 - 21-feature schema optimized for tree-based AML models.
    
    Core Features (8): customer_id, tier, archetype, transaction_type, amount, timestamp, direction, balance
    Temporal Features (5): tx_count_7d, volume_7d, night_tx_ratio, rapid_tx_ratio, volume_7d_vs_30d_ratio
    Network Features (3): is_international, distinct_counterparties_7d, fan_in_fan_out_ratio
    Structuring Features (3): close_to_limit_ratio, balance_retention_ratio, amount_roundness
    Labels (2): is_launderer, aml_scenario
    """
    customer_id: str = Field(..., description="Customer identifier (required)")
    tier: Optional[int] = Field(None, description="Customer tier (1-4)")
    archetype: Optional[str] = Field(None, description="Customer archetype")
    transaction_type: Optional[str] = Field(None, description="Transaction type")
    amount: float = Field(..., ge=0, description="Transaction amount")
    timestamp: datetime = Field(..., description="Transaction timestamp")
    direction: Optional[str] = Field(None, description="Transaction direction")
    balance: Optional[float] = Field(None, description="Account balance")
    
    # Temporal Features (5)
    tx_count_7d: Optional[int] = Field(None, ge=0, description="Transaction count in 7 days")
    volume_7d: Optional[float] = Field(None, ge=0, description="Transaction volume in 7 days")
    night_tx_ratio: Optional[float] = Field(None, ge=0, le=1, description="Night transaction ratio")
    rapid_tx_ratio: Optional[float] = Field(None, ge=0, le=1, description="Rapid transaction ratio")
    volume_7d_vs_30d_ratio: Optional[float] = Field(None, description="7d vs 30d volume ratio")
    
    # Network Features (3)
    is_international: Optional[bool] = Field(None, description="International transaction flag")
    distinct_counterparties_7d: Optional[int] = Field(None, ge=0, description="Distinct counterparties in 7 days")
    fan_in_fan_out_ratio: Optional[float] = Field(None, ge=0, description="Fan-in/fan-out ratio")
    
    # Structuring Features (3)
    close_to_limit_ratio: Optional[float] = Field(None, ge=0, le=1, description="Close to limit ratio")
    balance_retention_ratio: Optional[float] = Field(None, ge=0, le=1, description="Balance retention ratio")
    amount_roundness: Optional[float] = Field(None, ge=0, le=1, description="Amount roundness score")
    
    # Labels (2)
    is_launderer: Optional[bool] = Field(None, description="Money launderer label")
    aml_scenario: Optional[str] = Field(None, description="AML scenario type")
    
    partition_date: str = Field(..., description="Date partition for storage")
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
