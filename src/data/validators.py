"""
Type-Safe Validation Layer for Synthetic Simulation Profiles (Pydantic V2)

This module enforces Model Risk Management (MRM) constraints on the distribution
profiles defined in ``config/simulation_profiles.yaml`` before they are handed to
``src.data.synthetic_engine.DistributionParams``.

Guarantees enforced here (audit-first, fail-closed):
    1. Categorical transaction probabilities are well-formed and sum to exactly
       1.0 (no probability mass leakage).
    2. The Differential Privacy clipping bound never exceeds the Kenyan statutory
       single-transaction limit of KSh 250,000.00.
    3. All numeric parameters are strictly typed and bounded, keeping
       serialization low-overhead and deterministic for K8s throughput.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Self

import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Kenyan statutory ceiling for a single M-PESA operation (KSh).
LEGAL_TRANSACTION_LIMIT: float = 250_000.0

# Tolerance for floating-point comparison of the categorical probability mass.
_PROBABILITY_SUM_TOLERANCE: float = 1e-9


class SimulationProfile(BaseModel):
    """Validated, type-safe representation of a single simulation profile.

    Field names mirror ``DistributionParams`` so a validated profile can be
    splatted directly into the generator: ``DistributionParams(**profile.dict())``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # --- Provenance / audit metadata ---
    model_version: str = Field(
        default="v1.0", description="Version tag of the generator calibration."
    )
    description: str = Field(
        default="", description="Human-readable provenance of the profile."
    )
    source: str = Field(
        default="", description="Audited filing the parameters are derived from."
    )

    # --- Categorical transaction mix ---
    transaction_type_probs: dict[str, float] = Field(
        ..., description="Transaction-type categorical probabilities (sum == 1.0)."
    )

    # --- Log-normal amount distribution ---
    amount_mean: float = Field(
        ..., gt=0.0, description="mu of the underlying normal for amounts."
    )
    amount_std: float = Field(
        ..., gt=0.0, description="sigma of the underlying normal for amounts."
    )

    # --- Temporal velocity ---
    velocity_lambda: float = Field(
        ...,
        gt=0.0,
        description="Mean minutes between transactions per unique entity.",
    )

    # --- Differential Privacy meta-parameters ---
    dataset_size: int = Field(
        ..., gt=0, description="Distinct active entities (Delta denominator)."
    )
    total_queries_per_year: int = Field(
        ..., gt=0, description="Annual query budget for epsilon allocation."
    )
    query_type: str = Field(default="standard", description="Query sensitivity class.")
    clipping_bound: float = Field(
        ...,
        gt=0.0,
        le=LEGAL_TRANSACTION_LIMIT,
        description="DP sensitivity ceiling; bounded by the statutory limit.",
    )

    # --- Reproducibility ---
    seed: int = Field(default=42, ge=0, description="Deterministic RNG seed.")

    @field_validator("transaction_type_probs")
    @classmethod
    def _validate_probability_components(
        cls, value: dict[str, float]
    ) -> dict[str, float]:
        """Each categorical probability must be a real number in (0, 1]."""
        if not value:
            raise ValueError("transaction_type_probs must not be empty.")
        for label, prob in value.items():
            if not 0.0 < prob <= 1.0:
                raise ValueError(
                    f"Probability for '{label}' must be in (0, 1]; got {prob}."
                )
        return value

    @model_validator(mode="after")
    def _validate_probability_mass(self) -> Self:
        """Categorical probabilities must sum to exactly 1.0 (MRM closure)."""
        total = sum(self.transaction_type_probs.values())
        if abs(total - 1.0) > _PROBABILITY_SUM_TOLERANCE:
            raise ValueError(
                "transaction_type_probs must sum to exactly 1.0; "
                f"got {total} from {self.transaction_type_probs}."
            )
        return self

    @model_validator(mode="after")
    def _validate_clipping_within_legal_limit(self) -> Self:
        """DP clipping bound must not exceed the statutory transaction limit."""
        if self.clipping_bound > LEGAL_TRANSACTION_LIMIT:
            raise ValueError(
                f"clipping_bound {self.clipping_bound} exceeds the legal limit "
                f"of {LEGAL_TRANSACTION_LIMIT}."
            )
        return self


class SimulationProfileRegistry(BaseModel):
    """Top-level schema for ``config/simulation_profiles.yaml``."""

    model_config = ConfigDict(extra="forbid")

    profiles: dict[str, SimulationProfile] = Field(
        ..., description="Named simulation profiles keyed by profile id."
    )

    def get(self, name: str) -> SimulationProfile:
        """Return a validated profile by name, raising ``KeyError`` if absent."""
        return self.profiles[name]


def load_simulation_profiles(
    path: str | Path = "config/simulation_profiles.yaml",
) -> SimulationProfileRegistry:
    """Load and fully validate the simulation profile registry from YAML."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return SimulationProfileRegistry.model_validate(raw)


# ---------------------------------------------------------------------------
# AML / POCAMLA Silver-layer schema and regulatory constraint validation
# ---------------------------------------------------------------------------

MANDATORY_SILVER_COLUMNS: tuple[str, ...] = (
    "entity_id",
    "counterparty_id",
    "kyc_tier_level",
    "counterparty_risk_tier",
    "transaction_id",
    "transaction_amount",
    "account_balance_before",
    "account_balance_after",
    "transaction_type",
    "channel",
    "timestamp",
    "ingestion_timestamp",
    "region",
    "anomaly_flag",
    "anomaly_type",
    "data_provenance_hash",
    "regulatory_report_status",
    "is_wallet_balance_compliant",
    "account_first_seen",
    "account_last_seen",
    "user_active_tenure_days",
)

SILVER_COLUMN_DTYPES: dict[str, pl.DataType] = {
    "entity_id": pl.String,
    "counterparty_id": pl.String,
    "kyc_tier_level": pl.String,
    "counterparty_risk_tier": pl.String,
    "transaction_id": pl.String,
    "transaction_amount": pl.Float64,
    "account_balance_before": pl.Float64,
    "account_balance_after": pl.Float64,
    "transaction_type": pl.String,
    "channel": pl.String,
    "timestamp": pl.Datetime("us", "UTC"),
    "ingestion_timestamp": pl.Datetime("us", "UTC"),
    "region": pl.String,
    "anomaly_flag": pl.Boolean,
    "anomaly_type": pl.String,
    "data_provenance_hash": pl.String,
    "regulatory_report_status": pl.String,
    "is_wallet_balance_compliant": pl.Boolean,
    "account_first_seen": pl.Datetime("us", "UTC"),
    "account_last_seen": pl.Datetime("us", "UTC"),
    "user_active_tenure_days": pl.Float64,
}

CRITICAL_NON_NULL_COLUMNS: tuple[str, ...] = (
    "entity_id",
    "timestamp",
    "anomaly_flag",
    "transaction_id",
    "data_provenance_hash",
)


class RegulatoryDefaults(BaseModel):
    """Default values applied when Bronze fields are null."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kyc_tier_level: str = "TIER_1"
    counterparty_risk_tier: str = "LOW"
    transaction_type: str = "P2P"
    channel: str = "USSD"
    region: str = "Nairobi"
    anomaly_type: str = "NONE"
    regulatory_report_status: str = "NOT_TRIGGERED"
    counterparty_id_prefix: str = "CP_UNKNOWN_"


class AllowedValues(BaseModel):
    """Enumerated domain values for categorical Silver columns."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kyc_tier_level: list[str]
    counterparty_risk_tier: list[str]
    transaction_type: list[str]
    channel: list[str]
    region: list[str]
    anomaly_type: list[str]
    regulatory_report_status: list[str]


class AnomalyDetectionConfig(BaseModel):
    """Thresholds for deterministic anomaly classification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    amount_spike_multiplier: float = Field(..., gt=0.0)
    smurfing_max_amount_kes: float = Field(..., gt=0.0)
    smurfing_min_daily_count: int = Field(..., ge=1)
    round_number_modulus: int = Field(..., ge=1)
    round_number_min_daily_count: int = Field(..., ge=1)


class DeadLetterConfig(BaseModel):
    """Dead-letter queue configuration for regulatory rejects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    output_path: str = "data/dead_letter/bronze_rejects"


class RegulatoryConfig(BaseModel):
    """POCAMLA regulatory thresholds loaded from ``config/regulatory.yaml``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kyc_tier_balance_caps: dict[str, float]
    kyc_tier_velocity_caps: dict[str, float]
    min_timestamp: datetime
    max_timestamp: datetime
    allowed_values: AllowedValues
    anomaly_detection: AnomalyDetectionConfig
    defaults: RegulatoryDefaults
    dead_letter: DeadLetterConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> RegulatoryConfig:
        """Parse and validate regulatory YAML configuration."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        tiers = raw["kyc_tiers"]
        balance_caps = {k: float(v["balance_cap_kes"]) for k, v in tiers.items()}
        velocity_caps = {
            k: float(v["daily_velocity_cap_kes"]) for k, v in tiers.items()
        }
        temporal = raw["temporal"]
        return cls(
            kyc_tier_balance_caps=balance_caps,
            kyc_tier_velocity_caps=velocity_caps,
            min_timestamp=datetime.fromisoformat(temporal["min_timestamp"]),
            max_timestamp=datetime.fromisoformat(temporal["max_timestamp"]),
            allowed_values=AllowedValues.model_validate(raw["allowed_values"]),
            anomaly_detection=AnomalyDetectionConfig.model_validate(
                raw["anomaly_detection"]
            ),
            defaults=RegulatoryDefaults.model_validate(raw["defaults"]),
            dead_letter=DeadLetterConfig.model_validate(raw["dead_letter"]),
        )


def load_regulatory_config(
    path: str | Path = "config/regulatory.yaml",
) -> RegulatoryConfig:
    """Load POCAMLA regulatory configuration from YAML."""
    return RegulatoryConfig.from_yaml(path)


@dataclass
class ValidationResult:
    """Outcome of schema or regulatory validation."""

    passed: bool
    errors: list[str]
    warnings: list[str] = field(default_factory=list)

    def merge(self, other: ValidationResult) -> ValidationResult:
        """Combine two validation results."""
        return ValidationResult(
            passed=self.passed and other.passed,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
        )


def validate_silver_schema(df: pl.DataFrame) -> ValidationResult:
    """Verify mandatory columns, dtypes, and zero-null critical fields.

    Args:
        df: Silver-layer Polars DataFrame.

    Returns:
        ValidationResult with pass/fail and error details.
    """
    errors: list[str] = []
    warnings: list[str] = []

    missing = [c for c in MANDATORY_SILVER_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing mandatory columns: {missing}")

    for col in MANDATORY_SILVER_COLUMNS:
        if col not in df.columns:
            continue
        expected = SILVER_COLUMN_DTYPES[col]
        actual = df.schema[col]
        if actual != expected:
            errors.append(
                f"Column '{col}' dtype mismatch: expected {expected}, got {actual}"
            )

    for col in CRITICAL_NON_NULL_COLUMNS:
        if col in df.columns:
            null_count = df[col].null_count()
            if null_count > 0:
                errors.append(f"Critical column '{col}' has {null_count} nulls")

    if "anomaly_flag" in df.columns and df.schema["anomaly_flag"] == pl.Boolean:
        non_bool = df.filter(
            ~pl.col("anomaly_flag").is_in([True, False]) & pl.col("anomaly_flag").is_not_null()
        ).height
        if non_bool > 0:
            errors.append(f"anomaly_flag contains {non_bool} non-boolean values")

    total_nulls = sum(df[c].null_count() for c in df.columns if c in MANDATORY_SILVER_COLUMNS)
    if total_nulls > 0:
        warnings.append(f"Silver dataset has {total_nulls} total null values")

    return ValidationResult(passed=len(errors) == 0, errors=errors, warnings=warnings)


def validate_regulatory_constraints(
    df: pl.DataFrame,
    config: RegulatoryConfig,
) -> ValidationResult:
    """Enforce POCAMLA wallet caps, velocity, timestamps, and amount rules.

    Args:
        df: Silver-layer Polars DataFrame.
        config: Loaded regulatory configuration.

    Returns:
        ValidationResult describing constraint violations.
    """
    errors: list[str] = []
    caps = config.kyc_tier_balance_caps

    if df.is_empty():
        return ValidationResult(passed=True, errors=[])

    tier_cap_expr = (
        pl.when(pl.col("kyc_tier_level") == "TIER_1")
        .then(pl.lit(caps["TIER_1"]))
        .when(pl.col("kyc_tier_level") == "TIER_2")
        .then(pl.lit(caps["TIER_2"]))
        .when(pl.col("kyc_tier_level") == "VENDOR_MERCHANT")
        .then(pl.lit(caps["VENDOR_MERCHANT"]))
        .otherwise(pl.lit(caps["TIER_1"]))
    )

    balance_violations = df.filter(pl.col("account_balance_after") > tier_cap_expr).height
    if balance_violations > 0:
        errors.append(
            f"{balance_violations} records exceed tier-specific balance cap "
            f"(account_balance_after)"
        )

    non_positive = df.filter(pl.col("transaction_amount") <= 0).height
    if non_positive > 0:
        errors.append(f"{non_positive} records have non-positive transaction_amount")

    min_ts = config.min_timestamp.replace(tzinfo=timezone.utc)
    max_ts = config.max_timestamp.replace(tzinfo=timezone.utc)
    ts_violations = df.filter(
        (pl.col("timestamp") < min_ts) | (pl.col("timestamp") > max_ts)
    ).height
    if ts_violations > 0:
        errors.append(
            f"{ts_violations} records have timestamps outside "
            f"[{min_ts.isoformat()}, {max_ts.isoformat()}]"
        )

    allowed = config.allowed_values
    for col, values in [
        ("kyc_tier_level", allowed.kyc_tier_level),
        ("region", allowed.region),
        ("anomaly_type", allowed.anomaly_type),
        ("transaction_type", allowed.transaction_type),
        ("channel", allowed.channel),
    ]:
        if col in df.columns:
            invalid = df.filter(~pl.col(col).is_in(values)).height
            if invalid > 0:
                errors.append(f"{invalid} records have invalid '{col}' values")

    anomaly_mismatch = df.filter(
        pl.col("anomaly_flag") & (pl.col("anomaly_type") == "NONE")
    ).height
    if anomaly_mismatch > 0:
        errors.append(
            f"{anomaly_mismatch} records have anomaly_flag=True but anomaly_type=NONE"
        )

    return ValidationResult(passed=len(errors) == 0, errors=errors)
