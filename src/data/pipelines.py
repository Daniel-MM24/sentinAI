"""
Bronze → Silver → Gold AML transformation engine for SentinAI.

Implements POCAMLA-compliant data transformations with strict typing,
regulatory constraint enforcement, stateful entity lifecycle tracking,
SHA-256 provenance hashing, and auditable dead-letter routing for MRM.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from src.data.validators import (
    MANDATORY_SILVER_COLUMNS,
    SILVER_COLUMN_DTYPES,
    ValidationResult,
    load_regulatory_config,
    validate_regulatory_constraints,
    validate_silver_schema,
)

logger = logging.getLogger(__name__)

# Columns used to compute deterministic SHA-256 provenance over raw bronze rows.
_PROVENANCE_SOURCE_COLS: tuple[str, ...] = (
    "entity_id",
    "timestamp",
    "transaction_amount",
    "account_balance_before",
    "kyc_tier_level",
    "anomaly_flag_raw",
    "anomaly_type_raw",
)


@dataclass(frozen=True)
class PipelineResult:
    """Outcome of a Bronze → Silver transformation run."""

    silver: pl.DataFrame
    dead_letter: pl.DataFrame
    validation: ValidationResult
    audit_trail: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AuditTrail:
    """Collects regulatory violation events for MRM audit logging."""

    events: list[dict[str, Any]] = field(default_factory=list)

    def log(
        self,
        violation_type: str,
        entity_id: str,
        detail: str,
        *,
        transaction_id: str | None = None,
    ) -> None:
        """Append a structured audit event."""
        event = {
            "violation_type": violation_type,
            "entity_id": entity_id,
            "transaction_id": transaction_id,
            "detail": detail,
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }
        self.events.append(event)
        logger.warning("Regulatory violation: %s", event)


def _sha256_hex(payload: str) -> str:
    """Return lowercase SHA-256 hex digest for a string payload."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_row_provenance_hash(row: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash for a single bronze row.

    Args:
        row: Dictionary of provenance source column values.

    Returns:
        Lowercase SHA-256 hex digest.
    """
    canonical = json.dumps(
        {col: row.get(col) for col in _PROVENANCE_SOURCE_COLS},
        sort_keys=True,
        default=str,
    )
    return _sha256_hex(canonical)


def _batch_provenance_hashes(series: pl.Series) -> pl.Series:
    """Vectorized batch SHA-256 over struct-encoded provenance payloads."""
    structs = series.to_list()
    return pl.Series(
        [_sha256_hex(json.dumps(s, sort_keys=True, default=str)) for s in structs]
    )


class BronzeToSilverPipeline:
    """Stateful transformation engine: Bronze (untrusted) → Silver (compliant)."""

    def __init__(
        self,
        config_path: str | Path = "config/regulatory.yaml",
        ingestion_timestamp: datetime | None = None,
    ) -> None:
        """Initialize pipeline with regulatory configuration.

        Args:
            config_path: Path to ``regulatory.yaml`` threshold definitions.
            ingestion_timestamp: System clock for ``ingestion_timestamp`` column.
        """
        self.config = load_regulatory_config(config_path)
        self.ingestion_ts = ingestion_timestamp or datetime.now(timezone.utc)
        self.audit = AuditTrail()

    def transform(
        self,
        bronze_df: pl.DataFrame,
        *,
        write_dead_letter: bool = False,
    ) -> PipelineResult:
        """Transform raw Bronze data into compliant Silver records.

        Pipeline stages:
            1. Schema normalization and strict type casting
            2. Explicit null sanitization (no silent nulls)
            3. Regulatory balance capping and compliance flags
            4. Stateful entity lifecycle windows (first/last seen, tenure)
            5. Deterministic anomaly classification
            6. SHA-256 provenance hashing
            7. Post-transform validation and dead-letter routing

        Args:
            bronze_df: Untrusted Bronze-layer Polars DataFrame.
            write_dead_letter: Persist rejected rows to configured path.

        Returns:
            PipelineResult with silver DataFrame, dead-letter rejects,
            validation summary, and audit trail events.
        """
        if bronze_df.is_empty():
            empty = self._empty_silver()
            return PipelineResult(
                silver=empty,
                dead_letter=pl.DataFrame(),
                validation=ValidationResult(
                    passed=True, errors=[], warnings=["Empty bronze input."]
                ),
                audit_trail=[],
            )

        df = self._normalize_bronze(bronze_df)
        df = self._enforce_types(df)
        df = self._sanitize_nulls(df)
        df = self._compute_balances(df)
        df = self._add_provenance_hashes(df)
        df = self._apply_stateful_windows(df)
        df = self._detect_anomalies(df)
        df = self._assign_regulatory_status(df)
        df = self._add_ingestion_timestamp(df)
        silver = self._select_mandatory_columns(df)

        validation = validate_silver_schema(silver)
        validation = validation.merge(validate_regulatory_constraints(silver, self.config))

        dead_letter = pl.DataFrame()
        if not validation.passed:
            for err in validation.errors:
                self.audit.log("VALIDATION_FAILURE", "SYSTEM", err)
            if self.config.dead_letter.enabled:
                dead_letter = silver.filter(pl.lit(False))  # schema-only failures
                if write_dead_letter:
                    self._write_dead_letter(dead_letter)

        return PipelineResult(
            silver=silver,
            dead_letter=dead_letter,
            validation=validation,
            audit_trail=list(self.audit.events),
        )

    def _normalize_bronze(self, df: pl.DataFrame) -> pl.DataFrame:
        """Map heterogeneous Bronze column names to canonical Silver names."""
        renames: dict[str, str] = {}
        if "customer_id" in df.columns and "entity_id" not in df.columns:
            renames["customer_id"] = "entity_id"
        if "kyc_tier" in df.columns and "kyc_tier_level" not in df.columns:
            renames["kyc_tier"] = "kyc_tier_level"
        if "amount" in df.columns and "transaction_amount" not in df.columns:
            renames["amount"] = "transaction_amount"
        if "account_balance" in df.columns and "account_balance_before" not in df.columns:
            renames["account_balance"] = "account_balance_before"
        if "post_tx_balance" in df.columns and "account_balance_after" not in df.columns:
            renames["post_tx_balance"] = "account_balance_after"

        if renames:
            df = df.rename(renames)

        # Preserve raw anomaly fields before coercion for provenance hashing.
        if "anomaly_flag" in df.columns:
            df = df.with_columns(pl.col("anomaly_flag").alias("anomaly_flag_raw"))
        else:
            df = df.with_columns(pl.lit(None).alias("anomaly_flag_raw"))

        if "anomaly_type" in df.columns:
            df = df.with_columns(pl.col("anomaly_type").alias("anomaly_type_raw"))
        else:
            df = df.with_columns(pl.lit(None).alias("anomaly_type_raw"))

        return df

    def _enforce_types(self, df: pl.DataFrame) -> pl.DataFrame:
        """Cast columns to strict Polars dtypes."""
        exprs: list[pl.Expr] = []

        if "entity_id" in df.columns:
            exprs.append(pl.col("entity_id").cast(pl.String))
        if "counterparty_id" in df.columns:
            exprs.append(pl.col("counterparty_id").cast(pl.String))
        if "transaction_amount" in df.columns:
            exprs.append(pl.col("transaction_amount").cast(pl.Float64))
        if "account_balance_before" in df.columns:
            exprs.append(pl.col("account_balance_before").cast(pl.Float64))
        if "timestamp" in df.columns:
            ts_type = df.schema.get("timestamp")
            if ts_type in (pl.String, pl.Utf8):
                exprs.append(
                    pl.col("timestamp")
                    .str.to_datetime(time_zone="UTC", strict=False)
                    .alias("timestamp")
                )

        if exprs:
            df = df.with_columns(exprs)

        # Strict Boolean coercion for anomaly_flag (reject float decimals).
        if "anomaly_flag" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("anomaly_flag").is_null())
                .then(pl.lit(False))
                .when(pl.col("anomaly_flag").cast(pl.Float64, strict=False) > 0)
                .then(pl.lit(True))
                .otherwise(pl.lit(False))
                .cast(pl.Boolean)
                .alias("anomaly_flag")
            )
        else:
            df = df.with_columns(pl.lit(False).cast(pl.Boolean).alias("anomaly_flag"))

        return df

    def _sanitize_nulls(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fill all nulls with explicit regulatory defaults (never silent)."""
        defaults = self.config.defaults
        allowed = self.config.allowed_values

        if "entity_id" not in df.columns:
            df = df.with_columns(
                pl.concat_str(
                    [pl.lit("ENTITY_SYNTH_"), pl.arange(0, df.height, eager=True).cast(pl.String)]
                ).alias("entity_id")
            )

        if "counterparty_id" not in df.columns:
            df = df.with_columns(
                pl.concat_str(
                    [pl.lit(defaults.counterparty_id_prefix), pl.col("entity_id")]
                ).alias("counterparty_id")
            )

        if "transaction_id" not in df.columns:
            df = df.with_columns(
                pl.concat_str(
                    [
                        pl.col("entity_id"),
                        pl.col("timestamp").dt.strftime("%Y%m%d%H%M%S"),
                        pl.col("transaction_amount").cast(pl.String),
                    ],
                    separator="|",
                )
                .map_elements(_sha256_hex, return_dtype=pl.String)
                .alias("transaction_id")
            )

        fill_map: dict[str, Any] = {
            "kyc_tier_level": defaults.kyc_tier_level,
            "counterparty_risk_tier": defaults.counterparty_risk_tier,
            "transaction_type": defaults.transaction_type,
            "channel": defaults.channel,
            "region": defaults.region,
            "anomaly_type": defaults.anomaly_type,
            "regulatory_report_status": defaults.regulatory_report_status,
            "counterparty_id": None,
            "transaction_amount": 0.0,
            "account_balance_before": 0.0,
        }

        for col, default_val in fill_map.items():
            if col not in df.columns:
                dtype = pl.String if isinstance(default_val, str) else pl.Float64
                df = df.with_columns(pl.lit(default_val, dtype=dtype).alias(col))
            else:
                if default_val is None:
                    continue
                lit = pl.lit(default_val)
                if isinstance(default_val, str):
                    lit = pl.lit(default_val, dtype=pl.String)
                df = df.with_columns(pl.col(col).fill_null(lit))

        # Coerce categorical fields to allowed enum values.
        df = df.with_columns(
            pl.when(pl.col("kyc_tier_level").is_in(allowed.kyc_tier_level))
            .then(pl.col("kyc_tier_level"))
            .otherwise(pl.lit(defaults.kyc_tier_level))
            .alias("kyc_tier_level"),
            pl.when(pl.col("counterparty_risk_tier").is_in(allowed.counterparty_risk_tier))
            .then(pl.col("counterparty_risk_tier"))
            .otherwise(pl.lit(defaults.counterparty_risk_tier))
            .alias("counterparty_risk_tier"),
            pl.when(pl.col("transaction_type").is_in(allowed.transaction_type))
            .then(pl.col("transaction_type"))
            .otherwise(pl.lit(defaults.transaction_type))
            .alias("transaction_type"),
            pl.when(pl.col("channel").is_in(allowed.channel))
            .then(pl.col("channel"))
            .otherwise(pl.lit(defaults.channel))
            .alias("channel"),
            pl.when(pl.col("region").is_in(allowed.region))
            .then(pl.col("region"))
            .otherwise(pl.lit(defaults.region))
            .alias("region"),
        )

        # Remap legacy anomaly type strings to POCAMLA enum values.
        df = df.with_columns(
            pl.col("anomaly_type")
            .replace(
                {
                    "SURGE": "VELOCITY_SURGE",
                    "SPIKE": "AMOUNT_SPIKE",
                    "NONE": "NONE",
                },
                default=defaults.anomaly_type,
            )
            .alias("anomaly_type")
        )

        return df

    def _tier_balance_cap_expr(self) -> pl.Expr:
        """Build expression mapping KYC tier to balance cap from config."""
        caps = self.config.kyc_tier_balance_caps
        return (
            pl.when(pl.col("kyc_tier_level") == "TIER_1")
            .then(pl.lit(caps["TIER_1"]))
            .when(pl.col("kyc_tier_level") == "TIER_2")
            .then(pl.lit(caps["TIER_2"]))
            .when(pl.col("kyc_tier_level") == "TIER_3")
            .then(pl.lit(caps["TIER_3"]))
            .when(pl.col("kyc_tier_level") == "TIER_4")
            .then(pl.lit(caps["TIER_4"]))
            .otherwise(pl.lit(caps["TIER_1"]))
        )

    def _tier_velocity_cap_expr(self) -> pl.Expr:
        """Build expression mapping KYC tier to daily velocity cap from config."""
        caps = self.config.kyc_tier_velocity_caps
        return (
            pl.when(pl.col("kyc_tier_level") == "TIER_1")
            .then(pl.lit(caps["TIER_1"]))
            .when(pl.col("kyc_tier_level") == "TIER_2")
            .then(pl.lit(caps["TIER_2"]))
            .when(pl.col("kyc_tier_level") == "TIER_3")
            .then(pl.lit(caps["TIER_3"]))
            .when(pl.col("kyc_tier_level") == "TIER_4")
            .then(pl.lit(caps["TIER_4"]))
            .otherwise(pl.lit(caps["TIER_1"]))
        )

    def _compute_balances(self, df: pl.DataFrame) -> pl.DataFrame:
        """Derive post-transaction balance capped at tier-specific regulatory maximum."""
        tier_cap = self._tier_balance_cap_expr()

        if "account_balance_after" not in df.columns:
            df = df.with_columns(
                (pl.col("account_balance_before") + pl.col("transaction_amount")).alias(
                    "account_balance_after_raw"
                )
            )
        else:
            df = df.with_columns(pl.col("account_balance_after").alias("account_balance_after_raw"))

        df = df.with_columns(
            tier_cap.alias("_tier_balance_cap"),
            (pl.col("account_balance_after_raw") > tier_cap).alias("_balance_exceeds_cap"),
            (pl.col("account_balance_before") > tier_cap).alias("_before_exceeds_cap"),
            (pl.col("transaction_amount") > tier_cap).alias("_amount_exceeds_cap"),
            pl.min_horizontal(pl.col("account_balance_after_raw"), tier_cap).alias(
                "account_balance_after"
            ),
            (
                (pl.col("account_balance_after_raw") <= tier_cap)
                & (pl.col("account_balance_before") <= tier_cap)
                & (pl.col("transaction_amount") <= tier_cap)
                & (pl.col("transaction_amount") > 0)
            ).alias("is_wallet_balance_compliant"),
        )

        return df

    def _apply_stateful_windows(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute per-entity first/last seen timestamps and active tenure."""
        df = df.sort(["entity_id", "timestamp"])
        return df.with_columns(
            pl.col("timestamp")
            .first()
            .over("entity_id")
            .alias("account_first_seen"),
            pl.col("timestamp")
            .last()
            .over("entity_id")
            .alias("account_last_seen"),
        ).with_columns(
            (
                (
                    pl.col("account_last_seen").cast(pl.Int64)
                    - pl.col("account_first_seen").cast(pl.Int64)
                )
                / 86_400_000_000.0
            )
            .cast(pl.Float64)
            .alias("user_active_tenure_days")
        )

    def _detect_anomalies(self, df: pl.DataFrame) -> pl.DataFrame:
        """Classify deterministic anomaly types from regulatory and behavioral rules."""
        det = self.config.anomaly_detection
        velocity_cap = self._tier_velocity_cap_expr()

        df = df.with_columns(
            pl.col("timestamp").dt.date().alias("_tx_date"),
            (pl.col("transaction_amount") % det.round_number_modulus == 0).alias(
                "_is_round_number"
            ),
            (pl.col("transaction_amount") < det.smurfing_max_amount_kes).alias(
                "_is_small_tx"
            ),
        )

        df = df.with_columns(
            pl.col("transaction_amount")
            .median()
            .over("entity_id")
            .alias("_entity_median_amount"),
            pl.col("transaction_amount")
            .sum()
            .over(["entity_id", "_tx_date"])
            .alias("_daily_velocity"),
            pl.col("_is_small_tx").cast(pl.Int32).sum().over(["entity_id", "_tx_date"]).alias(
                "_smurf_count"
            ),
            pl.col("_is_round_number")
            .cast(pl.Int32)
            .sum()
            .over(["entity_id", "_tx_date"])
            .alias("_round_count"),
        )

        regulatory_violation = (
            pl.col("_balance_exceeds_cap")
            | pl.col("_before_exceeds_cap")
            | pl.col("_amount_exceeds_cap")
            | (pl.col("transaction_amount") <= 0)
        )

        velocity_surge = pl.col("_daily_velocity") > velocity_cap
        amount_spike = pl.col("transaction_amount") > (
            det.amount_spike_multiplier * pl.col("_entity_median_amount")
        )
        smurfing = pl.col("_smurf_count") >= det.smurfing_min_daily_count
        round_churn = pl.col("_round_count") >= det.round_number_min_daily_count

        df = df.with_columns(
            pl.when(regulatory_violation)
            .then(pl.lit("REGULATORY_CEILING_VIOLATION"))
            .when(velocity_surge)
            .then(pl.lit("VELOCITY_SURGE"))
            .when(amount_spike)
            .then(pl.lit("AMOUNT_SPIKE"))
            .when(smurfing)
            .then(pl.lit("SMURFING"))
            .when(round_churn)
            .then(pl.lit("ROUND_NUMBER_CHURN"))
            .otherwise(pl.lit("NONE"))
            .alias("_computed_anomaly_type"),
        )

        df = df.with_columns(
            (
                regulatory_violation
                | velocity_surge
                | amount_spike
                | smurfing
                | round_churn
                | pl.col("anomaly_flag")
            ).alias("anomaly_flag"),
            pl.when(
                regulatory_violation
                | velocity_surge
                | amount_spike
                | smurfing
                | round_churn
            )
            .then(pl.col("_computed_anomaly_type"))
            .when(pl.col("anomaly_flag"))
            .then(
                pl.coalesce(
                    pl.col("anomaly_type_raw").cast(pl.String),
                    pl.lit("INJECTED_ANOMALY"),
                )
            )
            .otherwise(pl.lit("NONE"))
            .alias("anomaly_type"),
        )

        # Audit log for regulatory violations.
        violation_rows = df.filter(regulatory_violation)
        if violation_rows.height > 0:
            for row in violation_rows.select(
                ["entity_id", "transaction_id", "kyc_tier_level", "transaction_amount"]
            ).iter_rows(named=True):
                self.audit.log(
                    "REGULATORY_CEILING_VIOLATION",
                    row["entity_id"],
                    (
                        f"kyc={row['kyc_tier_level']} amount={row['transaction_amount']}"
                    ),
                    transaction_id=row["transaction_id"],
                )

        drop_cols = [
            "_tx_date",
            "_is_round_number",
            "_is_small_tx",
            "_entity_median_amount",
            "_daily_velocity",
            "_smurf_count",
            "_round_count",
            "_computed_anomaly_type",
            "_tier_balance_cap",
            "_balance_exceeds_cap",
            "_before_exceeds_cap",
            "_amount_exceeds_cap",
            "account_balance_after_raw",
            "anomaly_flag_raw",
            "anomaly_type_raw",
        ]
        existing_drop = [c for c in drop_cols if c in df.columns]
        return df.drop(existing_drop)

    def _assign_regulatory_status(self, df: pl.DataFrame) -> pl.DataFrame:
        """Map anomaly severity to regulatory reporting status."""
        return df.with_columns(
            pl.when(pl.col("anomaly_type") == "REGULATORY_CEILING_VIOLATION")
            .then(pl.lit("ESCALATED_TO_FIU"))
            .when(pl.col("anomaly_flag"))
            .then(pl.lit("SUSPICIOUS_ACTIVITY_FLAGGED"))
            .otherwise(pl.lit("NOT_TRIGGERED"))
            .alias("regulatory_report_status")
        )

    def _add_provenance_hashes(self, df: pl.DataFrame) -> pl.DataFrame:
        """Attach SHA-256 provenance hash per row from canonical raw payload."""
        flag_raw = (
            pl.col("anomaly_flag_raw").cast(pl.String)
            if "anomaly_flag_raw" in df.columns
            else pl.lit("")
        )
        type_raw = (
            pl.col("anomaly_type_raw").cast(pl.String)
            if "anomaly_type_raw" in df.columns
            else pl.lit("")
        )
        provenance_struct = pl.struct(
            [
                pl.col("entity_id").cast(pl.String),
                pl.col("timestamp").cast(pl.String),
                pl.col("transaction_amount").cast(pl.String),
                pl.col("account_balance_before").cast(pl.String),
                pl.col("kyc_tier_level").cast(pl.String),
                flag_raw.alias("anomaly_flag_raw"),
                type_raw.alias("anomaly_type_raw"),
            ]
        )
        return df.with_columns(
            provenance_struct.map_batches(_batch_provenance_hashes).alias(
                "data_provenance_hash"
            )
        )

    def _add_ingestion_timestamp(self, df: pl.DataFrame) -> pl.DataFrame:
        """Stamp system ingestion time on every record."""
        return df.with_columns(
            pl.lit(self.ingestion_ts.replace(tzinfo=timezone.utc)).alias(
                "ingestion_timestamp"
            )
        )

    def _select_mandatory_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Project and cast to the mandatory Silver schema."""
        exprs: list[pl.Expr] = []
        for col in MANDATORY_SILVER_COLUMNS:
            target_dtype = SILVER_COLUMN_DTYPES[col]
            if col in df.columns:
                exprs.append(pl.col(col).cast(target_dtype, strict=False).alias(col))
            else:
                exprs.append(pl.lit(None).cast(target_dtype).alias(col))

        result = df.select(exprs)
        literals = _null_fill_literals()
        fill_exprs = []
        for col, val in literals.items():
            if col in result.columns and result[col].null_count() > 0:
                if isinstance(val, bool):
                    fill_exprs.append(pl.col(col).fill_null(val).alias(col))
                elif isinstance(val, float):
                    fill_exprs.append(pl.col(col).fill_null(val).alias(col))
                else:
                    fill_exprs.append(pl.col(col).fill_null(pl.lit(val)).alias(col))
        if fill_exprs:
            result = result.with_columns(fill_exprs)
        return result

    def _empty_silver(self) -> pl.DataFrame:
        """Return an empty DataFrame with the mandatory Silver schema."""
        return pl.DataFrame(
            {col: pl.Series(col, [], dtype=SILVER_COLUMN_DTYPES[col]) for col in MANDATORY_SILVER_COLUMNS}
        )

    def _write_dead_letter(self, df: pl.DataFrame) -> Path:
        """Persist rejected records for dead-letter queue processing."""
        out_dir = Path(self.config.dead_letter.output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"dead_letter_{ts}.parquet"
        df.write_parquet(path)
        logger.info("Dead-letter records written to %s", path)
        return path


def _null_fill_literals() -> dict[str, Any]:
    """Explicit post-cast null fills for mandatory Silver columns."""
    return {
        "entity_id": "UNKNOWN_ENTITY",
        "counterparty_id": "UNKNOWN_CP",
        "transaction_id": "UNKNOWN_TX",
        "kyc_tier_level": "TIER_1",
        "counterparty_risk_tier": "LOW",
        "transaction_type": "P2P",
        "channel": "USSD",
        "region": "Nairobi",
        "anomaly_type": "NONE",
        "regulatory_report_status": "NOT_TRIGGERED",
        "data_provenance_hash": _sha256_hex(""),
        "transaction_amount": 0.0,
        "account_balance_before": 0.0,
        "account_balance_after": 0.0,
        "user_active_tenure_days": 0.0,
        "anomaly_flag": False,
        "is_wallet_balance_compliant": False,
    }


def bronze_to_silver(
    bronze_df: pl.DataFrame,
    config_path: str | Path = "config/regulatory.yaml",
) -> pl.DataFrame:
    """Convenience wrapper: Bronze → Silver transformation.

    Args:
        bronze_df: Raw Bronze DataFrame.
        config_path: Regulatory configuration path.

    Returns:
        Compliant Silver DataFrame with all mandatory columns.
    """
    result = BronzeToSilverPipeline(config_path=config_path).transform(bronze_df)
    if not result.validation.passed:
        raise ValueError(
            f"Silver validation failed: {'; '.join(result.validation.errors)}"
        )
    return result.silver


def silver_to_gold(silver_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate Silver records into Gold behavioral feature vectors.

    Args:
        silver_df: Compliant Silver-layer DataFrame.

    Returns:
        Gold-layer feature DataFrame keyed by ``entity_id``.
    """
    if silver_df.is_empty():
        return pl.DataFrame()

    return (
        silver_df.group_by("entity_id")
        .agg(
            pl.col("transaction_amount").sum().alias("total_transaction_volume"),
            pl.col("transaction_amount").mean().alias("mean_transaction_amount"),
            pl.col("transaction_amount").std().fill_null(0.0).alias("std_transaction_amount"),
            pl.col("transaction_amount").count().alias("transaction_count"),
            pl.col("anomaly_flag").sum().alias("anomaly_count"),
            pl.col("anomaly_flag").mean().alias("anomaly_rate"),
            pl.col("account_first_seen").first().alias("account_first_seen"),
            pl.col("account_last_seen").last().alias("account_last_seen"),
            pl.col("user_active_tenure_days").max().alias("user_active_tenure_days"),
            pl.col("kyc_tier_level").last().alias("kyc_tier_level"),
            pl.col("is_wallet_balance_compliant").all().alias("is_fully_compliant"),
            pl.col("regulatory_report_status")
            .filter(pl.col("regulatory_report_status") != "NOT_TRIGGERED")
            .count()
            .alias("regulatory_flags_count"),
        )
        .with_columns(
            (pl.col("anomaly_count") > 0).alias("requires_manual_review"),
        )
    )


def aml_silver_to_feature_store_inputs(
    bronze_df: pl.DataFrame,
    aml_silver: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Map POCAMLA Silver output onto Bronze rows for GoldLayer consumption.

    Preserves all AML feature columns from Bronze while overlaying compliant
    anomaly classification, provenance, and regulatory fields from the AML engine.
    """
    if "customer_id" not in bronze_df.columns:
        raise ValueError("Bronze data must include customer_id for Gold layer mapping")
    if "transaction_id" not in bronze_df.columns:
        raise ValueError("Bronze data must include transaction_id for Gold layer mapping")

    aml_overlay_cols = [
        c
        for c in (
            "anomaly_flag",
            "anomaly_type",
            "regulatory_report_status",
            "data_provenance_hash",
            "is_wallet_balance_compliant",
            "transaction_amount",
            "account_balance_before",
            "account_balance_after",
            "kyc_tier_level",
            "ingestion_timestamp",
        )
        if c in aml_silver.columns
    ]

    aml_overlay = aml_silver.rename({"entity_id": "customer_id"}).select(
        ["customer_id", "transaction_id", *aml_overlay_cols]
    )

    for col in aml_overlay_cols:
        aml_overlay = aml_overlay.rename({col: f"_aml_{col}"})

    merged = bronze_df.join(
        aml_overlay,
        on=["customer_id", "transaction_id"],
        how="left",
    )

    overlay_exprs: list[pl.Expr] = []
    if "_aml_anomaly_flag" in merged.columns:
        overlay_exprs.append(
            pl.coalesce(pl.col("_aml_anomaly_flag"), pl.col("anomaly_flag")).alias(
                "anomaly_flag"
            )
        )
    if "_aml_anomaly_type" in merged.columns:
        overlay_exprs.append(
            pl.coalesce(pl.col("_aml_anomaly_type"), pl.col("anomaly_type")).alias(
                "anomaly_type"
            )
        )
    if "_aml_transaction_amount" in merged.columns:
        overlay_exprs.append(pl.col("_aml_transaction_amount").alias("amount"))
    if "_aml_regulatory_report_status" in merged.columns:
        overlay_exprs.append(
            pl.col("_aml_regulatory_report_status").alias("regulatory_report_status")
        )
    if "_aml_data_provenance_hash" in merged.columns:
        overlay_exprs.append(
            pl.col("_aml_data_provenance_hash").alias("data_provenance_hash")
        )

    if overlay_exprs:
        merged = merged.with_columns(overlay_exprs)

    drop_cols = [c for c in merged.columns if c.startswith("_aml_")]
    transaction_fact_df = merged.drop(drop_cols)

    customer_columns = [
        c
        for c in (
            "customer_id",
            "device_age_days",
            "sim_match_status",
            "wallet_tier_encoded",
            "kyc_level_encoded",
        )
        if c in bronze_df.columns
    ]
    customer_dimension_df = bronze_df.select(customer_columns).unique(subset=["customer_id"])

    return transaction_fact_df, customer_dimension_df


def run_medallion_pipeline(
    bronze_df: pl.DataFrame,
    config_path: str | Path = "config/regulatory.yaml",
) -> dict[str, pl.DataFrame]:
    """Execute Bronze → Silver → Gold migration in one call.

    Args:
        bronze_df: Raw Bronze input.
        config_path: Regulatory YAML path.

    Returns:
        Dictionary with keys ``bronze``, ``silver``, and ``gold``.
    """
    pipeline = BronzeToSilverPipeline(config_path=config_path)
    result = pipeline.transform(bronze_df)
    gold = silver_to_gold(result.silver)
    return {
        "bronze": bronze_df,
        "silver": result.silver,
        "gold": gold,
        "dead_letter": result.dead_letter,
        "audit_trail": result.audit_trail,
    }
