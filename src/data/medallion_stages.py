"""
Medallion pipeline stage library — Bronze → Silver → Gold.

Each stage is a callable library function instrumented with OpenLineage via
``lineage_trace`` or layer-level decorators. CLI entry points live in ``scripts/``.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import os

import polars as pl

from src.data.bronze import BronzeLayer
from src.data.feature_engineering import CustomerFeatureEngineer
from src.data.lineage_decorator import emit_transformation_metadata, lineage_trace
from src.data.pipelines import (
    BronzeToSilverPipeline,
    aml_silver_to_feature_store_inputs,
)
from src.data.synthetic_generator import AMLGenerator, AMLGeneratorConfig
from src.datasets.gold import GoldLayer, silver_to_transaction_features

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_REGULATORY_CONFIG = PROJECT_ROOT / "config" / "regulatory.yaml"


@dataclass(frozen=True)
class BronzeStageResult:
    bronze_path: str
    partition_key: str
    record_count: int
    anomaly_ratio: float


@dataclass(frozen=True)
class SilverStageResult:
    transactions_path: Path
    customers_path: Path
    aml_silver_path: Path
    partition_key: str
    transaction_count: int
    customer_count: int


@dataclass(frozen=True)
class GoldStageResult:
    gold_uri: str
    partition_key: str


def resolve_runtime_settings(
    fast_mode: bool = False,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Resolve runtime settings for the medallion workflow."""
    settings: dict[str, Any] = {
        "clean_data_directories": force_refresh or not fast_mode,
        "fast_mode": fast_mode,
        "bronze": {
            "num_customers": 75 if fast_mode else 10_000,
            "num_days": 3 if fast_mode else 365,
            "target_transactions": 5_000 if fast_mode else 1_000_000,
            "seed": 42,
        },
        "anomaly": {
            "anomaly_ratio": 0.015,
            "seed": 42,
        },
    }
    if fast_mode:
        logger.info("Fast mode enabled: compact synthetic dataset for local runs")
    else:
        logger.info("Full mode enabled: standard synthetic dataset size")
    return settings


def clean_data_directories(data_dir: Path | None = None) -> None:
    """Remove and recreate bronze, silver, and gold data directories."""
    root = data_dir or (PROJECT_ROOT / "data")
    for layer in ("bronze", "silver", "gold"):
        _clean_layer(root, layer)

    synthetic_db = root / "synthetic.duckdb"
    if synthetic_db.exists():
        logger.info("Cleaning synthetic database: %s", synthetic_db)
        synthetic_db.unlink()


def _clean_layer(root: Path, layer: str) -> None:
    """Remove and recreate a single medallion layer directory."""
    layer_dir = root / layer
    if layer_dir.exists():
        logger.info("Cleaning %s directory: %s", layer, layer_dir)
        shutil.rmtree(layer_dir)
    layer_dir.mkdir(parents=True, exist_ok=True)




@lineage_trace(
    job_name="bronze_ingest_synthetic",
    input_datasets=["synthetic_generation_config"],
    output_datasets=["bronze_transactions"],
    namespace="sentinai.bronze",
)
def run_bronze_stage(
    *,
    num_customers: int = 5000,
    num_days: int = 30,
    target_transactions: int | None = None,
    seed: int = 42,
    anomaly_ratio: float = 0.015,
    bronze_base_path: str | Path = "data/bronze",
    partition_key: str | None = None,
    skip_if_partition_exists: bool = True,
) -> BronzeStageResult:
    """Generate AML synthetic data, inject anomalies, and ingest to Bronze."""
    bronze_layer = BronzeLayer(bronze_base_path=str(bronze_base_path))
    partition_key = partition_key or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    existing = bronze_layer.read_bronze_partition(partition_key)
    if skip_if_partition_exists and existing.height > 0:
        logger.warning(
            "Bronze partition %s already has %s records; skipping ingestion",
            partition_key,
            existing.height,
        )
        bronze_path = str(Path(bronze_base_path) / "transactions" / partition_key)
        ratio = (
            float(existing["anomaly_flag"].cast(pl.Float64).mean())
            if "anomaly_flag" in existing.columns and existing.height > 0
            else 0.0
        )
        return BronzeStageResult(
            bronze_path=bronze_path,
            partition_key=partition_key,
            record_count=existing.height,
            anomaly_ratio=ratio,
        )

    config = AMLGeneratorConfig(
        num_customers=num_customers,
        num_days=num_days,
        target_transactions=target_transactions,
        seed=seed,
        enable_pii_hashing=True,  # Enable SHA-256 hash tokenization
        enforce_regulatory_caps=True,  # Enforce tier transaction/balance limits
        export_customer_metadata=True,  # Export customers_metadata.csv
    )
    logger.info(
        "Generating AML synthetic data with MRM compliance: customers=%s days=%s target=%s seed=%s",
        num_customers,
        num_days,
        target_transactions,
        seed,
    )
    generator = AMLGenerator(config)

    # Use normalized generation to separate customers and transactions.
    # Anomaly injection happens inside generate_normalized() on the combined
    # DataFrame so the injector has access to all INJECTABLE_FEATURES (aggregate
    # columns are stripped after injection).
    customers_df, transactions_df = generator.generate_normalized(
        anomaly_ratio=anomaly_ratio,
        anomaly_seed=seed,
    )
    logger.info("Generated %s customers and %s transactions (normalized schema)",
                customers_df.height, transactions_df.height)

    # Export customer metadata for MRM audit trails
    if config.export_customer_metadata:
        customers_df.write_csv(str(Path(bronze_base_path).parent / "customers_metadata.csv"))
        logger.info(f"Exported {len(customers_df)} customer metadata records")

    # Compute actual anomaly ratio from the labeled transactions
    if transactions_df.height > 0 and "anomaly_flag" in transactions_df.columns:
        anomaly_ratio_actual = float(
            transactions_df["anomaly_flag"].cast(pl.Float64).mean()
        )
    else:
        anomaly_ratio_actual = 0.0

    # Ingest both customers and transactions to bronze layer
    bronze_path = bronze_layer.ingest_normalized_synthetic_data(
        customers_df=customers_df,
        transactions_df=transactions_df,
        source_table="synthetic_transactions",
        partition_key=partition_key,
    )
    logger.info("Bronze data written to: %s", bronze_path)

    return BronzeStageResult(
        bronze_path=bronze_path,
        partition_key=partition_key,
        record_count=transactions_df.height,
        anomaly_ratio=anomaly_ratio_actual,
    )


@lineage_trace(
    job_name="silver_aml_transform",
    input_datasets=["bronze_customers", "bronze_transactions"],
    output_datasets=["silver_transactions", "silver_customers", "silver_aml_compliant"],
    namespace="sentinai.silver",
)
def run_silver_stage(
    *,
    partition_key: str | None = None,
    bronze_base_path: str | Path = "data/bronze",
    silver_base_path: str | Path = "data/silver",
    config_path: str | Path | None = None,
) -> SilverStageResult:
    """Transform Bronze to Silver using the POCAMLA AML engine with normalized schema."""
    partition_key = partition_key or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    config_path = Path(config_path or DEFAULT_REGULATORY_CONFIG)

    bronze_layer = BronzeLayer(bronze_base_path=str(bronze_base_path))
    
    # Read normalized bronze data (separate customers and transactions)
    bronze_customers_df, bronze_transactions_df = bronze_layer.read_normalized_bronze_partition(partition_key)
    
    if bronze_customers_df.is_empty() and bronze_transactions_df.is_empty():
        raise ValueError(f"No bronze data found for partition {partition_key}")

    logger.info("Read %s customers and %s transactions from Bronze partition %s", 
                bronze_customers_df.height, bronze_transactions_df.height, partition_key)

    # For now, pass transactions through silver transformation (customers are already normalized)
    # In future, we may want to apply silver transformations to both
    pipeline = BronzeToSilverPipeline(config_path=config_path)
    
    # Apply silver transformation to transactions only
    result = pipeline.transform(bronze_transactions_df)
    if not result.validation.passed:
        logger.warning("Silver validation issues: %s", result.validation.errors)

    silver_dir = Path(silver_base_path)
    silver_dir.mkdir(parents=True, exist_ok=True)

    # Write customers directly (already in normalized form)
    customers_path = silver_dir / f"silver_customers_{partition_key}.parquet"
    bronze_customers_df.write_parquet(customers_path)

    # Write transactions after silver transformation
    transactions_path = silver_dir / f"silver_transactions_{partition_key}.parquet"
    result.silver.write_parquet(transactions_path)
    
    # Write AML-compliant silver (legacy compatibility)
    aml_silver_path = silver_dir / f"silver_aml_compliant_{partition_key}.parquet"
    result.silver.write_parquet(aml_silver_path)

    logger.info(
        "Silver transformation complete: %s transactions + %s customers (AML silver at %s)",
        result.silver.height,
        bronze_customers_df.height,
        aml_silver_path,
    )

    return SilverStageResult(
        transactions_path=transactions_path,
        customers_path=customers_path,
        aml_silver_path=aml_silver_path,
        partition_key=partition_key,
        transaction_count=result.silver.height,
        customer_count=bronze_customers_df.height,
    )


@lineage_trace(
    job_name="gold_materialize_feature_store",
    input_datasets=["silver_transactions", "silver_customers"],
    output_datasets=["gold_feature_store"],
    namespace="sentinai.gold",
)
def run_gold_stage(
    *,
    partition_key: str | None = None,
    silver_base_path: str | Path = "data/silver",
    gold_version: str = "1.0",
) -> GoldStageResult:
    """Materialize Gold feature store from Silver fact and dimension tables using normalized schema."""
    partition_key = partition_key or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    silver_dir = Path(silver_base_path)

    transactions_path = silver_dir / f"silver_transactions_{partition_key}.parquet"
    customers_path = silver_dir / f"silver_customers_{partition_key}.parquet"
    if not transactions_path.exists():
        raise FileNotFoundError(f"Silver transactions not found: {transactions_path}")
    if not customers_path.exists():
        raise FileNotFoundError(f"Silver customers not found: {customers_path}")

    transactions_df = pl.read_parquet(transactions_path)
    customers_df = pl.read_parquet(customers_path)
    logger.info(
        "Read %s transactions and %s customers from Silver",
        transactions_df.height,
        customers_df.height,
    )

    if "anomaly_flag" in transactions_df.columns:
        anomaly_rate = transactions_df["anomaly_flag"].cast(pl.Float64).mean()
        logger.info("FMS anomaly vectors in Silver stream: rate=%.4f", anomaly_rate)

    # Use new feature engineering pipeline to compute customer features from raw transactions
    feature_engineer = CustomerFeatureEngineer()
    customer_features_df = feature_engineer.compute_features(
        transactions_df=transactions_df,
        feature_date=datetime.now(timezone.utc)
    )
    
    # Create gold output directory
    gold_dir = Path("data/gold/features") / f"v{gold_version}"
    gold_dir.mkdir(parents=True, exist_ok=True)
    
    # Write customer features
    customer_features_path = gold_dir / f"customer_features_{partition_key}.parquet"
    customer_features_df.write_parquet(customer_features_path)
    logger.info(f"Customer features written to: {customer_features_path}")
    
    # Also write the legacy silver_to_transaction_features for backward compatibility
    features = silver_to_transaction_features(
        transactions_df,
        version=gold_version,
        output_dir="data/gold/features",
    )
    
    gold_uri = os.path.join("data/gold/features", f"v{gold_version}")
    logger.info("Gold feature store created at: %s", gold_uri)

    return GoldStageResult(gold_uri=gold_uri, partition_key=partition_key)


@lineage_trace(
    job_name="run_medallion_pipeline",
    input_datasets=["pipeline_config"],
    output_datasets=["bronze_transactions", "silver_transactions", "gold_feature_store"],
    namespace="sentinai.pipeline",
)
def run_medallion_orchestrator(
    *,
    fast_mode: bool = False,
    force_refresh: bool = False,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute Bronze → Silver → Gold sequentially with OpenLineage tracking."""
    settings = resolve_runtime_settings(fast_mode=fast_mode, force_refresh=force_refresh)
    root = data_dir or (PROJECT_ROOT / "data")

    if settings["clean_data_directories"]:
        clean_data_directories(root)

    bronze_cfg = settings["bronze"]
    anomaly_cfg = settings["anomaly"]

    bronze_result = run_bronze_stage(
        num_customers=bronze_cfg["num_customers"],
        num_days=bronze_cfg["num_days"],
        target_transactions=bronze_cfg.get("target_transactions"),
        seed=bronze_cfg["seed"],
        anomaly_ratio=anomaly_cfg["anomaly_ratio"],
        bronze_base_path=root / "bronze",
        skip_if_partition_exists=not force_refresh,
    )

    silver_result = run_silver_stage(
        partition_key=bronze_result.partition_key,
        bronze_base_path=root / "bronze",
        silver_base_path=root / "silver",
    )

    gold_result = run_gold_stage(
        partition_key=silver_result.partition_key,
        silver_base_path=root / "silver",
    )

    emit_transformation_metadata(
        job_name="run_medallion_pipeline",
        run_id=str(uuid.uuid4()),
        transformation_python="Bronze AML synth + anomaly inject → AML Silver → Gold features",
        input_rows=bronze_result.record_count,
        output_rows=silver_result.transaction_count,
    )

    return {
        "bronze": bronze_result,
        "silver": silver_result,
        "gold": gold_result,
        "fast_mode": fast_mode,
    }
