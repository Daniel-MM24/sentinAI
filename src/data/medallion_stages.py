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
    derive_temporal_features,
)
from src.data.synthetic_generator import AMLGenerator, AMLGeneratorConfig
from src.datasets.gold import GoldLayer
from src.data.anomaly_injector import FinancialAnomalyInjector, TVAEInjectorConfig
from src.data.hybrid_reconstructor import BalanceReconstructor

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


@dataclass(frozen=True)
class TVAEHybridStageResult:
    gold_path: str
    partition_key: str
    record_count: int
    generation_method: str


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
    force_clean_baseline: bool = False,
) -> BronzeStageResult:
    """Generate AML synthetic data, inject anomalies, and ingest to Bronze.
    
    Args:
        force_clean_baseline: If True, override anomaly_ratio to 0 for clean TVAE baseline
    """
    bronze_layer = BronzeLayer(bronze_base_path=str(bronze_base_path))
    partition_key = partition_key or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Override anomaly_ratio for clean baseline generation
    if force_clean_baseline:
        anomaly_ratio = 0.0
        logger.info("Force clean baseline: anomaly_ratio set to 0.0")

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

    # Generate clean data without anomaly injection
    # Anomaly injection happens later in TVAE hybrid stage or after feature engineering
    logger.info("Generating clean baseline data (anomaly injection deferred)")
    customers_df, transactions_df = generator.generate_normalized(
        anomaly_ratio=None,  # Skip anomaly injection in bronze stage
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

    # Derive temporal features in Silver layer
    silver_with_temporal = derive_temporal_features(result.silver)
    logger.info("Added temporal features to Silver: hour, day_of_week, month, is_weekend, is_night")

    silver_dir = Path(silver_base_path)
    silver_dir.mkdir(parents=True, exist_ok=True)

    # Write customers directly (already in normalized form)
    customers_path = silver_dir / f"silver_customers_{partition_key}.parquet"
    bronze_customers_df.write_parquet(customers_path)

    # Write transactions after silver transformation with temporal features
    transactions_path = silver_dir / f"silver_transactions_{partition_key}.parquet"
    silver_with_temporal.write_parquet(transactions_path)
    
    # Write AML-compliant silver (legacy compatibility)
    aml_silver_path = silver_dir / f"silver_aml_compliant_{partition_key}.parquet"
    silver_with_temporal.write_parquet(aml_silver_path)

    logger.info(
        "Silver transformation complete: %s transactions + %s customers (AML silver at %s)",
        silver_with_temporal.height,
        bronze_customers_df.height,
        aml_silver_path,
    )

    return SilverStageResult(
        transactions_path=transactions_path,
        customers_path=customers_path,
        aml_silver_path=aml_silver_path,
        partition_key=partition_key,
        transaction_count=silver_with_temporal.height,
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
    
    gold_uri = os.path.join("data/gold/features", f"v{gold_version}")
    logger.info("Gold feature store created at: %s", gold_uri)

    return GoldStageResult(gold_uri=gold_uri, partition_key=partition_key)


@lineage_trace(
    job_name="tvae_hybrid_generation",
    input_datasets=["tvae_model", "monte_carlo_baseline"],
    output_datasets=["tvae_hybrid_gold"],
    namespace="sentinai.tvae_hybrid",
)
def run_tvae_hybrid_stage(
    *,
    partition_key: str | None = None,
    n_samples: int = 50000,
    anomaly_ratio: float = 0.015,
    baseline_dir: str | Path = "data/bronze",
    model_dir: str | Path = "models",
    data_dir: str | Path = "data",
    gold_dir: str | Path = "data/gold",
    num_customers: int = 10000,
    num_transactions: int = 100000,
    skip_baseline: bool = True,
    retrain_tvae: bool = False,
) -> TVAEHybridStageResult:
    """Run TVAE hybrid pipeline to generate synthetic AML data.
    
    This stage replaces the bronze stage when TVAE hybrid generation is enabled.
    It uses a deep generative model (TVAE) combined with deterministic post-processing
    to create realistic synthetic financial data.
    
    Args:
        partition_key: Partition key for data versioning
        n_samples: Number of TVAE samples to generate
        anomaly_ratio: Percentage of customers to flag as launderers
        baseline_dir: Directory for baseline data
        model_dir: Directory for TVAE models
        data_dir: Directory for intermediate data
        gold_dir: Directory for final gold output
        num_customers: Number of customers for Monte Carlo baseline
        num_transactions: Number of transactions for Monte Carlo baseline
        skip_baseline: Skip Monte Carlo generation if baseline exists
        retrain_tvae: Force TVAE retraining even if model exists
    
    Returns:
        TVAEHybridStageResult with paths and statistics
    """
    partition_key = partition_key or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    logger.info("=" * 60)
    logger.info("TVAE Hybrid Stage Starting")
    logger.info(f"Partition: {partition_key}")
    logger.info(f"Samples: {n_samples}")
    logger.info(f"Anomaly ratio: {anomaly_ratio}")
    logger.info("=" * 60)
    
    # Create directories
    Path(baseline_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    Path(gold_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if gold output already exists
    gold_path = Path(gold_dir) / f"tvae_hybrid_gold_{partition_key}.parquet"
    if gold_path.exists():
        logger.warning(f"TVAE hybrid gold already exists at {gold_path}")
        gold_df = pl.read_parquet(gold_path)
        return TVAEHybridStageResult(
            gold_path=str(gold_path),
            partition_key=partition_key,
            record_count=gold_df.height,
            generation_method="tvae_hybrid",
        )
    
    # Phase 1: Skip Monte Carlo baseline generation - use existing TVAE model directly
    baseline_path = Path(baseline_dir) / f"monte_carlo_baseline_{partition_key}.parquet"
    logger.info("Phase 1: Skipping Monte Carlo baseline generation - using existing TVAE model")
    logger.info(f"Baseline path: {baseline_path}, exists: {baseline_path.exists()}")
    
    # Phase 2: TVAE training - use existing model
    model_path = Path(model_dir) / f"tvae_model_{partition_key}.pkl"
    
    # Check for existing model with current partition key
    if not model_path.exists():
        # Try to find any existing TVAE model
        existing_models = list(Path(model_dir).glob("tvae_model_*.pkl"))
        if existing_models:
            model_path = existing_models[-1]  # Use most recent model
            logger.info(f"Phase 2: Using existing TVAE model at {model_path}")
        else:
            logger.error("Phase 2: No TVAE model found. Please train a TVAE model first.")
            raise FileNotFoundError("No TVAE model found. Please train a TVAE model first.")
    else:
        logger.info(f"Phase 2: Using existing TVAE model at {model_path}")
    
    # Phase 3: TVAE sampling
    logger.info("Phase 3: Sampling from TVAE model")
    
    # Load trained model
    import pickle
    import pandas as pd
    import numpy as np
    
    with open(model_path, "rb") as f:
        generator = pickle.load(f)
    logger.info(f"Loaded TVAE model from {model_path}")
    
    # Sample synthetic events
    logger.info(f"Sampling {n_samples} synthetic events...")
    synthetic_df = generator.sample(n_samples)
    logger.info(f"Sampled {len(synthetic_df)} events")
    
    # Post-process samples (inverse log-transform amount)
    # Define log-transform params for consistency with training phase
    LOG_TRANSFORM_PARAMS = {
        "amount_mean": 6.02,
        "amount_std": 1.25,
        "clip_min": 1.0,
        "clip_max": 250000.0,
    }
    
    # Check if amount is log-transformed (if values are small, assume log-transformed)
    if synthetic_df["amount"].mean() < 10:  # Log-transformed amounts are typically small
        synthetic_df["amount"] = np.expm1(synthetic_df["amount"])
    
    synthetic_df["amount"] = synthetic_df["amount"].clip(
        lower=LOG_TRANSFORM_PARAMS["clip_min"],
        upper=LOG_TRANSFORM_PARAMS["clip_max"]
    )
    synthetic_df["amount"] = synthetic_df["amount"].round(2)
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(synthetic_df["timestamp"]):
        synthetic_df["timestamp"] = pd.to_datetime(synthetic_df["timestamp"])
    
    # Save raw events
    raw_events_path = Path(data_dir) / f"tvae_raw_events_{partition_key}.parquet"
    synthetic_df.to_parquet(raw_events_path, index=False)
    logger.info(f"Raw events saved to: {raw_events_path}")
    
    # Convert to polars for next phases
    raw_events_df = pl.from_pandas(synthetic_df)
    
    # Map TVAE output columns to expected 8-core schema for balance reconstruction
    # Expected: customer_id, tier, archetype, transaction_type, amount, timestamp, direction, is_international
    # Check for tier column alternatives and handle duplicates
    tier_columns = [col for col in raw_events_df.columns if col in ["customer_tier", "wallet_tier_encoded", "tier"]]
    
    if "tier" not in raw_events_df.columns and tier_columns:
        # Use the first available tier column
        tier_col = tier_columns[0]
        raw_events_df = raw_events_df.rename({tier_col: "tier"})
        logger.info(f"Mapped column {tier_col} to tier")
    elif "tier" in raw_events_df.columns and len(tier_columns) > 1:
        # Drop duplicate tier columns, keep the one named 'tier'
        duplicate_tier_cols = [col for col in tier_columns if col != "tier"]
        raw_events_df = raw_events_df.drop(duplicate_tier_cols)
        logger.info(f"Dropped duplicate tier columns: {duplicate_tier_cols}")
    
    # Add missing columns with default values
    if "archetype" not in raw_events_df.columns:
        raw_events_df = raw_events_df.with_columns(
            pl.lit("P2P").alias("archetype")
        )
        logger.info("Added missing archetype column with default value 'P2P'")
    
    if "direction" not in raw_events_df.columns:
        raw_events_df = raw_events_df.with_columns(
            pl.when(pl.col("amount") > 0)
            .then(pl.lit("outflow"))
            .otherwise(pl.lit("inflow"))
            .alias("direction")
        )
        logger.info("Added missing direction column derived from amount")
    
    if "is_international" not in raw_events_df.columns:
        raw_events_df = raw_events_df.with_columns(
            pl.lit(False).alias("is_international")
        )
        logger.info("Added missing is_international column with default value False")
    
    # Select only the 8 core columns needed for balance reconstruction
    CORE_TVAE_COLUMNS = [
        "customer_id", "tier", "archetype", "transaction_type", 
        "amount", "timestamp", "direction", "is_international"
    ]
    
    # Ensure all core columns exist
    missing_cols = [col for col in CORE_TVAE_COLUMNS if col not in raw_events_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns after mapping: {missing_cols}")
    
    raw_events_df = raw_events_df.select(CORE_TVAE_COLUMNS)
    logger.info(f"Selected 8 core TVAE columns: {raw_events_df.columns.tolist()}")
    
    # Phase 4: Balance reconstruction
    logger.info("Phase 4: Balance reconstruction")
    reconstructor = BalanceReconstructor(output_dir=str(data_dir))
    balance_df = reconstructor.reconstruct(raw_events_df.to_pandas(), partition=partition_key)
    logger.info(f"Balance reconstruction complete: {len(balance_df)} records")
    
    # Phase 5: Feature engineering
    logger.info("Phase 5: Feature engineering")
    engineer = CustomerFeatureEngineer()
    enriched_df = engineer.compute_features(
        transactions_df=pl.from_pandas(balance_df),
        feature_date=datetime.now(timezone.utc)
    )
    logger.info(f"Feature engineering complete: {len(enriched_df)} records")
    
    # Phase 6: Anomaly injection
    logger.info("Phase 6: Anomaly injection")
    config = TVAEInjectorConfig(launderer_fraction=anomaly_ratio)
    injector = FinancialAnomalyInjector(config)
    gold_df = injector.inject(
        enriched_df,
        partition=partition_key,
        output_dir=str(gold_dir)
    )
    logger.info(f"Anomaly injection complete: {len(gold_df)} records")
    
    # Write final gold output (injector already saved it, but we ensure it's at the expected path)
    # The injector already saves to the gold_path location, so we just need to verify
    if not gold_path.exists():
        logger.warning(f"Gold file not found at expected path {gold_path}, injector may have saved elsewhere")
    
    logger.info(f"TVAE hybrid gold at {gold_path}")
    
    # Load the gold data to get record count
    if gold_path.exists():
        gold_pl_df = pl.read_parquet(gold_path)
    else:
        # Fallback: try to find the file the injector created
        gold_files = list(Path(gold_dir).glob(f"tvae_hybrid_gold_{partition_key}.parquet"))
        if gold_files:
            gold_path = gold_files[0]
            gold_pl_df = pl.read_parquet(gold_path)
            logger.info(f"Using gold file from injector location: {gold_path}")
        else:
            logger.error("No gold file found")
            gold_pl_df = pl.DataFrame()
    
    logger.info("=" * 60)
    logger.info("TVAE Hybrid Stage Completed")
    logger.info(f"Total records: {gold_pl_df.height}")
    logger.info("=" * 60)
    
    return TVAEHybridStageResult(
        gold_path=str(gold_path),
        partition_key=partition_key,
        record_count=gold_pl_df.height,
        generation_method="tvae_hybrid",
    )


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
    use_tvae_hybrid: bool = False,
    n_samples: int = 50000,
    anomaly_ratio: float = 0.015,
) -> dict[str, Any]:
    """Execute Bronze → Silver → Gold sequentially with OpenLineage tracking.
    
    Args:
        fast_mode: Enable compact synthetic dataset for local runs
        force_refresh: Force regeneration of all data
        data_dir: Root data directory
        use_tvae_hybrid: Use TVAE hybrid generation instead of pure Monte Carlo
        n_samples: Number of TVAE samples to generate (when use_tvae_hybrid=True)
        anomaly_ratio: Anomaly ratio for generation
    
    Returns:
        Dictionary containing stage results and metadata
    """
    settings = resolve_runtime_settings(fast_mode=fast_mode, force_refresh=force_refresh)
    root = data_dir or (PROJECT_ROOT / "data")

    if settings["clean_data_directories"]:
        clean_data_directories(root)

    # Log which generation method is being used
    generation_method = "tvae_hybrid" if use_tvae_hybrid else "monte_carlo"
    logger.info("=" * 60)
    logger.info(f"Medallion Orchestrator Starting")
    logger.info(f"Generation method: {generation_method}")
    logger.info(f"Fast mode: {fast_mode}")
    logger.info(f"Force refresh: {force_refresh}")
    logger.info("=" * 60)

    if use_tvae_hybrid:
        # TVAE Hybrid Path: Skip bronze, run TVAE hybrid stage
        logger.info("Using TVAE hybrid generation path")
        
        tvae_cfg = settings["bronze"]
        tvae_result = run_tvae_hybrid_stage(
            partition_key=None,
            n_samples=n_samples,
            anomaly_ratio=anomaly_ratio,
            baseline_dir=root / "bronze",
            model_dir=PROJECT_ROOT / "models",
            data_dir=root,
            gold_dir=root / "gold",
            num_customers=tvae_cfg["num_customers"],
            num_transactions=tvae_cfg.get("target_transactions"),
            skip_baseline=not force_refresh,
            retrain_tvae=force_refresh,
        )
        
        # For TVAE hybrid, we still need to run silver and gold stages
        # but they will use the TVAE hybrid output as input
        # For now, we'll skip silver/gold since TVAE hybrid produces gold directly
        # In a full implementation, we might want to apply additional silver transformations
        
        logger.info("TVAE hybrid produces gold directly, skipping silver stage")
        
        emit_transformation_metadata(
            job_name="run_medallion_pipeline_tvae_hybrid",
            run_id=str(uuid.uuid4()),
            transformation_python="TVAE Hybrid Generation → Gold features",
            input_rows=tvae_result.record_count,
            output_rows=tvae_result.record_count,
        )
        
        return {
            "generation_method": generation_method,
            "tvae_hybrid": tvae_result,
            "fast_mode": fast_mode,
        }
    else:
        # Standard Monte Carlo Path: Bronze → Silver → Gold
        logger.info("Using standard Monte Carlo generation path")
        
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
            "generation_method": generation_method,
            "bronze": bronze_result,
            "silver": silver_result,
            "gold": gold_result,
            "fast_mode": fast_mode,
        }
