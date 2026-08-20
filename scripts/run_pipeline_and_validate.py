"""
Comprehensive pipeline execution and validation script.
This script runs the full medallion pipeline (Bronze → Silver → Gold) and validates the output.
"""

import sys
from pathlib import Path
import shutil
from datetime import datetime
import polars as pl

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.medallion_stages import (
    run_bronze_stage,
    run_silver_stage,
    run_gold_stage,
    resolve_runtime_settings,
    _clean_layer,
)
from src.data.bronze import BronzeLayer


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_success(message):
    """Print a success message."""
    print(f"✓ {message}")


def print_error(message):
    """Print an error message."""
    print(f"✗ {message}")


def print_info(message):
    """Print an info message."""
    print(f"  {message}")


def clean_data_directories():
    """Clean existing data directories."""
    print_section("Step 1: Cleaning existing data directories")
    data_dir = Path(__file__).parent.parent / "data"
    
    for layer in ["bronze", "silver", "gold"]:
        layer_dir = data_dir / layer
        if layer_dir.exists():
            shutil.rmtree(layer_dir)
            print_info(f"Cleaned {layer} layer")
        layer_dir.mkdir(parents=True, exist_ok=True)
    
    print_success("Data directories cleaned")


def run_bronze_layer():
    """Run Bronze layer."""
    print_section("Step 2: Running Bronze layer")
    
    settings = resolve_runtime_settings(fast_mode=True)
    bronze_cfg = settings["bronze"]
    anomaly_cfg = settings["anomaly"]
    
    data_dir = Path(__file__).parent.parent / "data"
    _clean_layer(data_dir, "bronze")
    
    result = run_bronze_stage(
        num_customers=bronze_cfg["num_customers"],
        num_days=bronze_cfg["num_days"],
        target_transactions=bronze_cfg.get("target_transactions"),
        seed=bronze_cfg["seed"],
        anomaly_ratio=anomaly_cfg["anomaly_ratio"],
    )
    
    print_info(f"Bronze layer completed: {result.record_count} records")
    print_info(f"Partition key: {result.partition_key}")
    print_info(f"Anomaly ratio: {result.anomaly_ratio:.4f}")
    print_success("Bronze layer completed")
    
    return result


def run_silver_layer(partition_key):
    """Run Silver layer."""
    print_section("Step 3: Running Silver layer")
    
    data_dir = Path(__file__).parent.parent / "data"
    
    result = run_silver_stage(
        partition_key=partition_key,
        bronze_base_path=data_dir / "bronze",
        silver_base_path=data_dir / "silver",
    )
    
    print_info(f"Silver layer completed: {result.transaction_count} transactions")
    print_info(f"Customer count: {result.customer_count}")
    print_success("Silver layer completed")
    
    return result


def run_gold_layer(partition_key):
    """Run Gold layer."""
    print_section("Step 4: Running Gold layer")
    
    data_dir = Path(__file__).parent.parent / "data"
    
    result = run_gold_stage(
        partition_key=partition_key,
        silver_base_path=data_dir / "silver",
    )
    
    print_info(f"Gold layer completed: {result.gold_uri}")
    print_success("Gold layer completed")
    
    return result


def validate_pipeline(bronze_result, silver_result, gold_result):
    """Validate pipeline outputs and generate report."""
    print_section("Step 5: Pipeline Validation Report")
    
    print_info(f"Report generated: {datetime.now().isoformat()}")
    print()
    
    # Bronze layer validation
    print("BRONZE LAYER:")
    bronze_layer = BronzeLayer(bronze_base_path="data/bronze")
    customers_df, transactions_df = bronze_layer.read_normalized_bronze_partition(bronze_result.partition_key)
    
    print_info(f"  Customers: {customers_df.height}")
    print_info(f"  Transactions: {transactions_df.height}")
    print_info(f"  Customer columns: {len(customers_df.columns)}")
    print_info(f"  Transaction columns: {len(transactions_df.columns)}")
    print_info(f"  Customer columns: {list(customers_df.columns[:5])}...")
    print_info(f"  Transaction columns: {list(transactions_df.columns[:5])}...")
    print()
    
    # Silver layer validation
    print("SILVER LAYER:")
    silver_path = Path("data/silver/silver_transactions_2026-08-05.parquet")
    if silver_path.exists():
        silver_df = pl.read_parquet(silver_path)
        print_info(f"  Transactions: {silver_df.height}")
        print_info(f"  Columns: {len(silver_df.columns)}")
        
        temporal_features = ["hour", "day_of_week", "month", "is_weekend", "is_night"]
        present_temporal = [f for f in temporal_features if f in silver_df.columns]
        print_info(f"  Temporal features present: {len(present_temporal)}/{len(temporal_features)}")
        print_info(f"  Present: {present_temporal}")
    else:
        print_info("  Silver data not found")
    print()
    
    # Gold layer validation
    print("GOLD LAYER:")
    gold_dir = Path("data/gold/features/v1.0")
    customer_features_path = gold_dir / f"customer_features_{gold_result.partition_key}.parquet"
    gold_features_path = gold_dir / "gold_features_consolidated.parquet"
    
    if customer_features_path.exists():
        customer_features_df = pl.read_parquet(customer_features_path)
        print_info(f"  Customer features: {customer_features_df.height} customers")
        print_info(f"  Customer feature columns: {len(customer_features_df.columns)}")
        
        engineered_features = ["tx_count_7d", "volume_7d", "night_tx_ratio", "rapid_tx_ratio"]
        present_engineered = [f for f in engineered_features if f in customer_features_df.columns]
        print_info(f"  Engineered features present: {len(present_engineered)}/{len(engineered_features)}")
        print_info(f"  Present: {present_engineered}")
    else:
        print_info("  Customer features not found")
    
    if gold_features_path.exists():
        gold_features_df = pl.read_parquet(gold_features_path)
        print_info(f"  Gold features: {gold_features_df.height} transactions")
        print_info(f"  Gold feature columns: {len(gold_features_df.columns)}")
    else:
        print_info("  Gold features not found")
    print()
    
    # Data quality checks
    print("DATA QUALITY CHECKS:")
    
    # Check for nulls in key columns
    key_columns = ["customer_id", "transaction_id", "amount", "timestamp"]
    null_counts = {}
    for col in key_columns:
        if col in transactions_df.columns:
            null_counts[col] = transactions_df[col].null_count()
    
    print_info("  Null counts in key columns:")
    for col, count in null_counts.items():
        status = "✓" if count == 0 else "✗"
        print_info(f"    {status} {col}: {count}")
    
    # Check anomaly rate
    if "anomaly_flag" in transactions_df.columns:
        anomaly_rate = (transactions_df["anomaly_flag"].sum() / transactions_df.height) * 100
        print_info(f"  Anomaly rate: {anomaly_rate:.2f}%")
    
    print()
    print_success("Validation complete")


def main():
    """Main execution function."""
    print_section("SentinAI Pipeline Execution & Validation")
    
    try:
        # Step 1: Clean data
        clean_data_directories()
        
        # Step 2: Run Bronze
        bronze_result = run_bronze_layer()
        
        # Step 3: Run Silver
        silver_result = run_silver_layer(bronze_result.partition_key)
        
        # Step 4: Run Gold
        gold_result = run_gold_layer(silver_result.partition_key)
        
        # Step 5: Validate
        validate_pipeline(bronze_result, silver_result, gold_result)
        
        print_section("Pipeline Execution Complete")
        print_success("All stages completed successfully")
        
    except Exception as e:
        print_error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
