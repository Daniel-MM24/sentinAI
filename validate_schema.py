#!/usr/bin/env python3
"""
Validation script to verify medallion pipeline schema changes.

Validates:
- Bronze: 3 customer columns + 8 transaction columns
- Silver: 8 transaction columns + 5 derived temporal columns
- Gold: Final schema with 4 engineered features
"""

import polars as pl
from datetime import datetime, timezone
from src.data.medallion_stages import run_bronze_stage, run_silver_stage, run_gold_stage
from pathlib import Path

def validate_bronze_schema(bronze_customers: pl.DataFrame, bronze_transactions: pl.DataFrame):
    """Validate Bronze schema has 3 customer + 8 transaction columns."""
    print("\n=== Bronze Schema Validation ===")
    
    expected_customer_cols = 3
    expected_transaction_cols = 8
    
    actual_customer_cols = len(bronze_customers.columns)
    actual_transaction_cols = len(bronze_transactions.columns)
    
    print(f"Expected customer columns: {expected_customer_cols}")
    print(f"Actual customer columns: {actual_customer_cols}")
    print(f"Customer columns: {bronze_customers.columns}")
    
    print(f"\nExpected transaction columns: {expected_transaction_cols}")
    print(f"Actual transaction columns: {actual_transaction_cols}")
    print(f"Transaction columns: {bronze_transactions.columns}")
    
    # Check if we're close to expected (allowing for metadata columns)
    customer_valid = actual_customer_cols >= expected_customer_cols
    transaction_valid = actual_transaction_cols >= expected_transaction_cols
    
    if customer_valid and transaction_valid:
        print("✓ Bronze schema validation PASSED")
        return True
    else:
        print("✗ Bronze schema validation FAILED")
        return False

def validate_silver_schema(silver_transactions: pl.DataFrame):
    """Validate Silver schema has 8 core + 5 temporal columns."""
    print("\n=== Silver Schema Validation ===")
    
    expected_core_cols = 8
    expected_temporal_cols = 5
    temporal_cols = ["hour", "day_of_week", "month", "is_weekend", "is_night"]
    
    actual_cols = len(silver_transactions.columns)
    temporal_present = sum(1 for col in temporal_cols if col in silver_transactions.columns)
    
    print(f"Expected core columns: {expected_core_cols}")
    print(f"Expected temporal columns: {expected_temporal_cols}")
    print(f"Actual total columns: {actual_cols}")
    print(f"Temporal columns present: {temporal_present}/{expected_temporal_cols}")
    print(f"Silver columns: {silver_transactions.columns}")
    
    # Check temporal columns are present
    temporal_valid = temporal_present == expected_temporal_cols
    total_valid = actual_cols >= (expected_core_cols + expected_temporal_cols)
    
    if temporal_valid and total_valid:
        print("✓ Silver schema validation PASSED")
        return True
    else:
        print("✗ Silver schema validation FAILED")
        return False

def validate_gold_schema(gold_features: pl.DataFrame):
    """Validate Gold schema has 4 engineered features."""
    print("\n=== Gold Schema Validation ===")
    
    expected_features = 4
    feature_cols = ["tx_count_7d", "volume_7d", "night_tx_ratio", "rapid_tx_ratio"]
    
    actual_cols = len(gold_features.columns)
    features_present = sum(1 for col in feature_cols if col in gold_features.columns)
    
    print(f"Expected engineered features: {expected_features}")
    print(f"Features present: {features_present}/{expected_features}")
    print(f"Actual total columns: {actual_cols}")
    print(f"Gold columns: {gold_features.columns}")
    
    # Check engineered features are present
    features_valid = features_present == expected_features
    
    if features_valid:
        print("✓ Gold schema validation PASSED")
        return True
    else:
        print("✗ Gold schema validation FAILED")
        return False

def main():
    """Run validation on medallion pipeline stages."""
    print("=" * 60)
    print("Medallion Pipeline Schema Validation")
    print("=" * 60)
    
    # Use fast mode for quick validation
    partition_key = "2026-08-05-validation"
    
    try:
        # Run Bronze stage
        print("\n--- Running Bronze Stage ---")
        bronze_result = run_bronze_stage(
            num_customers=100,
            num_days=7,
            target_transactions=1000,
            seed=42,
            anomaly_ratio=0.01,
            bronze_base_path="data/bronze",
            partition_key=partition_key,
            skip_if_partition_exists=False,
        )
        
        # Read bronze data
        from src.data.bronze import BronzeLayer
        bronze_layer = BronzeLayer(bronze_base_path="data/bronze")
        bronze_customers, bronze_transactions = bronze_layer.read_normalized_bronze_partition(partition_key)
        
        bronze_valid = validate_bronze_schema(bronze_customers, bronze_transactions)
        
        # Run Silver stage
        print("\n--- Running Silver Stage ---")
        silver_result = run_silver_stage(
            partition_key=partition_key,
            bronze_base_path="data/bronze",
            silver_base_path="data/silver",
        )
        
        # Read silver data
        silver_transactions = pl.read_parquet(silver_result.transactions_path)
        silver_valid = validate_silver_schema(silver_transactions)
        
        # Run Gold stage
        print("\n--- Running Gold Stage ---")
        gold_result = run_gold_stage(
            partition_key=partition_key,
            silver_base_path="data/silver",
        )
        
        # Read gold data
        gold_features_path = Path("data/gold/features/v1.0") / f"customer_features_{partition_key}.parquet"
        if gold_features_path.exists():
            gold_features = pl.read_parquet(gold_features_path)
            gold_valid = validate_gold_schema(gold_features)
        else:
            print(f"Gold features file not found: {gold_features_path}")
            gold_valid = False
        
        # Summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Bronze: {'✓ PASSED' if bronze_valid else '✗ FAILED'}")
        print(f"Silver: {'✓ PASSED' if silver_valid else '✗ FAILED'}")
        print(f"Gold:   {'✓ PASSED' if gold_valid else '✗ FAILED'}")
        
        all_valid = bronze_valid and silver_valid and gold_valid
        print(f"\nOverall: {'✓ ALL VALIDATIONS PASSED' if all_valid else '✗ SOME VALIDATIONS FAILED'}")
        print("=" * 60)
        
        return all_valid
        
    except Exception as e:
        print(f"\n✗ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
