"""
Pipeline Retention Verification Test

This test verifies that the Medallion pipeline maintains acceptable retention rates
across Bronze -> Silver -> Gold transformations, ensuring data completeness and
compliance with MRM standards.
"""

import pytest
import polars as pl
from pathlib import Path
import json
import sys
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.bronze import BronzeLayer
from src.datasets.silver import SilverLayer
from src.datasets.gold import GoldLayer
from src.data.synthetic_engine import SyntheticMpesaGenerator


def test_pipeline_retention_rate():
    """
    Test that the pipeline maintains >95% retention rate from Bronze to Gold.
    
    This test:
    1. Generates 10,000 synthetic records
    2. Runs them through Bronze -> Silver -> Gold pipeline
    3. Verifies retention rates at each stage
    4. Ensures overall retention >95%
    """
    # Setup
    bronze_layer = BronzeLayer(bronze_base_path="data/bronze")
    silver_layer = SilverLayer()
    gold_layer = GoldLayer(version="test_v1.0")
    
    partition_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    # Generate synthetic data
    generator = SyntheticMpesaGenerator(
        target_distribution_params={
            "transaction_type_probs": {"P2P": 0.6, "C2B": 0.3, "B2C": 0.1},
            "amount_mean": 5.0,
            "amount_std": 1.0,
            "velocity_lambda": 10.0,
            "dataset_size": 100000,
            "total_queries_per_year": 12,
            "query_type": "standard",
            "clipping_bound": 5000.0,
            "seed": 42,
            "model_version": "v1.0"
        }
    )
    
    n_records = 10000
    synthetic_data = generator.generate_batch(n_records=n_records, num_users=2000)
    
    # Bronze ingestion
    bronze_path = bronze_layer.ingest_synthetic_data(
        synthetic_data,
        source_table="test_synthetic_transactions",
        partition_key=partition_key
    )
    
    # Read Bronze data
    bronze_df = bronze_layer.read_bronze_partition(partition_key)
    
    # Silver transformation
    silver_df = silver_layer.transform_to_silver(bronze_df, partition_key=partition_key)
    
    # Gold transformation
    gold_uri = gold_layer.create_feature_store(
        transactions_df=silver_df,
        customers_df=silver_df
    )
    
    # Read Gold data
    gold_file = Path(gold_uri) / "features.parquet"
    gold_df = pl.read_parquet(gold_file)
    
    # Calculate retention rates
    bronze_count = bronze_df.height
    silver_count = silver_df.height
    gold_count = gold_df.height
    
    bronze_to_silver_rate = (silver_count / bronze_count) * 100
    silver_to_gold_rate = (gold_count / silver_count) * 100
    overall_rate = (gold_count / bronze_count) * 100
    
    # Assertions
    assert bronze_count == n_records, f"Bronze should have {n_records} records, got {bronze_count}"
    assert bronze_to_silver_rate >= 95.0, f"Bronze->Silver retention {bronze_to_silver_rate:.2f}% is below 95%"
    assert silver_to_gold_rate >= 95.0, f"Silver->Gold retention {silver_to_gold_rate:.2f}% is below 95%"
    assert overall_rate >= 95.0, f"Overall retention {overall_rate:.2f}% is below 95%"
    
    print(f"\nPipeline Retention Test Results:")
    print(f"  Bronze records: {bronze_count}")
    print(f"  Silver records: {silver_count}")
    print(f"  Gold records: {gold_count}")
    print(f"  Bronze->Silver: {bronze_to_silver_rate:.2f}%")
    print(f"  Silver->Gold: {silver_to_gold_rate:.2f}%")
    print(f"  Overall: {overall_rate:.2f}%")
    print(f"  ✓ All retention rates meet 95% threshold")
    
    # Cleanup test data
    import shutil
    bronze_partition = Path("data/bronze/transactions") / partition_key
    silver_file = Path("data/silver") / f"silver_{partition_key}.parquet"
    gold_dir = Path(gold_uri)
    
    if bronze_partition.exists():
        shutil.rmtree(bronze_partition)
    if silver_file.exists():
        silver_file.unlink()
    if gold_dir.exists():
        shutil.rmtree(gold_dir)


if __name__ == "__main__":
    test_pipeline_retention_rate()
    print("\n✓ Pipeline retention test passed successfully")
