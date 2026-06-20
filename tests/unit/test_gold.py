import os
import json
import pytest
import polars as pl
from unittest.mock import MagicMock
from src.datasets.gold import GoldLayer
from src.datasets.registry import FeatureRegistry
from src.data.features import FeatureEngineer
from src.datasets.schemas import GoldFeatureSchema

@pytest.fixture
def transactions_data():
    df = pl.DataFrame({
        "customer_id": ["c1", "c2", "c3"],
        "amount": [100.0, 200.0, 300.0],
        "timestamp": ["2023-10-01T10:00:00", "2023-10-02T10:00:00", "2023-10-01T11:00:00"]
    })
    # Cast timestamp to datetime
    df = df.with_columns(pl.col("timestamp").str.to_datetime())
    
    # Apply Feature Engineering
    fe = FeatureEngineer()
    df = fe.create_temporal_features(df)
    df = fe.create_risk_indicators(df)
    return df

@pytest.fixture
def customers_data():
    return pl.DataFrame({
        "customer_id": ["c1", "c2", "c4"],
        "name": ["Alice", "Bob", "Charlie"],
        "segment": ["A", "B", "C"]
    })

def test_gold_layer_create_feature_store(transactions_data, customers_data):
    mock_ol_client = MagicMock()
    mock_registry = MagicMock(spec=FeatureRegistry)
    
    layer = GoldLayer(version="1.0", ol_client=mock_ol_client, registry=mock_registry)
    
    output_uri = layer.create_feature_store(transactions_data, customers_data)
    
    assert output_uri == "s3://sentinai/gold/features/v1.0/"
    
    local_dir = "/tmp/sentinai/gold/features/v1.0/"
    assert os.path.exists(os.path.join(local_dir, "manifest.json"))
    
    with open(os.path.join(local_dir, "manifest.json"), "r") as f:
        manifest = json.load(f)
        
    assert manifest["input_rows_fact"] == 3
    assert manifest["input_rows_dim"] == 3
    assert manifest["output_rows"] == 2  # c1 and c2 match
    assert manifest["orphan_rows"] == 1  # c3 is orphan
    
    # Check that anomaly report for orphans was created
    orphan_path = os.path.join(local_dir, "anomaly_orphans.parquet")
    assert os.path.exists(orphan_path)
    orphans_df = pl.read_parquet(orphan_path)
    assert orphans_df.height == 1
    assert orphans_df["customer_id"][0] == "c3"
    
    # Read back a partition and validate against GoldFeatureSchema
    # (Parquet writes to partition folders, e.g. partition_date=...)
    # We can use PyArrow dataset or just read the dir with Polars
    result_df = pl.read_parquet(local_dir)
    GoldFeatureSchema.validate(result_df)
    
    mock_registry.register_feature_set.assert_called_once()
    assert mock_ol_client.emit.call_count == 2
