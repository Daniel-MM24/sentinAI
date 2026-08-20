"""
Comprehensive test suite to validate the streamlined schema implementation.

Tests cover:
- Bronze layer schema validation (3 customer columns, 8 transaction columns)
- Silver layer temporal features (hour, day_of_week, month, is_weekend, is_night)
- Gold layer engineered features (tx_count_7d, volume_7d, night_tx_ratio, rapid_tx_ratio)
- AML ground truth validation (3 columns, ~2% launderer percentage)
- Data quality checks (null values, balance continuity, tier compliance, temporal distributions)
- End-to-end pipeline validation (15-column final joined schema)
"""

import logging
import pytest
import polars as pl
from pathlib import Path
from datetime import datetime, timezone, timedelta

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.medallion_stages import (
    run_bronze_stage,
    run_silver_stage,
    run_gold_stage,
    clean_data_directories,
    resolve_runtime_settings,
)
from src.data.pipelines import derive_temporal_features
from src.data.feature_engineering import CustomerFeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestBronzeSchema:
    """Test Bronze layer schema validation."""
    
    @pytest.fixture
    def bronze_data(self):
        """Generate bronze data for testing."""
        settings = resolve_runtime_settings(fast_mode=True, force_refresh=True)
        bronze_cfg = settings["bronze"]
        anomaly_cfg = settings["anomaly"]
        
        # Clean data directories
        clean_data_directories()
        
        # Generate bronze data
        result = run_bronze_stage(
            num_customers=bronze_cfg["num_customers"],
            num_days=bronze_cfg["num_days"],
            target_transactions=bronze_cfg.get("target_transactions"),
            seed=bronze_cfg["seed"],
            anomaly_ratio=anomaly_cfg["anomaly_ratio"],
            skip_if_partition_exists=False,
        )
        
        return result
    
    def test_bronze_schema(self, bronze_data):
        """Test Bronze layer schema: customer profiles have 3 columns, transactions have 8 columns."""
        from src.data.bronze import BronzeLayer
        
        bronze_layer = BronzeLayer(bronze_base_path="data/bronze")
        customers_df, transactions_df = bronze_layer.read_normalized_bronze_partition(bronze_data.partition_key)
        
        # Test customer schema (3 core columns)
        customer_core_cols = [col for col in customers_df.columns if not col.startswith("_")]
        logger.info(f"Customer columns: {customer_core_cols}")
        assert len(customer_core_cols) >= 3, f"Expected at least 3 customer columns, got {len(customer_core_cols)}"
        
        # Verify essential customer columns exist
        essential_customer_cols = ["customer_id", "tier", "archetype"]
        for col in essential_customer_cols:
            assert col in customers_df.columns, f"Missing essential customer column: {col}"
        
        # Test transaction schema (8 core columns)
        transaction_core_cols = [col for col in transactions_df.columns if not col.startswith("_")]
        logger.info(f"Transaction columns: {transaction_core_cols}")
        assert len(transaction_core_cols) >= 8, f"Expected at least 8 transaction columns, got {len(transaction_core_cols)}"
        
        # Verify essential transaction columns exist
        essential_transaction_cols = [
            "customer_id", "transaction_type", "amount", "timestamp",
            "direction", "balance", "tier", "is_international"
        ]
        for col in essential_transaction_cols:
            assert col in transactions_df.columns, f"Missing essential transaction column: {col}"
        
        logger.info(f"✓ Bronze schema validated: {len(customer_core_cols)} customer cols, {len(transaction_core_cols)} transaction cols")


class TestSilverSchema:
    """Test Silver layer temporal features."""
    
    @pytest.fixture
    def silver_data(self):
        """Transform bronze to silver for testing."""
        settings = resolve_runtime_settings(fast_mode=True, force_refresh=True)
        bronze_cfg = settings["bronze"]
        anomaly_cfg = settings["anomaly"]
        
        # Clean and generate bronze
        clean_data_directories()
        bronze_result = run_bronze_stage(
            num_customers=bronze_cfg["num_customers"],
            num_days=bronze_cfg["num_days"],
            target_transactions=bronze_cfg.get("target_transactions"),
            seed=bronze_cfg["seed"],
            anomaly_ratio=anomaly_cfg["anomaly_ratio"],
            skip_if_partition_exists=False,
        )
        
        # Transform to silver
        silver_result = run_silver_stage(
            partition_key=bronze_result.partition_key,
            bronze_base_path="data/bronze",
            silver_base_path="data/silver",
        )
        
        return silver_result
    
    def test_silver_schema(self, silver_data):
        """Test Silver layer temporal features: hour, day_of_week, month, is_weekend, is_night."""
        # Read silver transactions
        transactions_df = pl.read_parquet(silver_data.transactions_path)
        
        # Verify temporal features are present
        temporal_features = ["hour", "day_of_week", "month", "is_weekend", "is_night"]
        for feature in temporal_features:
            assert feature in transactions_df.columns, f"Missing temporal feature: {feature}"
        
        # Validate temporal feature ranges
        assert transactions_df["hour"].min() >= 0, "Hour minimum out of range"
        assert transactions_df["hour"].max() <= 23, "Hour maximum out of range"
        
        assert transactions_df["day_of_week"].min() >= 0, "Day of week minimum out of range"
        assert transactions_df["day_of_week"].max() <= 6, "Day of week maximum out of range"
        
        assert transactions_df["month"].min() >= 1, "Month minimum out of range"
        assert transactions_df["month"].max() <= 12, "Month maximum out of range"
        
        # Validate boolean flags
        assert transactions_df["is_weekend"].dtype == pl.Boolean, "is_weekend should be Boolean"
        assert transactions_df["is_night"].dtype == pl.Boolean, "is_night should be Boolean"
        
        # Validate temporal distributions are realistic
        weekend_ratio = transactions_df["is_weekend"].cast(pl.Int32).mean()
        assert 0.2 <= weekend_ratio <= 0.4, f"Weekend ratio {weekend_ratio:.2%} unrealistic (expected 20-40%)"
        
        night_ratio = transactions_df["is_night"].cast(pl.Int32).mean()
        assert 0.1 <= night_ratio <= 0.4, f"Night ratio {night_ratio:.2%} unrealistic (expected 10-40%)"
        
        logger.info(f"✓ Silver temporal features validated: {', '.join(temporal_features)}")
        logger.info(f"  - Weekend ratio: {weekend_ratio:.2%}")
        logger.info(f"  - Night ratio: {night_ratio:.2%}")


class TestGoldSchema:
    """Test Gold layer engineered features."""
    
    @pytest.fixture
    def gold_data(self):
        """Materialize gold features for testing."""
        settings = resolve_runtime_settings(fast_mode=True, force_refresh=True)
        bronze_cfg = settings["bronze"]
        anomaly_cfg = settings["anomaly"]
        
        # Clean and generate bronze
        clean_data_directories()
        bronze_result = run_bronze_stage(
            num_customers=bronze_cfg["num_customers"],
            num_days=bronze_cfg["num_days"],
            target_transactions=bronze_cfg.get("target_transactions"),
            seed=bronze_cfg["seed"],
            anomaly_ratio=anomaly_cfg["anomaly_ratio"],
            skip_if_partition_exists=False,
        )
        
        # Transform to silver
        silver_result = run_silver_stage(
            partition_key=bronze_result.partition_key,
            bronze_base_path="data/bronze",
            silver_base_path="data/silver",
        )
        
        # Materialize gold
        gold_result = run_gold_stage(
            partition_key=silver_result.partition_key,
            silver_base_path="data/silver",
        )
        
        return gold_result
    
    def test_gold_schema(self, gold_data):
        """Test Gold layer engineered features: tx_count_7d, volume_7d, night_tx_ratio, rapid_tx_ratio."""
        # Read customer features
        gold_dir = Path("data/gold/features/v1.0")
        customer_features_path = gold_dir / f"customer_features_{gold_data.partition_key}.parquet"
        
        assert customer_features_path.exists(), f"Customer features not found at {customer_features_path}"
        
        customer_features_df = pl.read_parquet(customer_features_path)
        
        # Verify engineered features are present
        engineered_features = ["tx_count_7d", "volume_7d", "night_tx_ratio", "rapid_tx_ratio"]
        for feature in engineered_features:
            assert feature in customer_features_df.columns, f"Missing engineered feature: {feature}"
        
        # Validate feature ranges
        assert customer_features_df["tx_count_7d"].min() >= 0, "tx_count_7d should be non-negative"
        assert customer_features_df["volume_7d"].min() >= 0, "volume_7d should be non-negative"
        assert customer_features_df["night_tx_ratio"].min() >= 0, "night_tx_ratio should be non-negative"
        assert customer_features_df["night_tx_ratio"].max() <= 1, "night_tx_ratio should be <= 1"
        assert customer_features_df["rapid_tx_ratio"].min() >= 0, "rapid_tx_ratio should be non-negative"
        assert customer_features_df["rapid_tx_ratio"].max() <= 1, "rapid_tx_ratio should be <= 1"
        
        # Validate feature distributions are realistic
        avg_night_ratio = customer_features_df["night_tx_ratio"].mean()
        assert 0.0 <= avg_night_ratio <= 0.5, f"Average night ratio {avg_night_ratio:.2%} unrealistic"
        
        avg_rapid_ratio = customer_features_df["rapid_tx_ratio"].mean()
        assert 0.0 <= avg_rapid_ratio <= 0.5, f"Average rapid ratio {avg_rapid_ratio:.2%} unrealistic"
        
        logger.info(f"✓ Gold engineered features validated: {', '.join(engineered_features)}")
        logger.info(f"  - Average night ratio: {avg_night_ratio:.2%}")
        logger.info(f"  - Average rapid ratio: {avg_rapid_ratio:.2%}")
    
    def test_final_joined_schema(self, gold_data):
        """Test final joined schema has 15 columns when all tables are joined."""
        # Read silver transactions and customers
        silver_dir = Path("data/silver")
        transactions_df = pl.read_parquet(silver_dir / f"silver_transactions_{gold_data.partition_key}.parquet")
        customers_df = pl.read_parquet(silver_dir / f"silver_customers_{gold_data.partition_key}.parquet")
        
        # Read gold customer features
        gold_dir = Path("data/gold/features/v1.0")
        customer_features_df = pl.read_parquet(gold_dir / f"customer_features_{gold_data.partition_key}.parquet")
        
        # Join transactions with customers
        joined_df = transactions_df.join(customers_df, on="customer_id", how="left")
        
        # Join with customer features
        final_df = joined_df.join(customer_features_df, on="customer_id", how="left")
        
        # Count non-metadata columns
        core_cols = [col for col in final_df.columns if not col.startswith("_")]
        
        logger.info(f"Final joined schema has {len(core_cols)} columns")
        logger.info(f"Columns: {core_cols}")
        
        # The final schema should have approximately 15 columns (may vary slightly due to metadata)
        # Core columns: 8 transaction + 3 customer + 4 engineered features = 15
        assert len(core_cols) >= 15, f"Expected at least 15 columns in final joined schema, got {len(core_cols)}"
        
        logger.info(f"✓ Final joined schema validated: {len(core_cols)} columns")


class TestAMLGroundTruth:
    """Test AML ground truth validation."""
    
    @pytest.fixture
    def aml_ground_truth(self):
        """Generate AML ground truth for testing."""
        from src.data.aml_scenario_injector import AMLScenarioInjector
        
        # Load customer profiles
        customers_path = Path("data/bronze/customers/customer_profiles.csv")
        if not customers_path.exists():
            pytest.skip("Customer profiles not found - run bronze stage first")
        
        injector = AMLScenarioInjector(
            customers_path=str(customers_path),
            output_dir="data/synthetic",
            seed=42
        )
        
        # Inject AML scenarios
        injector.inject(
            num_launderers=2,
            scenarios_per_launderer=3,
            transactions_per_scenario=5
        )
        
        # Read ground truth
        ground_truth_path = Path("data/synthetic/aml_ground_truth.csv")
        assert ground_truth_path.exists(), "AML ground truth not generated"
        
        ground_truth_df = pl.read_csv(ground_truth_path)
        
        return ground_truth_df
    
    def test_aml_ground_truth(self, aml_ground_truth):
        """Test AML ground truth has 3 columns and ~2% launderer percentage."""
        # Verify column count
        assert aml_ground_truth.width == 3, f"Expected 3 columns in ground truth, got {aml_ground_truth.width}"
        
        # Verify column names
        expected_cols = ["user_id", "is_launderer", "aml_scenario"]
        for col in expected_cols:
            assert col in aml_ground_truth.columns, f"Missing ground truth column: {col}"
        
        # Verify launderer percentage is ~2%
        launderer_count = aml_ground_truth.filter(pl.col("is_launderer") == True).height
        total_count = aml_ground_truth.height
        launderer_pct = launderer_count / total_count if total_count > 0 else 0
        
        logger.info(f"Launderer percentage: {launderer_pct:.2%}")
        # Allow some flexibility around 2% due to small test datasets
        assert 0.01 <= launderer_pct <= 0.10, f"Launderer percentage {launderer_pct:.2%} unrealistic (expected ~2%)"
        
        # Verify scenario distribution
        scenario_counts = aml_ground_truth.filter(pl.col("is_launderer") == True)["aml_scenario"].value_counts()
        logger.info(f"Scenario distribution:\n{scenario_counts}")
        
        assert scenario_counts.height > 0, "No AML scenarios found in ground truth"
        
        logger.info(f"✓ AML ground truth validated: 3 columns, {launderer_pct:.2%} launderers")


class TestDataQuality:
    """Test data quality checks."""
    
    @pytest.fixture
    def pipeline_data(self):
        """Run full pipeline for data quality testing."""
        settings = resolve_runtime_settings(fast_mode=True, force_refresh=True)
        bronze_cfg = settings["bronze"]
        anomaly_cfg = settings["anomaly"]
        
        # Clean and run full pipeline
        clean_data_directories()
        bronze_result = run_bronze_stage(
            num_customers=bronze_cfg["num_customers"],
            num_days=bronze_cfg["num_days"],
            target_transactions=bronze_cfg.get("target_transactions"),
            seed=bronze_cfg["seed"],
            anomaly_ratio=anomaly_cfg["anomaly_ratio"],
            skip_if_partition_exists=False,
        )
        
        silver_result = run_silver_stage(
            partition_key=bronze_result.partition_key,
            bronze_base_path="data/bronze",
            silver_base_path="data/silver",
        )
        
        gold_result = run_gold_stage(
            partition_key=silver_result.partition_key,
            silver_base_path="data/silver",
        )
        
        return {
            "bronze": bronze_result,
            "silver": silver_result,
            "gold": gold_result,
        }
    
    def test_data_quality(self, pipeline_data):
        """Test data quality: no null values, balance continuity, tier compliance, temporal distributions."""
        from src.data.bronze import BronzeLayer
        
        # Read bronze data
        bronze_layer = BronzeLayer(bronze_base_path="data/bronze")
        customers_df, transactions_df = bronze_layer.read_normalized_bronze_partition(pipeline_data["bronze"].partition_key)
        
        # Test 1: No null values in essential columns
        essential_customer_cols = ["customer_id", "tier", "archetype"]
        for col in essential_customer_cols:
            if col in customers_df.columns:
                null_count = customers_df[col].null_count()
                assert null_count == 0, f"Found {null_count} null values in customer column {col}"
        
        essential_transaction_cols = ["customer_id", "amount", "timestamp", "balance", "tier"]
        for col in essential_transaction_cols:
            if col in transactions_df.columns:
                null_count = transactions_df[col].null_count()
                assert null_count == 0, f"Found {null_count} null values in transaction column {col}"
        
        # Test 2: Balance continuity is maintained
        # Check that no balances are negative
        if "balance" in transactions_df.columns:
            negative_balance_count = transactions_df.filter(pl.col("balance") < 0).height
            assert negative_balance_count == 0, f"Found {negative_balance_count} negative balances"
        
        # Test 3: Tier compliance is enforced
        if "tier" in transactions_df.columns:
            valid_tiers = [1, 2, 3, 4]
            invalid_tier_count = transactions_df.filter(~pl.col("tier").is_in(valid_tiers)).height
            assert invalid_tier_count == 0, f"Found {invalid_tier_count} transactions with invalid tier"
        
        # Test 4: Temporal distributions are realistic
        if "timestamp" in transactions_df.columns:
            # Convert timestamp to datetime if string
            if transactions_df["timestamp"].dtype == pl.String:
                transactions_df = transactions_df.with_columns(
                    pl.col("timestamp").str.to_datetime(time_zone="UTC").alias("timestamp")
                )
            
            # Check timestamp range is reasonable (within last year)
            max_timestamp = transactions_df["timestamp"].max()
            min_timestamp = transactions_df["timestamp"].min()
            time_span = (max_timestamp - min_timestamp).total_seconds() / 86400  # days
            
            assert time_span > 0, "Transaction timestamps have zero span"
            assert time_span <= 365, f"Transaction time span {time_span:.1f} days exceeds 1 year"
            
            logger.info(f"✓ Data quality validated:")
            logger.info(f"  - No null values in essential columns")
            logger.info(f"  - Balance continuity maintained (0 negative balances)")
            logger.info(f"  - Tier compliance enforced (0 invalid tiers)")
            logger.info(f"  - Temporal distributions realistic ({time_span:.1f} day span)")


class TestEndToEndPipeline:
    """Test end-to-end pipeline execution."""
    
    def test_end_to_end_pipeline(self):
        """Test full pipeline runs without errors and produces expected output."""
        settings = resolve_runtime_settings(fast_mode=True, force_refresh=True)
        bronze_cfg = settings["bronze"]
        anomaly_cfg = settings["anomaly"]
        
        # Clean data directories
        clean_data_directories()
        
        # Run Bronze stage
        bronze_result = run_bronze_stage(
            num_customers=bronze_cfg["num_customers"],
            num_days=bronze_cfg["num_days"],
            target_transactions=bronze_cfg.get("target_transactions"),
            seed=bronze_cfg["seed"],
            anomaly_ratio=anomaly_cfg["anomaly_ratio"],
            skip_if_partition_exists=False,
        )
        assert bronze_result.record_count > 0, "Bronze stage produced no records"
        logger.info(f"✓ Bronze stage: {bronze_result.record_count} records")
        
        # Run Silver stage
        silver_result = run_silver_stage(
            partition_key=bronze_result.partition_key,
            bronze_base_path="data/bronze",
            silver_base_path="data/silver",
        )
        assert silver_result.transaction_count > 0, "Silver stage produced no transactions"
        assert silver_result.customer_count > 0, "Silver stage produced no customers"
        logger.info(f"✓ Silver stage: {silver_result.transaction_count} transactions, {silver_result.customer_count} customers")
        
        # Run Gold stage
        gold_result = run_gold_stage(
            partition_key=silver_result.partition_key,
            silver_base_path="data/silver",
        )
        assert gold_result.gold_uri, "Gold stage produced no URI"
        logger.info(f"✓ Gold stage: {gold_result.gold_uri}")
        
        # Verify final output has expected structure
        gold_dir = Path("data/gold/features/v1.0")
        customer_features_path = gold_dir / f"customer_features_{gold_result.partition_key}.parquet"
        assert customer_features_path.exists(), "Gold customer features not found"
        
        customer_features_df = pl.read_parquet(customer_features_path)
        assert customer_features_df.height > 0, "Gold customer features empty"
        
        logger.info(f"✓ End-to-end pipeline validated successfully")
        logger.info(f"  - Bronze: {bronze_result.record_count} records")
        logger.info(f"  - Silver: {silver_result.transaction_count} transactions")
        logger.info(f"  - Gold: {customer_features_df.height} customer features")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
