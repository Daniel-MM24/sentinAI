import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone
import polars as pl

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.bronze import BronzeLayer
from src.datasets.silver import SilverLayer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Silver Layer Transformation")
    
    try:
        bronze_layer = BronzeLayer(bronze_base_path="data/bronze")
        silver_layer = SilverLayer()
        
        partition_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        logger.info(f"Reading bronze data for partition: {partition_key}")
        bronze_df = bronze_layer.read_bronze_partition(partition_key)
        logger.info(f"Read {bronze_df.height} records from Bronze layer")
        
        logger.info("Transforming data to Silver...")
        transaction_fact_df, customer_dimension_df = silver_layer.transform_to_silver(
            bronze_df,
            partition_key=partition_key
        )
        
        # Persist silver dataframes to disk
        silver_dir = Path("data/silver")
        silver_dir.mkdir(parents=True, exist_ok=True)
        
        # Persist transaction fact stream
        transactions_path = silver_dir / f"silver_transactions_{partition_key}.parquet"
        transaction_fact_df.write_parquet(transactions_path)
        
        # Persist customer dimension registry
        customers_path = silver_dir / f"silver_customers_{partition_key}.parquet"
        customer_dimension_df.write_parquet(customers_path)
        
        logger.info(f"Transaction fact stream persisted to: {transactions_path}")
        logger.info(f"Customer dimension registry persisted to: {customers_path}")
        logger.info(f"Silver transformation complete: {bronze_df.height} → {transaction_fact_df.height} (facts) + {customer_dimension_df.height} (customers)")
        logger.info("Silver Layer Transformation successfully completed.")
        
    except Exception as e:
        logger.error(f"Silver Layer failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
