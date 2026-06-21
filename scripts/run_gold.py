import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone
import polars as pl

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.gold import GoldLayer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Gold Layer Materialization")
    
    try:
        gold_layer = GoldLayer(version="v1.0")
        
        partition_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        silver_path = Path("data/silver") / f"silver_{partition_key}.parquet"
        
        if not silver_path.exists():
            logger.error(f"Silver data not found at {silver_path}")
            sys.exit(1)
            
        logger.info(f"Reading silver data from: {silver_path}")
        silver_df = pl.read_parquet(silver_path)
        logger.info(f"Read {silver_df.height} records from Silver layer")
        
        logger.info("Creating feature store...")
        gold_uri = gold_layer.create_feature_store(
            transactions_df=silver_df,
            customers_df=silver_df  # Using same data for demo
        )
        
        logger.info(f"Gold feature store created at: {gold_uri}")
        logger.info("Gold Layer Materialization successfully completed.")
        
    except Exception as e:
        logger.error(f"Gold Layer failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
