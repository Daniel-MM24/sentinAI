import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.bronze import BronzeLayer
from src.data.synthetic_engine import SyntheticMpesaGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Bronze Layer Ingestion")
    
    try:
        bronze_layer = BronzeLayer(bronze_base_path="data/bronze")
        
        logger.info("Generating synthetic M-Pesa transaction data...")
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
        
        synthetic_data = generator.generate_batch(n_records=1000, num_users=200)
        logger.info(f"Generated {synthetic_data.height} synthetic records")
        
        partition_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        bronze_path = bronze_layer.ingest_synthetic_data(
            synthetic_data,
            source_table="synthetic_transactions",
            partition_key=partition_key
        )
        logger.info(f"Bronze data written to: {bronze_path}")
        logger.info("Bronze Layer Ingestion successfully completed.")
        
    except Exception as e:
        logger.error(f"Bronze Layer failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
