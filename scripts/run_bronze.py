import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone
import uuid

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.bronze import BronzeLayer
from src.data.synthetic_generator import AMLGenerator, AMLGeneratorConfig
from openlineage.client import OpenLineageClient
from openlineage.client.run import RunEvent, RunState, Run, Job, Dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Bronze Layer Ingestion")
    
    # Initialize OpenLineage client
    ol_client = OpenLineageClient()
    run_id = str(uuid.uuid4())
    namespace = "sentinai.bronze"
    job_name = "bronze_ingest_script"
    
    # Emit START event
    try:
        start_event = RunEvent(
            eventType=RunState.START,
            eventTime=datetime.now(timezone.utc).isoformat(),
            run=Run(runId=run_id),
            job=Job(namespace=namespace, name=job_name),
            producer="sentinai",
            inputs=[],
            outputs=[]
        )
        ol_client.emit(start_event)
        logger.info(f"Lineage START emitted: {job_name} (run_id={run_id})")
    except Exception as e:
        logger.warning(f"Failed to emit START event: {e}")
    
    try:
        os.makedirs("data/bronze/transactions", exist_ok=True)
        bronze_layer = BronzeLayer(bronze_base_path="data/bronze")
        
        fast_mode = os.getenv("SENTINAI_FAST_MODE", "1").lower() in {"1", "true", "yes", "on"}
        config = AMLGeneratorConfig(
            num_customers=int(os.getenv("SENTINAI_BROZE_NUM_CUSTOMERS", "1000" if fast_mode else "1000000")),
            num_days=int(os.getenv("SENTINAI_BROZE_NUM_DAYS", "7" if fast_mode else "365")),
            seed=int(os.getenv("SENTINAI_BROZE_SEED", "42"))
        )
        logger.info(
            "Generating AML-focused synthetic transaction data with config: customers=%s days=%s seed=%s",
            config.num_customers,
            config.num_days,
            config.seed,
        )
        generator = AMLGenerator(config)
        synthetic_data = generator.generate()
        logger.info(f"Generated {synthetic_data.height} synthetic records")
        
        partition_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        # Check if Bronze partition already has data to prevent duplicate ingestion
        existing_bronze = bronze_layer.read_bronze_partition(partition_key)
        if existing_bronze.height > 0:
            logger.warning(f"Bronze partition {partition_key} already contains {existing_bronze.height} records. Skipping ingestion to prevent duplicates.")
            logger.info("To re-ingest, remove the existing partition data first.")
            bronze_path = f"data/bronze/transactions/{partition_key}"
        else:
            bronze_path = bronze_layer.ingest_synthetic_data(
                synthetic_data,
                source_table="synthetic_transactions",
                partition_key=partition_key
            )
        logger.info(f"Bronze data written to: {bronze_path}")
        logger.info("Bronze Layer Ingestion successfully completed.")
        
        # Emit COMPLETE event
        try:
            complete_event = RunEvent(
                eventType=RunState.COMPLETE,
                eventTime=datetime.now(timezone.utc).isoformat(),
                run=Run(runId=run_id),
                job=Job(namespace=namespace, name=job_name),
                producer="sentinai",
                inputs=[],
                outputs=[Dataset(namespace=namespace, name="bronze_transactions")]
            )
            ol_client.emit(complete_event)
            logger.info(f"Lineage COMPLETE emitted: {job_name} (run_id={run_id})")
        except Exception as e:
            logger.warning(f"Failed to emit COMPLETE event: {e}")
        
    except Exception as e:
        # Emit FAIL event
        try:
            fail_event = RunEvent(
                eventType=RunState.FAIL,
                eventTime=datetime.now(timezone.utc).isoformat(),
                run=Run(runId=run_id),
                job=Job(namespace=namespace, name=job_name),
                producer="sentinai",
                inputs=[],
                outputs=[]
            )
            ol_client.emit(fail_event)
            logger.info(f"Lineage FAIL emitted: {job_name} (run_id={run_id})")
        except Exception as lineage_error:
            logger.warning(f"Failed to emit FAIL event: {lineage_error}")
        
        logger.error(f"Bronze Layer failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
