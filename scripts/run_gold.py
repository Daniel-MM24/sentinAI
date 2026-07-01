import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone
import polars as pl
import uuid

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.gold import GoldLayer
from openlineage.client import OpenLineageClient
from openlineage.client.run import RunEvent, RunState, Run, Job, Dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Gold Layer Materialization")
    
    # Initialize OpenLineage client
    ol_client = OpenLineageClient()
    run_id = str(uuid.uuid4())
    namespace = "sentinai.gold"
    job_name = "gold_materialize_script"
    
    # Emit START event
    try:
        start_event = RunEvent(
            eventType=RunState.START,
            eventTime=datetime.now(timezone.utc).isoformat(),
            run=Run(runId=run_id),
            job=Job(namespace=namespace, name=job_name),
            producer="sentinai",
            inputs=[
                Dataset(namespace="sentinai.silver", name="silver_transactions"),
                Dataset(namespace="sentinai.silver", name="silver_customers")
            ],
            outputs=[]
        )
        ol_client.emit(start_event)
        logger.info(f"Lineage START emitted: {job_name} (run_id={run_id})")
    except Exception as e:
        logger.warning(f"Failed to emit START event: {e}")
    
    try:
        gold_layer = GoldLayer(version="v1.0")
        
        partition_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        # Read transaction fact stream
        transactions_path = Path("data/silver") / f"silver_transactions_{partition_key}.parquet"
        if not transactions_path.exists():
            logger.error(f"Transaction fact stream not found at {transactions_path}")
            sys.exit(1)
        
        logger.info(f"Reading transaction fact stream from: {transactions_path}")
        transactions_df = pl.read_parquet(transactions_path)
        logger.info(f"Read {transactions_df.height} transaction records from Silver layer")
        
        # Read customer dimension registry
        customers_path = Path("data/silver") / f"silver_customers_{partition_key}.parquet"
        if not customers_path.exists():
            logger.error(f"Customer dimension registry not found at {customers_path}")
            sys.exit(1)
        
        logger.info(f"Reading customer dimension registry from: {customers_path}")
        customers_df = pl.read_parquet(customers_path)
        logger.info(f"Read {customers_df.height} customer records from Silver layer")
        
        logger.info("Creating feature store...")
        gold_uri = gold_layer.create_feature_store(
            transactions_df=transactions_df,
            customers_df=customers_df
        )
        
        logger.info(f"Gold feature store created at: {gold_uri}")
        logger.info("Gold Layer Materialization successfully completed.")
        
        # Emit COMPLETE event
        try:
            complete_event = RunEvent(
                eventType=RunState.COMPLETE,
                eventTime=datetime.now(timezone.utc).isoformat(),
                run=Run(runId=run_id),
                job=Job(namespace=namespace, name=job_name),
                producer="sentinai",
                inputs=[
                    Dataset(namespace="sentinai.silver", name="silver_transactions"),
                    Dataset(namespace="sentinai.silver", name="silver_customers")
                ],
                outputs=[Dataset(namespace=namespace, name="gold_feature_store")]
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
                inputs=[
                    Dataset(namespace="sentinai.silver", name="silver_transactions"),
                    Dataset(namespace="sentinai.silver", name="silver_customers")
                ],
                outputs=[]
            )
            ol_client.emit(fail_event)
            logger.info(f"Lineage FAIL emitted: {job_name} (run_id={run_id})")
        except Exception as lineage_error:
            logger.warning(f"Failed to emit FAIL event: {lineage_error}")
        
        logger.error(f"Gold Layer failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
