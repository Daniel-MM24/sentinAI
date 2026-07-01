"""
Medallion Pipeline Runner - Bronze → Silver → Gold

This script orchestrates the complete data pipeline by executing individual scripts
for each layer (Bronze, Silver, Gold). This approach provides process isolation and
makes it easier to catch errors at specific stages.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timezone
import uuid

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openlineage.client import OpenLineageClient
from openlineage.client.run import RunEvent, RunState, Run, Job, Dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_script(script_path: str, stage_name: str):
    """
    Execute a python script using subprocess and handle errors.
    """
    logger.info(f"--- Starting {stage_name} ---")
    logger.info(f"Running: python {script_path}")
    
    try:
        # Use subprocess to run the script, streaming output to console
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            text=True
        )
        logger.info(f"--- {stage_name} Completed Successfully ---")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"--- {stage_name} Failed ---")
        logger.error(f"Script {script_path} exited with status {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"--- {stage_name} Failed with unexpected error: {e} ---")
        return False

def run_complete_pipeline():
    """
    Execute the complete Bronze → Silver → Gold pipeline sequentially.
    """
    logger.info("Starting Orchestration: Medallion Pipeline (Bronze → Silver → Gold)")
    logger.info("=" * 60)
    
    # Initialize OpenLineage client for pipeline-level tracking
    ol_client = OpenLineageClient()
    run_id = str(uuid.uuid4())
    namespace = "sentinai.pipeline"
    job_name = "medallion_pipeline_orchestration"
    
    # Emit START event for pipeline orchestration
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
        logger.info(f"Pipeline Lineage START emitted: {job_name} (run_id={run_id})")
    except Exception as e:
        logger.warning(f"Failed to emit pipeline START event: {e}")
    
    # Check OpenLineage configuration
    openlineage_url = os.getenv("OPENLINEAGE_URL")
    if openlineage_url:
        logger.info(f"OpenLineage tracking enabled: {openlineage_url}")
    else:
        logger.warning("OPENLINEAGE_URL not set. Lineage events will go to console.")
        logger.info("Set OPENLINEAGE_URL=http://localhost:5000 to enable Marquez tracking")
    
    scripts_dir = Path(__file__).parent
    
    # Define the pipeline stages
    stages = [
        ("Bronze Layer", scripts_dir / "run_bronze.py"),
        ("Silver Layer", scripts_dir / "run_silver.py"),
        ("Gold Layer", scripts_dir / "run_gold.py"),
    ]
    
    # Execute stages sequentially
    pipeline_success = True
    for stage_name, script_path in stages:
        if not run_script(str(script_path), stage_name):
            logger.error(f"Pipeline halted at {stage_name} due to errors.")
            pipeline_success = False
            break
            
    if pipeline_success:
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("Data for all layers persisted in data/ directory.")
        logger.info("=" * 60)
        
        # Emit COMPLETE event for pipeline orchestration
        try:
            complete_event = RunEvent(
                eventType=RunState.COMPLETE,
                eventTime=datetime.now(timezone.utc).isoformat(),
                run=Run(runId=run_id),
                job=Job(namespace=namespace, name=job_name),
                producer="sentinai",
                inputs=[],
                outputs=[
                    Dataset(namespace="sentinai.bronze", name="bronze_transactions"),
                    Dataset(namespace="sentinai.silver", name="silver_transactions"),
                    Dataset(namespace="sentinai.silver", name="silver_customers"),
                    Dataset(namespace="sentinai.gold", name="gold_feature_store")
                ]
            )
            ol_client.emit(complete_event)
            logger.info(f"Pipeline Lineage COMPLETE emitted: {job_name} (run_id={run_id})")
        except Exception as e:
            logger.warning(f"Failed to emit pipeline COMPLETE event: {e}")
    else:
        # Emit FAIL event for pipeline orchestration
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
            logger.info(f"Pipeline Lineage FAIL emitted: {job_name} (run_id={run_id})")
        except Exception as e:
            logger.warning(f"Failed to emit pipeline FAIL event: {e}")
        
        sys.exit(1)
    
    if openlineage_url:
        logger.info(f"View lineage DAG in Marquez UI: http://localhost:3001")
        logger.info(f"Marquez API: {openlineage_url}")

if __name__ == "__main__":
    run_complete_pipeline()
