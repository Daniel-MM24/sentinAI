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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    for stage_name, script_path in stages:
        if not run_script(str(script_path), stage_name):
            logger.error(f"Pipeline halted at {stage_name} due to errors.")
            sys.exit(1)
            
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("Data for all layers persisted in data/ directory.")
    logger.info("=" * 60)
    
    if openlineage_url:
        logger.info(f"View lineage DAG in Marquez UI: http://localhost:3001")
        logger.info(f"Marquez API: {openlineage_url}")

if __name__ == "__main__":
    run_complete_pipeline()
