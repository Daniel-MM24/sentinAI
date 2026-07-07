"""
Medallion Pipeline Runner - Bronze → Silver → Gold

This script orchestrates the complete data pipeline by executing individual scripts
for each layer (Bronze, Silver, Gold). The orchestration follows the same
lineage-aware structure as the synthetic audit/synthesis entrypoint.
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.lineage_decorator import emit_transformation_metadata, lineage_trace

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SentinAI bronze/silver/gold pipeline")
    parser.add_argument("--fast-mode", action="store_true", help="Use a smaller synthetic footprint for fast local runs")
    parser.add_argument("--full-mode", action="store_true", help="Use the default full synthetic footprint")
    parser.add_argument("--force-refresh", action="store_true", help="Delete existing data directories before running")
    return parser.parse_args()


def resolve_runtime_settings(fast_mode: bool = False, force_refresh: bool = False) -> Dict[str, Any]:
    settings: Dict[str, Any] = {
        "clean_data_directories": force_refresh or not fast_mode,
        "fast_mode": fast_mode,
    }

    if fast_mode:
        logger.info("Fast mode enabled for the medallion pipeline")
    else:
        logger.info("Full mode enabled for the medallion pipeline")

    return settings


def clean_data_directories() -> None:
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    for layer in ["bronze", "silver", "gold"]:
        layer_dir = data_dir / layer
        if layer_dir.exists():
            logger.info(f"Cleaning {layer} directory: {layer_dir}")
            shutil.rmtree(layer_dir)
        layer_dir.mkdir(parents=True, exist_ok=True)


def run_subprocess(script_path: str, stage_name: str, extra_env: Dict[str, str] | None = None) -> bool:
    logger.info(f"--- Starting {stage_name} ---")
    logger.info(f"Running: python {script_path}")

    project_root = str(Path(__file__).parent.parent)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    env.setdefault("PYTHONUNBUFFERED", "1")

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=project_root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(f"--- {stage_name} Failed with unexpected error: {exc} ---")
        return False

    if result.returncode != 0:
        logger.error(f"--- {stage_name} Failed ---")
        logger.error(f"Script {script_path} exited with status {result.returncode}")
        if result.stdout:
            logger.error(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            logger.error(f"STDERR:\n{result.stderr}")
        return False

    logger.info(f"--- {stage_name} Completed Successfully ---")
    if result.stdout:
        logger.info(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        logger.warning(f"STDERR:\n{result.stderr}")
    return True


@lineage_trace(
    job_name="run_medallion_pipeline",
    input_datasets=["pipeline_config"],
    output_datasets=["bronze_transactions", "silver_transactions", "gold_feature_store"],
    namespace="sentinai.pipeline",
)
def run_pipeline(fast_mode: bool = False, force_refresh: bool = False) -> Dict[str, Any]:
    """Execute the complete Bronze → Silver → Gold pipeline sequentially."""
    logger.info("Starting Orchestration: Medallion Pipeline (Bronze → Silver → Gold)")
    logger.info("=" * 60)

    settings = resolve_runtime_settings(fast_mode=fast_mode, force_refresh=force_refresh)
    if settings["clean_data_directories"]:
        clean_data_directories()

    openlineage_url = os.getenv("OPENLINEAGE_URL")
    if openlineage_url:
        logger.info(f"OpenLineage tracking enabled: {openlineage_url}")
    else:
        logger.warning("OPENLINEAGE_URL not set. Lineage events will go to console.")
        logger.info("Set OPENLINEAGE_URL=http://localhost:8001 to enable Marquez tracking")

    scripts_dir = Path(__file__).parent
    stages: List[Tuple[str, Path]] = [
        ("Bronze Layer", scripts_dir / "run_bronze.py"),
        ("Silver Layer", scripts_dir / "run_silver.py"),
        ("Gold Layer", scripts_dir / "run_gold.py"),
    ]

    extra_env = {
        "SENTINAI_FAST_MODE": str(fast_mode).lower(),
    }

    completed_stages: List[str] = []
    for stage_name, script_path in stages:
        if not run_subprocess(str(script_path), stage_name, extra_env=extra_env):
            logger.error(f"Pipeline halted at {stage_name} due to errors.")
            raise RuntimeError(f"Pipeline halted at {stage_name}")
        completed_stages.append(stage_name)

    emit_transformation_metadata(
        job_name="run_medallion_pipeline",
        run_id=str(uuid.uuid4()),
        transformation_python="Execute bronze, silver, and gold stage scripts",
        input_rows=0,
        output_rows=len(completed_stages),
    )

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("Data for all layers persisted in data/ directory.")
    logger.info("=" * 60)

    if openlineage_url:
        logger.info(f"View lineage DAG in Marquez UI: http://localhost:3001")
        logger.info(f"Marquez API: {openlineage_url}")

    return {
        "completed_stages": completed_stages,
        "fast_mode": fast_mode,
    }


def main() -> None:
    args = parse_args()
    fast_mode = args.fast_mode or os.getenv("SENTINAI_FAST_MODE", "0").lower() in {"1", "true", "yes", "on"}
    if args.full_mode:
        fast_mode = False

    run_pipeline(fast_mode=fast_mode, force_refresh=args.force_refresh)


if __name__ == "__main__":
    main()
