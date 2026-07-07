import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def resolve_runtime_settings(fast_mode: bool = False, force_refresh: bool = False) -> Dict[str, Any]:
    """Resolve runtime settings for the audit/synthesis workflow.

    Fast mode prefers a compact synthetic dataset so the whole pipeline can run
    quickly for local development and demos while still exercising the same
    stages end-to-end.
    """
    settings: Dict[str, Any] = {
        "clean_data_directories": force_refresh or not fast_mode,
        "bronze": {
            "num_customers": 5000 if fast_mode else 1000000,
            "num_days": 30 if fast_mode else 365,
            "seed": 42,
        },
    }

    if fast_mode:
        logger.info("Fast mode enabled: using a compact synthetic dataset for near real-time execution")
    else:
        logger.info("Full mode enabled: using the standard synthetic dataset size")

    return settings


def run_subprocess(script_path: str, extra_env: Dict[str, str] | None = None):
    logger.info(f"Executing decoupled script: {script_path}")
    project_root = str(Path(__file__).parent.parent)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    env.setdefault("PYTHONUNBUFFERED", "1")

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=project_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    if result.returncode != 0:
        logger.error(f"Script {script_path} failed with exit code {result.returncode}")
        if result.stdout:
            logger.error(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            logger.error(f"STDERR:\n{result.stderr}")
        sys.exit(result.returncode)

    logger.info(f"Script {script_path} completed successfully.")
    if result.stdout:
        logger.info(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        logger.warning(f"STDERR:\n{result.stderr}")


def clean_data_directories():
    import shutil

    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    for layer in ["bronze", "silver", "gold"]:
        layer_dir = data_dir / layer
        if layer_dir.exists():
            logger.info(f"Cleaning {layer} directory: {layer_dir}")
            shutil.rmtree(layer_dir)
        layer_dir.mkdir(parents=True, exist_ok=True)

    synthetic_db = data_dir / "synthetic.duckdb"
    if synthetic_db.exists():
        logger.info(f"Cleaning synthetic database: {synthetic_db}")
        synthetic_db.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SentinAI bronze/silver/gold pipeline")
    parser.add_argument("--fast-mode", action="store_true", help="Use a smaller synthetic footprint for fast local runs")
    parser.add_argument("--full-mode", action="store_true", help="Use the default full synthetic footprint")
    parser.add_argument("--force-refresh", action="store_true", help="Delete existing data directories before running")
    return parser.parse_args()


def main():
    args = parse_args()
    fast_mode = args.fast_mode or os.getenv("SENTINAI_FAST_MODE", "1").lower() in {"1", "true", "yes", "on"}
    if args.full_mode:
        fast_mode = False

    settings = resolve_runtime_settings(fast_mode=fast_mode, force_refresh=args.force_refresh)

    if settings["clean_data_directories"]:
        clean_data_directories()

    base_dir = Path(__file__).parent
    scripts = [
        base_dir / "run_bronze.py",
        base_dir / "run_silver.py",
        base_dir / "run_gold.py",
    ]

    extra_env = {
        "SENTINAI_FAST_MODE": str(fast_mode).lower(),
        "SENTINAI_BROZE_NUM_CUSTOMERS": str(settings["bronze"]["num_customers"]),
        "SENTINAI_BROZE_NUM_DAYS": str(settings["bronze"]["num_days"]),
        "SENTINAI_BROZE_SEED": str(settings["bronze"]["seed"]),
    }

    for script in scripts:
        if not script.exists():
            logger.error(f"Required script not found: {script}")
            sys.exit(1)

        run_subprocess(str(script), extra_env=extra_env)


if __name__ == "__main__":
    main()
