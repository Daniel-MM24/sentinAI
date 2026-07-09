"""CLI wrapper for the Bronze layer ingestion stage."""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.medallion_stages import resolve_runtime_settings, run_bronze_stage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Bronze layer ingestion")
    parser.add_argument("--fast-mode", action="store_true")
    parser.add_argument("--full-mode", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fast_mode = args.fast_mode or os.getenv("SENTINAI_FAST_MODE", "1").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if args.full_mode:
        fast_mode = False

    settings = resolve_runtime_settings(
        fast_mode=fast_mode,
        force_refresh=args.force_refresh,
    )
    bronze_cfg = settings["bronze"]
    anomaly_cfg = settings["anomaly"]

    try:
        result = run_bronze_stage(
            num_customers=bronze_cfg["num_customers"],
            num_days=bronze_cfg["num_days"],
            target_transactions=bronze_cfg.get("target_transactions"),
            seed=bronze_cfg["seed"],
            anomaly_ratio=anomaly_cfg["anomaly_ratio"],
            skip_if_partition_exists=not args.force_refresh,
        )
    except Exception:
        logger.exception("Bronze layer failed")
        sys.exit(1)

    logger.info("Bronze layer completed: %s records at %s", result.record_count, result.bronze_path)


if __name__ == "__main__":
    main()
