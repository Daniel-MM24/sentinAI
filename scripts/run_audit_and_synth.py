"""CLI wrapper for the SentinAI medallion pipeline orchestrator."""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.medallion_stages import run_medallion_orchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SentinAI bronze/silver/gold medallion pipeline"
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Use a smaller synthetic footprint for fast local runs",
    )
    parser.add_argument(
        "--full-mode",
        action="store_true",
        help="Use the default full synthetic footprint",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Delete existing data directories before running",
    )
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

    openlineage_url = os.getenv("OPENLINEAGE_URL")
    if openlineage_url:
        logger.info("OpenLineage tracking enabled: %s", openlineage_url)
    else:
        logger.warning("OPENLINEAGE_URL not set. Lineage events will go to console.")

    try:
        result = run_medallion_orchestrator(
            fast_mode=fast_mode,
            force_refresh=args.force_refresh,
        )
    except Exception:
        logger.exception("Medallion pipeline failed")
        sys.exit(1)

    logger.info("Pipeline completed successfully")
    logger.info("Bronze records: %s", result["bronze"].record_count)
    logger.info("Silver transactions: %s", result["silver"].transaction_count)
    logger.info("Gold feature store: %s", result["gold"].gold_uri)

    if openlineage_url:
        logger.info("View lineage DAG in Marquez UI: http://localhost:3001")


if __name__ == "__main__":
    main()
