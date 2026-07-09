"""CLI wrapper for the Silver layer AML transformation stage."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.medallion_stages import run_silver_stage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    try:
        result = run_silver_stage()
    except Exception:
        logger.exception("Silver layer failed")
        sys.exit(1)

    logger.info(
        "Silver layer completed: %s transactions, %s customers",
        result.transaction_count,
        result.customer_count,
    )


if __name__ == "__main__":
    main()
