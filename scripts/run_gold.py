"""CLI wrapper for the Gold layer feature store materialization stage."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.medallion_stages import _clean_layer, run_gold_stage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    # Clean Gold layer before writing fresh data
    data_dir = Path(__file__).parent.parent / "data"
    _clean_layer(data_dir, "gold")

    try:
        result = run_gold_stage()
    except Exception:
        logger.exception("Gold layer failed")
        sys.exit(1)

    logger.info("Gold layer completed: feature store at %s", result.gold_uri)


if __name__ == "__main__":
    main()
