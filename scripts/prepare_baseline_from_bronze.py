"""Prepare existing bronze data as TVAE baseline.

This script takes existing bronze transaction data and prepares it
as the baseline for TVAE training by filtering to the 8 core columns.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Core TVAE columns required for model training
CORE_TVAE_COLUMNS = [
    "customer_id",
    "tier",
    "archetype",
    "transaction_type",
    "amount",
    "timestamp",
    "direction",
    "is_international",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare existing bronze data as TVAE baseline"
    )
    parser.add_argument(
        "--bronze-path",
        type=str,
        required=True,
        help="Path to existing bronze transaction data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/bronze",
        help="Output directory for baseline data (default: data/bronze)",
    )
    parser.add_argument(
        "--partition-key",
        type=str,
        required=True,
        help="Partition key for output (e.g., 2026-08-05)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Loading bronze data from: %s", args.bronze_path)
    df = pl.read_parquet(args.bronze_path)
    logger.info("Loaded %s transactions", len(df))
    logger.info("Columns: %s", df.columns)

    # Load customer metadata to get tier information
    customers_metadata_path = Path(args.output_dir).parent / "customers_metadata.csv"
    if customers_metadata_path.exists():
        logger.info("Loading customer metadata from: %s", customers_metadata_path)
        customers_df = pl.read_csv(customers_metadata_path)
        logger.info("Loaded %s customers", len(customers_df))
        
        # Join transactions with customer metadata to get tier
        df = df.join(customers_df.select(["customer_id", "customer_tier"]), on="customer_id", how="left")
        # Map customer_tier to tier
        df = df.with_columns(
            pl.col("customer_tier").alias("tier")
        )
    else:
        logger.warning("Customer metadata not found, using default tier")
        df = df.with_columns(
            pl.lit(1).alias("tier")
        )

    # Map transaction_type to direction (simplified mapping)
    # Assume Send Money/Withdrawal are outflows, Deposit/Receive are inflows
    df = df.with_columns(
        pl.when(pl.col("transaction_type").is_in(["Send Money", "Withdrawal", "Agent Withdrawal"]))
        .then(pl.lit("outflow"))
        .otherwise(pl.lit("inflow"))
        .alias("direction")
    )

    # Add missing columns with defaults
    if "archetype" not in df.columns:
        logger.warning("Adding default archetype")
        df = df.with_columns(
            pl.lit("retail_standard").alias("archetype")
        )

    if "is_international" not in df.columns:
        logger.warning("Adding default is_international")
        df = df.with_columns(
            pl.lit(False).alias("is_international")
        )

    # Select only core TVAE columns
    logger.info("Selecting core TVAE columns...")
    core_df = df.select(CORE_TVAE_COLUMNS)

    # Remove any anomaly flags or labels
    for col in core_df.columns:
        if "anomaly" in col.lower() or "label" in col.lower() or "launderer" in col.lower():
            logger.warning("Removing unexpected column: %s", col)
            core_df = core_df.drop(col)

    # Ensure tier is integer
    core_df = core_df.with_columns(
        pl.col("tier").cast(pl.Int32)
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write output
    output_path = output_dir / f"monte_carlo_baseline_{args.partition_key}.parquet"
    core_df.write_parquet(output_path)
    logger.info("Baseline data written to: %s", output_path)
    logger.info("Total transactions: %s", len(core_df))
    logger.info("Unique customers: %s", core_df["customer_id"].n_unique())


if __name__ == "__main__":
    main()
