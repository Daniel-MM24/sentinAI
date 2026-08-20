"""Generate Monte Carlo baseline data using BehavioralTransactionGenerator.

This script generates clean baseline transaction data (no anomaly injection)
with realistic M-PESA behavioral patterns, tier constraints, and temporal patterns.
Output is filtered to the 8 core TVAE columns required for model training.
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl

from src.data.behavioral_generator import (
    BehavioralGeneratorConfig,
    BehavioralTransactionGenerator,
)

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
        description="Generate Monte Carlo baseline transaction data"
    )
    parser.add_argument(
        "--num-customers",
        type=int,
        default=10_000,
        help="Number of customers to generate (default: 10,000)",
    )
    parser.add_argument(
        "--num-transactions",
        type=int,
        default=100_000,
        help="Number of transactions to generate (default: 100,000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
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
        default=None,
        help="Partition key for output (default: current date)",
    )
    return parser.parse_args()


def log_generation_statistics(df: pl.DataFrame) -> None:
    """Log detailed distribution summaries for each core column."""
    if df.is_empty():
        logger.warning("No transactions generated")
        return

    total = len(df)
    logger.info("=" * 60)
    logger.info("MONTE CARLO BASELINE GENERATION STATISTICS")
    logger.info("=" * 60)
    logger.info("Total transactions: %s", total)
    logger.info("Unique customers: %s", df["customer_id"].n_unique())
    logger.info("")

    # Customer tier distribution
    logger.info("Tier distribution:")
    tier_dist = df.group_by("tier").len().sort("tier")
    for row in tier_dist.iter_rows(named=True):
        logger.info(
            "  Tier %s: %s (%.1f%%)",
            row["tier"],
            row["len"],
            row["len"] / total * 100,
        )
    logger.info("")

    # Archetype distribution
    logger.info("Archetype distribution:")
    arch_dist = df.group_by("archetype").len().sort("len", descending=True)
    for row in arch_dist.iter_rows(named=True):
        logger.info(
            "  %s: %s (%.1f%%)",
            row["archetype"],
            row["len"],
            row["len"] / total * 100,
        )
    logger.info("")

    # Transaction type distribution
    logger.info("Transaction type distribution:")
    tx_dist = df.group_by("transaction_type").len().sort("len", descending=True)
    for row in tx_dist.iter_rows(named=True):
        logger.info(
            "  %s: %s (%.1f%%)",
            row["transaction_type"],
            row["len"],
            row["len"] / total * 100,
        )
    logger.info("")

    # Direction distribution
    logger.info("Direction distribution:")
    dir_dist = df.group_by("direction").len().sort("direction")
    for row in dir_dist.iter_rows(named=True):
        logger.info(
            "  %s: %s (%.1f%%)",
            row["direction"],
            row["len"],
            row["len"] / total * 100,
        )
    logger.info("")

    # International flag distribution
    intl_count = df.filter(pl.col("is_international")).height
    logger.info(
        "International transactions: %s (%.1f%%)",
        intl_count,
        intl_count / total * 100,
    )
    logger.info("")

    # Amount statistics
    logger.info("Amount statistics (KES):")
    logger.info("  Mean: %.2f", df["amount"].mean())
    logger.info("  Median: %.2f", df["amount"].median())
    logger.info("  Std Dev: %.2f", df["amount"].std())
    logger.info("  Min: %.2f", df["amount"].min())
    logger.info("  Max: %.2f", df["amount"].max())
    logger.info("  25th percentile: %.2f", df["amount"].quantile(0.25))
    logger.info("  75th percentile: %.2f", df["amount"].quantile(0.75))
    logger.info("")

    # Temporal statistics
    logger.info("Temporal statistics:")
    df_with_ts = df.with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime, format="%+")
    )
    logger.info("  Start: %s", df_with_ts["timestamp"].min())
    logger.info("  End: %s", df_with_ts["timestamp"].max())
    logger.info("  Duration: %s days", (df_with_ts["timestamp"].max() - df_with_ts["timestamp"].min()).total_seconds() / 86400)
    logger.info("")

    # Transactions per customer
    logger.info("Transactions per customer:")
    tx_per_cust = df.group_by("customer_id").len()
    logger.info("  Mean: %.2f", tx_per_cust["len"].mean())
    logger.info("  Median: %.2f", tx_per_cust["len"].median())
    logger.info("  Min: %s", tx_per_cust["len"].min())
    logger.info("  Max: %s", tx_per_cust["len"].max())
    logger.info("")

    logger.info("=" * 60)


def main() -> None:
    args = parse_args()

    # Generate partition key if not provided
    partition_key = args.partition_key or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    logger.info("Starting Monte Carlo baseline generation")
    logger.info("Configuration:")
    logger.info("  Customers: %s", args.num_customers)
    logger.info("  Transactions: %s", args.num_transactions)
    logger.info("  Seed: %s", args.seed)
    logger.info("  Output: %s/monte_carlo_baseline_%s.parquet", args.output_dir, partition_key)

    # Configure generator with CBK-aligned tier caps and constraints
    config = BehavioralGeneratorConfig(
        seed=args.seed,
        num_customers=args.num_customers,
        # Tier caps are already CBK-aligned in the default config
        # Tier 1: 10K, Tier 2: 50K, Tier 3: 150K, Tier 4: 500K
        # Daily velocity: Tier 1: 25K, Tier 2: 100K, Tier 3: 500K, Tier 4: 10M
        # Balance caps: Tier 1: 50K, Tier 2: 200K, Tier 3: 1M, Tier 4: 5M
    )

    generator = BehavioralTransactionGenerator(config)

    # Generate transactions using the temporal model for realistic timestamps
    logger.info("Generating transactions with temporal model and tier constraints...")
    df = generator.generate_transactions(
        n_transactions=args.num_transactions,
        output_path="/tmp/monte_carlo_raw.csv",  # Temporary file, will be filtered
    )

    # Filter to only the 8 core TVAE columns
    logger.info("Filtering to core TVAE columns...")
    core_df = df.select(CORE_TVAE_COLUMNS)

    # Ensure no anomaly flags or labels are present
    # (BehavioralTransactionGenerator doesn't add these by default,
    # but we explicitly ensure they're not in the output)
    for col in core_df.columns:
        if "anomaly" in col.lower() or "label" in col.lower():
            logger.warning("Found unexpected column %s, removing", col)
            core_df = core_df.drop(col)

    # Verify all core columns are present
    missing_cols = set(CORE_TVAE_COLUMNS) - set(core_df.columns)
    if missing_cols:
        raise ValueError(f"Missing core columns: {missing_cols}")

    # Ensure archetype is present (from customer state)
    if "archetype" not in core_df.columns:
        logger.warning("Archetype not in generated data, adding from customer state")
        # Add archetype from customer state if not already present
        core_df = core_df.with_columns(
            pl.lit("retail_standard").alias("archetype")
        )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write output
    output_path = output_dir / f"monte_carlo_baseline_{partition_key}.parquet"
    core_df.write_parquet(output_path)
    logger.info("Baseline data written to: %s", output_path)

    # Log generation statistics
    log_generation_statistics(core_df)

    logger.info("Monte Carlo baseline generation complete")


if __name__ == "__main__":
    main()
