"""Train TVAE model on Monte Carlo baseline data.

This script reads Monte Carlo baseline data, trains a TVAE model,
validates it with KS tests, and saves the trained model with metrics.
"""

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.tvae_generator import TVAEGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Log-transform parameters from stat_distribution.md
LOG_TRANSFORM_PARAMS = {
    "amount_mean": 6.02,  # μ from FY26 profile
    "amount_std": 1.25,   # σ from FY26 profile
    "clip_min": 1.0,
    "clip_max": 250000.0,
}

# Core TVAE columns
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
        description="Train TVAE model on Monte Carlo baseline data"
    )
    parser.add_argument(
        "--partition-key",
        type=str,
        required=True,
        help="Partition key for baseline data (e.g., 2026-08-05)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/bronze",
        help="Input directory for baseline data (default: data/bronze)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs (default: 300)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Training batch size (default: 500)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for trained model (default: models)",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="reports",
        help="Output directory for training report (default: reports)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of samples for validation (default: 1000)",
    )
    return parser.parse_args()


def log_transform_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log-transform to amount column using parameters from stat_distribution.md."""
    df = df.copy()
    
    # Clip amount to valid range
    df["amount"] = df["amount"].clip(
        lower=LOG_TRANSFORM_PARAMS["clip_min"],
        upper=LOG_TRANSFORM_PARAMS["clip_max"]
    )
    
    # Apply log-transform: log(amount + 1) to handle zero values
    df["amount"] = np.log1p(df["amount"])
    
    logger.info("Applied log-transform to amount column")
    logger.info(f"  Original range: [{LOG_TRANSFORM_PARAMS['clip_min']}, {LOG_TRANSFORM_PARAMS['clip_max']}]")
    logger.info(f"  Log-transformed range: [{df['amount'].min():.4f}, {df['amount'].max():.4f}]")
    
    return df


def ensure_categorical_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure categorical columns are properly typed."""
    df = df.copy()
    
    categorical_columns = [
        "customer_id",
        "tier",
        "archetype",
        "transaction_type",
        "direction",
        "is_international",
    ]
    
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
            logger.info(f"Converted {col} to string type")
    
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data for TVAE training."""
    logger.info("Preprocessing data...")
    
    # Ensure all core columns are present
    missing_cols = set(CORE_TVAE_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing core columns: {missing_cols}")
    
    # Select only core columns
    df = df[CORE_TVAE_COLUMNS].copy()
    
    # Apply log-transform to amount
    df = log_transform_amount(df)
    
    # Ensure categorical types
    df = ensure_categorical_types(df)
    
    # Convert timestamp to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    logger.info(f"Preprocessing complete. Shape: {df.shape}")
    return df


def validate_model(
    generator: TVAEGenerator,
    train_df: pd.DataFrame,
    sample_size: int = 1000,
) -> Dict[str, Any]:
    """Validate model by sampling and comparing distributions using KS tests."""
    logger.info(f"Validating model with {sample_size} samples...")
    
    # Sample from trained model
    synthetic_df = generator.sample(sample_size)
    
    # Apply same preprocessing to synthetic data for comparison
    synthetic_df["amount"] = np.log1p(synthetic_df["amount"].clip(
        lower=LOG_TRANSFORM_PARAMS["clip_min"],
        upper=LOG_TRANSFORM_PARAMS["clip_max"]
    ))
    
    # Run KS tests on numerical columns
    numerical_cols = ["amount"]
    ks_results = {}
    
    for col in numerical_cols:
        if col in train_df.columns and col in synthetic_df.columns:
            train_values = train_df[col].dropna().values
            synth_values = synthetic_df[col].dropna().values
            
            if len(train_values) > 0 and len(synth_values) > 0:
                statistic, p_value = stats.ks_2samp(train_values, synth_values)
                ks_results[col] = {
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                }
                logger.info(f"KS test for {col}: statistic={statistic:.4f}, p_value={p_value:.4f}")
    
    # Compare categorical distributions
    categorical_cols = ["tier", "archetype", "transaction_type", "direction"]
    categorical_comparison = {}
    
    for col in categorical_cols:
        if col in train_df.columns and col in synthetic_df.columns:
            train_dist = train_df[col].value_counts(normalize=True).to_dict()
            synth_dist = synthetic_df[col].value_counts(normalize=True).to_dict()
            
            categorical_comparison[col] = {
                "train_distribution": train_dist,
                "synthetic_distribution": synth_dist,
            }
    
    validation_results = {
        "ks_tests": ks_results,
        "categorical_comparison": categorical_comparison,
        "sample_size": sample_size,
    }
    
    return validation_results


def save_training_report(
    report: Dict[str, Any],
    output_path: Path,
) -> None:
    """Save training report to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Training report saved to: {output_path}")


def main() -> None:
    args = parse_args()
    
    # Generate timestamp for outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("Starting TVAE training on Monte Carlo baseline")
    logger.info(f"Configuration:")
    logger.info(f"  Partition key: {args.partition_key}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Sample size for validation: {args.sample_size}")
    
    # Read baseline data
    input_path = Path(args.input_dir) / f"monte_carlo_baseline_{args.partition_key}.parquet"
    logger.info(f"Reading baseline data from: {input_path}")
    
    if not input_path.exists():
        raise FileNotFoundError(f"Baseline data not found: {input_path}")
    
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} records")
    
    # Preprocess data
    train_df = preprocess_data(df)
    
    # Initialize TVAE generator
    logger.info("Initializing TVAE generator...")
    generator = TVAEGenerator(epochs=args.epochs, batch_size=args.batch_size)
    
    # Train model
    logger.info("Training TVAE model...")
    start_time = datetime.now()
    generator.fit(train_df)
    training_duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"Training completed in {training_duration:.2f} seconds")
    
    # Save trained model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"tvae_model_{timestamp}.pkl"
    
    with open(model_path, "wb") as f:
        pickle.dump(generator, f)
    
    logger.info(f"Trained model saved to: {model_path}")
    
    # Validate model
    validation_results = validate_model(generator, train_df, args.sample_size)
    
    # Compile training report
    report = {
        "timestamp": timestamp,
        "training_config": {
            "partition_key": args.partition_key,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "training_records": len(train_df),
            "training_duration_seconds": training_duration,
        },
        "log_transform_params": LOG_TRANSFORM_PARAMS,
        "validation": validation_results,
        "model_path": str(model_path),
        "input_data_path": str(input_path),
    }
    
    # Save training report
    report_dir = Path(args.report_dir)
    report_path = report_dir / f"tvae_training_{timestamp}.json"
    save_training_report(report, report_path)
    
    logger.info("TVAE training pipeline complete")


if __name__ == "__main__":
    main()
