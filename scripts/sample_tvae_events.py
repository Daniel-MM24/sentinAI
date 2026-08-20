"""Sample synthetic events from trained TVAE model.

This script loads a trained TVAE model, samples synthetic events,
post-processes them, and outputs raw events to parquet.
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.tvae_generator import TVAEGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Log-transform parameters from stat_distribution.md
LOG_TRANSFORM_PARAMS = {
    "amount_mean": 6.02,
    "amount_std": 1.25,
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
        description="Sample synthetic events from trained TVAE model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained TVAE model (e.g., models/tvae_model_20240101_120000.pkl)",
    )
    parser.add_argument(
        "--partition",
        type=str,
        required=True,
        help="Partition identifier for output (e.g., 2026-08-05)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50000,
        help="Number of synthetic events to sample (default: 50000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for raw events (default: data)",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="reports",
        help="Output directory for sampling report (default: reports)",
    )
    return parser.parse_args()


def load_model(model_path: Path) -> TVAEGenerator:
    """Load trained TVAE model from pickle file."""
    logger.info(f"Loading TVAE model from: {model_path}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, "rb") as f:
        generator = pickle.load(f)
    
    logger.info("Model loaded successfully")
    return generator


def inverse_log_transform_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Apply inverse log-transform to amount column."""
    df = df.copy()
    
    # Inverse log-transform: exp(amount) - 1
    df["amount"] = np.expm1(df["amount"])
    
    # Clip to valid range
    df["amount"] = df["amount"].clip(
        lower=LOG_TRANSFORM_PARAMS["clip_min"],
        upper=LOG_TRANSFORM_PARAMS["clip_max"]
    )
    
    # Round to 2 decimal places
    df["amount"] = df["amount"].round(2)
    
    logger.info("Applied inverse log-transform to amount column")
    logger.info(f"  Inverse-transformed range: [{df['amount'].min():.2f}, {df['amount'].max():.2f}]")
    
    return df


def ensure_categorical_validity(
    df: pd.DataFrame,
    training_data: pd.DataFrame,
) -> pd.DataFrame:
    """Ensure categorical columns have valid values from training data."""
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
        if col in df.columns and col in training_data.columns:
            # Get valid values from training data
            valid_values = set(training_data[col].astype(str).unique())
            
            # Clip invalid values to nearest valid value or mode
            invalid_mask = ~df[col].astype(str).isin(valid_values)
            if invalid_mask.any():
                mode_value = training_data[col].mode()[0]
                df.loc[invalid_mask, col] = mode_value
                logger.warning(f"Clipped {invalid_mask.sum()} invalid values in {col} to mode: {mode_value}")
    
    return df


def post_process_samples(
    synthetic_df: pd.DataFrame,
    training_data: pd.DataFrame,
) -> pd.DataFrame:
    """Post-process synthetic samples."""
    logger.info("Post-processing synthetic samples...")
    
    # Apply inverse log-transform to amount
    synthetic_df = inverse_log_transform_amount(synthetic_df)
    
    # Ensure categorical validity
    synthetic_df = ensure_categorical_validity(synthetic_df, training_data)
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(synthetic_df["timestamp"]):
        synthetic_df["timestamp"] = pd.to_datetime(synthetic_df["timestamp"])
    
    logger.info(f"Post-processing complete. Shape: {synthetic_df.shape}")
    return synthetic_df


def validate_core_columns(df: pd.DataFrame) -> None:
    """Validate that all 8 core columns are present and properly typed."""
    logger.info("Validating core columns...")
    
    missing_cols = set(CORE_TVAE_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing core columns: {missing_cols}")
    
    # Validate column types
    expected_types = {
        "customer_id": "object",
        "tier": "object",
        "archetype": "object",
        "transaction_type": "object",
        "amount": "float64",
        "timestamp": "datetime64[ns]",
        "direction": "object",
        "is_international": "object",
    }
    
    for col, expected_type in expected_types.items():
        if col in df.columns:
            if expected_type == "datetime64[ns]":
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    raise ValueError(f"Column {col} is not datetime type")
            elif expected_type == "object":
                if df[col].dtype != "object":
                    logger.warning(f"Column {col} has dtype {df[col].dtype}, expected object")
            else:
                if df[col].dtype != expected_type:
                    logger.warning(f"Column {col} has dtype {df[col].dtype}, expected {expected_type}")
    
    logger.info("Core columns validation passed")


def compute_sampling_statistics(
    synthetic_df: pd.DataFrame,
    training_data: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute sampling statistics comparing to training distributions."""
    logger.info("Computing sampling statistics...")
    
    stats = {
        "sample_size": len(synthetic_df),
        "training_size": len(training_data),
    }
    
    # Numerical column statistics
    numerical_cols = ["amount"]
    for col in numerical_cols:
        if col in synthetic_df.columns and col in training_data.columns:
            stats[col] = {
                "synthetic": {
                    "mean": float(synthetic_df[col].mean()),
                    "std": float(synthetic_df[col].std()),
                    "min": float(synthetic_df[col].min()),
                    "max": float(synthetic_df[col].max()),
                },
                "training": {
                    "mean": float(training_data[col].mean()),
                    "std": float(training_data[col].std()),
                    "min": float(training_data[col].min()),
                    "max": float(training_data[col].max()),
                },
            }
    
    # Categorical column distributions
    categorical_cols = ["tier", "archetype", "transaction_type", "direction"]
    for col in categorical_cols:
        if col in synthetic_df.columns and col in training_data.columns:
            synth_dist = synthetic_df[col].value_counts(normalize=True).to_dict()
            train_dist = training_data[col].value_counts(normalize=True).to_dict()
            stats[col] = {
                "synthetic_distribution": synth_dist,
                "training_distribution": train_dist,
            }
    
    # Timestamp range
    if "timestamp" in synthetic_df.columns and "timestamp" in training_data.columns:
        stats["timestamp"] = {
            "synthetic": {
                "min": str(synthetic_df["timestamp"].min()),
                "max": str(synthetic_df["timestamp"].max()),
            },
            "training": {
                "min": str(training_data["timestamp"].min()),
                "max": str(training_data["timestamp"].max()),
            },
        }
    
    logger.info("Sampling statistics computed")
    return stats


def save_sampling_report(
    report: Dict[str, Any],
    output_path: Path,
) -> None:
    """Save sampling report to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Sampling report saved to: {output_path}")


def main() -> None:
    args = parse_args()
    
    logger.info("Starting TVAE sampling")
    logger.info(f"Configuration:")
    logger.info(f"  Model path: {args.model_path}")
    logger.info(f"  Partition: {args.partition}")
    logger.info(f"  Number of samples: {args.n_samples}")
    
    # Load trained model
    model_path = Path(args.model_path)
    generator = load_model(model_path)
    
    # Load training data for reference (needed for categorical validity)
    # Try to extract training data path from model metadata if available
    training_data_path = None
    if hasattr(generator, 'training_data_path'):
        training_data_path = Path(generator.training_data_path)
    
    if training_data_path and training_data_path.exists():
        logger.info(f"Loading training data from: {training_data_path}")
        training_data = pd.read_parquet(training_data_path)
    else:
        logger.warning("Training data not found, skipping categorical validity checks")
        training_data = None
    
    # Sample synthetic events
    logger.info(f"Sampling {args.n_samples} synthetic events...")
    synthetic_df = generator.sample(args.n_samples)
    logger.info(f"Sampled {len(synthetic_df)} events")
    
    # Post-process samples
    if training_data is not None:
        synthetic_df = post_process_samples(synthetic_df, training_data)
    else:
        # Apply basic post-processing without training data reference
        synthetic_df = inverse_log_transform_amount(synthetic_df)
        if not pd.api.types.is_datetime64_any_dtype(synthetic_df["timestamp"]):
            synthetic_df["timestamp"] = pd.to_datetime(synthetic_df["timestamp"])
    
    # Validate core columns
    validate_core_columns(synthetic_df)
    
    # Compute sampling statistics
    if training_data is not None:
        sampling_stats = compute_sampling_statistics(synthetic_df, training_data)
    else:
        sampling_stats = {"sample_size": len(synthetic_df)}
    
    # Save raw events
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"tvae_raw_events_{args.partition}.parquet"
    
    synthetic_df.to_parquet(output_path, index=False)
    logger.info(f"Raw events saved to: {output_path}")
    
    # Compile and save sampling report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": timestamp,
        "sampling_config": {
            "model_path": str(model_path),
            "partition": args.partition,
            "n_samples": args.n_samples,
        },
        "statistics": sampling_stats,
        "output_path": str(output_path),
    }
    
    report_dir = Path(args.report_dir)
    report_path = report_dir / f"tvae_sampling_{timestamp}.json"
    save_sampling_report(report, report_path)
    
    logger.info("TVAE sampling pipeline complete")


if __name__ == "__main__":
    main()
