"""TVAE Hybrid Pipeline Orchestrator.

This script orchestrates the full TVAE hybrid pipeline for synthetic AML data generation:
- Phase 1: Monte Carlo baseline generation (if not exists)
- Phase 2: TVAE training (if model not exists)
- Phase 3: TVAE sampling
- Phase 4: Balance reconstruction
- Phase 5: Feature engineering
- Phase 6: Anomaly injection

The script implements checkpointing to skip completed phases, logs progress with timing,
and outputs a comprehensive pipeline report with schema validation and statistics.
"""

import argparse
import json
import logging
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time

import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.hybrid_reconstructor import BalanceReconstructor
from src.data.feature_engineering import CustomerFeatureEngineer
from src.data.anomaly_injector import FinancialAnomalyInjector, TVAEInjectorConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Target 21-feature gold schema
GOLD_SCHEMA_COLUMNS = [
    "customer_id",
    "tier",
    "archetype",
    "transaction_type",
    "amount",
    "timestamp",
    "direction",
    "balance",
    "tx_count_7d",
    "volume_7d",
    "night_tx_ratio",
    "rapid_tx_ratio",
    "volume_7d_vs_30d_ratio",
    "is_international",
    "distinct_counterparties_7d",
    "fan_in_fan_out_ratio",
    "close_to_limit_ratio",
    "balance_retention_ratio",
    "amount_roundness",
    "is_launderer",
    "aml_scenario",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orchestrate TVAE hybrid pipeline for synthetic AML data generation"
    )
    parser.add_argument(
        "--partition",
        type=str,
        required=True,
        help="Partition key for data versioning (e.g., 2026-08-05)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50000,
        help="Number of TVAE samples to generate (default: 50000)",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip Monte Carlo generation if baseline exists",
    )
    parser.add_argument(
        "--retrain-tvae",
        action="store_true",
        help="Force TVAE retraining even if model exists",
    )
    parser.add_argument(
        "--anomaly-ratio",
        type=float,
        default=0.015,
        help="Percentage of customers to flag as launderers (default: 0.015 = 1.5%%)",
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default="data/bronze",
        help="Directory for baseline data (default: data/bronze)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory for TVAE models (default: models)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory for intermediate data (default: data)",
    )
    parser.add_argument(
        "--gold-dir",
        type=str,
        default="data/gold",
        help="Directory for final gold output (default: data/gold)",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="reports",
        help="Directory for pipeline reports (default: reports)",
    )
    parser.add_argument(
        "--num-customers",
        type=int,
        default=10000,
        help="Number of customers for Monte Carlo baseline (default: 10000)",
    )
    parser.add_argument(
        "--num-transactions",
        type=int,
        default=100000,
        help="Number of transactions for Monte Carlo baseline (default: 100000)",
    )
    parser.add_argument(
        "--tvae-epochs",
        type=int,
        default=300,
        help="TVAE training epochs (default: 300)",
    )
    parser.add_argument(
        "--tvae-batch-size",
        type=int,
        default=500,
        help="TVAE training batch size (default: 500)",
    )
    return parser.parse_args()


class PipelinePhase:
    """Represents a pipeline phase with checkpointing and timing."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration: Optional[float] = None
        self.status = "pending"
        self.output_path: Optional[str] = None

    def start(self) -> None:
        self.start_time = time.time()
        self.status = "running"
        logger.info(f"{'=' * 60}")
        logger.info(f"Starting Phase: {self.name}")
        logger.info(f"Description: {self.description}")
        logger.info(f"{'=' * 60}")

    def complete(self, output_path: Optional[str] = None) -> None:
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = "completed"
        self.output_path = output_path
        logger.info(f"Phase {self.name} completed in {self.duration:.2f}s")
        if output_path:
            logger.info(f"Output: {output_path}")

    def skip(self, reason: str) -> None:
        self.status = "skipped"
        self.end_time = time.time()
        self.duration = 0.0
        logger.info(f"Phase {self.name} skipped: {reason}")

    def fail(self, error: str) -> None:
        self.end_time = time.time()
        self.duration = 0.0
        self.status = "failed"
        logger.error(f"Phase {self.name} failed: {error}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "duration_seconds": self.duration,
            "output_path": self.output_path,
        }


class TVAEHybridPipeline:
    """Orchestrates the full TVAE hybrid pipeline."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.phases: Dict[str, PipelinePhase] = {}
        self.pipeline_start_time = time.time()
        self.report: Dict[str, Any] = {
            "partition": args.partition,
            "timestamp": datetime.now().isoformat(),
            "configuration": vars(args),
            "phases": {},
        }

        # Create directories
        Path(args.baseline_dir).mkdir(parents=True, exist_ok=True)
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)
        Path(args.data_dir).mkdir(parents=True, exist_ok=True)
        Path(args.gold_dir).mkdir(parents=True, exist_ok=True)
        Path(args.report_dir).mkdir(parents=True, exist_ok=True)

    def _get_phase(self, name: str, description: str) -> PipelinePhase:
        if name not in self.phases:
            self.phases[name] = PipelinePhase(name, description)
        return self.phases[name]

    def phase_1_monte_carlo_baseline(self) -> None:
        """Phase 1: Generate Monte Carlo baseline data."""
        phase = self._get_phase(
            "phase_1_monte_carlo_baseline",
            "Generate Monte Carlo baseline data for TVAE training"
        )
        phase.start()

        baseline_path = Path(self.args.baseline_dir) / f"monte_carlo_baseline_{self.args.partition}.parquet"

        # Check if baseline exists
        if baseline_path.exists() and self.args.skip_baseline:
            phase.skip(f"Baseline already exists at {baseline_path}")
            self.report["phases"][phase.name] = phase.to_dict()
            return

        # Run Monte Carlo baseline generation
        cmd = [
            sys.executable,
            "scripts/run_monte_carlo_baseline.py",
            "--num-customers", str(self.args.num_customers),
            "--num-transactions", str(self.args.num_transactions),
            "--partition-key", self.args.partition,
            "--output-dir", self.args.baseline_dir,
        ]

        logger.info(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(result.stdout)
            if result.stderr:
                logger.warning(result.stderr)
        except subprocess.CalledProcessError as e:
            phase.fail(f"Monte Carlo baseline generation failed: {e.stderr}")
            self.report["phases"][phase.name] = phase.to_dict()
            raise

        if not baseline_path.exists():
            phase.fail(f"Baseline file not created at {baseline_path}")
            self.report["phases"][phase.name] = phase.to_dict()
            raise FileNotFoundError(f"Baseline file not found: {baseline_path}")

        phase.complete(str(baseline_path))
        self.report["phases"][phase.name] = phase.to_dict()

    def phase_2_tvae_training(self) -> None:
        """Phase 2: Train TVAE model on baseline data."""
        phase = self._get_phase(
            "phase_2_tvae_training",
            "Train TVAE model on Monte Carlo baseline data"
        )
        phase.start()

        baseline_path = Path(self.args.baseline_dir) / f"monte_carlo_baseline_{self.args.partition}.parquet"

        if not baseline_path.exists():
            phase.fail(f"Baseline data not found at {baseline_path}")
            self.report["phases"][phase.name] = phase.to_dict()
            raise FileNotFoundError(f"Baseline data not found: {baseline_path}")

        # Find existing model or create new path
        model_pattern = f"tvae_model_*.pkl"
        existing_models = list(Path(self.args.model_dir).glob(model_pattern))

        if existing_models and not self.args.retrain_tvae:
            model_path = existing_models[-1]  # Use most recent model
            phase.skip(f"Using existing model at {model_path}")
            self.report["phases"][phase.name] = phase.to_dict()
            self.report["tvae_model_path"] = str(model_path)
            return

        # Train new model
        cmd = [
            sys.executable,
            "scripts/train_tvae_on_baseline.py",
            "--partition-key", self.args.partition,
            "--input-dir", self.args.baseline_dir,
            "--epochs", str(self.args.tvae_epochs),
            "--batch-size", str(self.args.tvae_batch_size),
            "--output-dir", self.args.model_dir,
            "--report-dir", self.args.report_dir,
        ]

        logger.info(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(result.stdout)
            if result.stderr:
                logger.warning(result.stderr)
        except subprocess.CalledProcessError as e:
            phase.fail(f"TVAE training failed: {e.stderr}")
            self.report["phases"][phase.name] = phase.to_dict()
            raise

        # Find newly created model
        new_models = list(Path(self.args.model_dir).glob(model_pattern))
        if not new_models:
            phase.fail("No model file created after training")
            self.report["phases"][phase.name] = phase.to_dict()
            raise FileNotFoundError("TVAE model not found after training")

        model_path = new_models[-1]
        self.report["tvae_model_path"] = str(model_path)
        phase.complete(str(model_path))
        self.report["phases"][phase.name] = phase.to_dict()

    def phase_3_tvae_sampling(self) -> None:
        """Phase 3: Sample synthetic events from trained TVAE model."""
        phase = self._get_phase(
            "phase_3_tvae_sampling",
            "Sample synthetic events from trained TVAE model"
        )
        phase.start()

        model_path = self.report.get("tvae_model_path")
        if not model_path:
            phase.fail("TVAE model path not found in report")
            self.report["phases"][phase.name] = phase.to_dict()
            raise ValueError("TVAE model path not available")

        raw_events_path = Path(self.args.data_dir) / f"tvae_raw_events_{self.args.partition}.parquet"

        # Check if raw events already exist
        if raw_events_path.exists():
            phase.skip(f"Raw events already exist at {raw_events_path}")
            self.report["phases"][phase.name] = phase.to_dict()
            return

        cmd = [
            sys.executable,
            "scripts/sample_tvae_events.py",
            "--model-path", model_path,
            "--partition", self.args.partition,
            "--n-samples", str(self.args.n_samples),
            "--output-dir", self.args.data_dir,
            "--report-dir", self.args.report_dir,
        ]

        logger.info(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(result.stdout)
            if result.stderr:
                logger.warning(result.stderr)
        except subprocess.CalledProcessError as e:
            phase.fail(f"TVAE sampling failed: {e.stderr}")
            self.report["phases"][phase.name] = phase.to_dict()
            raise

        if not raw_events_path.exists():
            phase.fail(f"Raw events file not created at {raw_events_path}")
            self.report["phases"][phase.name] = phase.to_dict()
            raise FileNotFoundError(f"Raw events not found: {raw_events_path}")

        phase.complete(str(raw_events_path))
        self.report["phases"][phase.name] = phase.to_dict()

    def phase_4_balance_reconstruction(self) -> None:
        """Phase 4: Reconstruct deterministic balances from TVAE events."""
        phase = self._get_phase(
            "phase_4_balance_reconstruction",
            "Reconstruct deterministic balances and enforce tier caps"
        )
        phase.start()

        raw_events_path = Path(self.args.data_dir) / f"tvae_raw_events_{self.args.partition}.parquet"
        balance_corrected_path = Path(self.args.data_dir) / f"tvae_balance_corrected_{self.args.partition}.parquet"

        if not raw_events_path.exists():
            phase.fail(f"Raw events not found at {raw_events_path}")
            self.report["phases"][phase.name] = phase.to_dict()
            raise FileNotFoundError(f"Raw events not found: {raw_events_path}")

        # Check if balance corrected already exists
        if balance_corrected_path.exists():
            phase.skip(f"Balance corrected data already exists at {balance_corrected_path}")
            self.report["phases"][phase.name] = phase.to_dict()
            return

        # Load raw events
        logger.info(f"Loading raw events from {raw_events_path}")
        raw_df = pd.read_parquet(raw_events_path)
        logger.info(f"Loaded {len(raw_df)} raw events")

        # Initialize reconstructor
        reconstructor = BalanceReconstructor(output_dir=self.args.data_dir)

        # Reconstruct balances
        logger.info("Reconstructing balances...")
        balance_df = reconstructor.reconstruct(raw_df, partition=self.args.partition)

        phase.complete(str(balance_corrected_path))
        self.report["phases"][phase.name] = phase.to_dict()

    def phase_5_feature_engineering(self) -> None:
        """Phase 5: Compute enriched features from balance-corrected events."""
        phase = self._get_phase(
            "phase_5_feature_engineering",
            "Compute temporal and behavioral features"
        )
        phase.start()

        balance_corrected_path = Path(self.args.data_dir) / f"tvae_balance_corrected_{self.args.partition}.parquet"
        enriched_path = Path(self.args.data_dir) / f"tvae_enriched_{self.args.partition}.parquet"

        if not balance_corrected_path.exists():
            phase.fail(f"Balance corrected data not found at {balance_corrected_path}")
            self.report["phases"][phase.name] = phase.to_dict()
            raise FileNotFoundError(f"Balance corrected data not found: {balance_corrected_path}")

        # Check if enriched data already exists
        if enriched_path.exists():
            phase.skip(f"Enriched data already exists at {enriched_path}")
            self.report["phases"][phase.name] = phase.to_dict()
            return

        # Load balance corrected data
        logger.info(f"Loading balance corrected data from {balance_corrected_path}")
        balance_df = pl.read_parquet(balance_corrected_path)
        logger.info(f"Loaded {len(balance_df)} balance-corrected events")

        # Initialize feature engineer
        engineer = CustomerFeatureEngineer()

        # Compute features
        logger.info("Computing features...")
        enriched_df = engineer.compute_features(
            balance_df,
            partition=self.args.partition,
            output_dir=self.args.data_dir
        )

        phase.complete(str(enriched_path))
        self.report["phases"][phase.name] = phase.to_dict()

    def phase_6_anomaly_injection(self) -> None:
        """Phase 6: Inject AML scenarios into enriched data."""
        phase = self._get_phase(
            "phase_6_anomaly_injection",
            "Inject deterministic AML scenarios with labels"
        )
        phase.start()

        enriched_path = Path(self.args.data_dir) / f"tvae_enriched_{self.args.partition}.parquet"
        gold_path = Path(self.args.gold_dir) / f"tvae_hybrid_gold_{self.args.partition}.parquet"

        if not enriched_path.exists():
            phase.fail(f"Enriched data not found at {enriched_path}")
            self.report["phases"][phase.name] = phase.to_dict()
            raise FileNotFoundError(f"Enriched data not found: {enriched_path}")

        # Check if gold data already exists
        if gold_path.exists():
            phase.skip(f"Gold data already exists at {gold_path}")
            self.report["phases"][phase.name] = phase.to_dict()
            return

        # Load enriched data
        logger.info(f"Loading enriched data from {enriched_path}")
        enriched_df = pl.read_parquet(enriched_path)
        logger.info(f"Loaded {len(enriched_df)} enriched events")

        # Initialize injector with configured anomaly ratio
        config = TVAEInjectorConfig(launderer_fraction=self.args.anomaly_ratio)
        injector = FinancialAnomalyInjector(config)

        # Inject anomalies
        logger.info(f"Injecting anomalies with ratio {self.args.anomaly_ratio}")
        gold_df = injector.inject(
            enriched_df,
            partition=self.args.partition,
            output_dir=self.args.gold_dir
        )

        phase.complete(str(gold_path))
        self.report["phases"][phase.name] = phase.to_dict()

    def validate_final_schema(self) -> None:
        """Validate final output schema matches 21-feature target."""
        logger.info("Validating final schema...")
        gold_path = Path(self.args.gold_dir) / f"tvae_hybrid_gold_{self.args.partition}.parquet"

        if not gold_path.exists():
            raise FileNotFoundError(f"Gold data not found: {gold_path}")

        gold_df = pl.read_parquet(gold_path)
        actual_columns = set(gold_df.columns)
        expected_columns = set(GOLD_SCHEMA_COLUMNS)

        missing = expected_columns - actual_columns
        extra = actual_columns - expected_columns

        validation_result = {
            "total_rows": len(gold_df),
            "total_columns": len(gold_df.columns),
            "expected_columns": len(expected_columns),
            "actual_columns": list(actual_columns),
            "missing_columns": list(missing),
            "extra_columns": list(extra),
            "schema_valid": len(missing) == 0,
        }

        self.report["schema_validation"] = validation_result

        if missing:
            logger.error(f"Missing columns: {missing}")
            raise ValueError(f"Schema validation failed: missing columns {missing}")

        if extra:
            logger.warning(f"Extra columns present: {extra}")

        logger.info("Schema validation passed")
        logger.info(f"Total rows: {validation_result['total_rows']}")
        logger.info(f"Total columns: {validation_result['total_columns']}")

    def compute_summary_statistics(self) -> None:
        """Compute summary statistics comparing TVAE hybrid vs pure Monte Carlo."""
        logger.info("Computing summary statistics...")

        gold_path = Path(self.args.gold_dir) / f"tvae_hybrid_gold_{self.args.partition}.parquet"
        baseline_path = Path(self.args.baseline_dir) / f"monte_carlo_baseline_{self.args.partition}.parquet"

        if not gold_path.exists():
            logger.warning("Gold data not found, skipping statistics comparison")
            return

        gold_df = pl.read_parquet(gold_path)

        stats = {
            "tvae_hybrid": {
                "total_rows": len(gold_df),
                "unique_customers": gold_df["customer_id"].n_unique(),
                "launderer_count": gold_df.filter(pl.col("is_launderer")).height,
                "launderer_ratio": float(gold_df.filter(pl.col("is_launderer")).height / len(gold_df)),
                "scenario_distribution": gold_df.group_by("aml_scenario").len().to_dict(as_series=False),
            },
            "feature_statistics": {},
        }

        # Compute feature statistics for key columns
        numerical_features = [
            "amount", "balance", "tx_count_7d", "volume_7d",
            "night_tx_ratio", "rapid_tx_ratio", "volume_7d_vs_30d_ratio",
            "distinct_counterparties_7d", "fan_in_fan_out_ratio",
            "close_to_limit_ratio", "balance_retention_ratio", "amount_roundness"
        ]

        for feature in numerical_features:
            if feature in gold_df.columns:
                col_data = gold_df[feature]
                stats["feature_statistics"][feature] = {
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "median": float(col_data.median()),
                }

        # Compare with baseline if available
        if baseline_path.exists():
            baseline_df = pl.read_parquet(baseline_path)
            stats["monte_carlo_baseline"] = {
                "total_rows": len(baseline_df),
                "unique_customers": baseline_df["customer_id"].n_unique(),
            }

            # Compare amount distributions
            if "amount" in baseline_df.columns and "amount" in gold_df.columns:
                baseline_amount = baseline_df["amount"]
                gold_amount = gold_df["amount"]
                stats["amount_comparison"] = {
                    "baseline_mean": float(baseline_amount.mean()),
                    "gold_mean": float(gold_amount.mean()),
                    "baseline_std": float(baseline_amount.std()),
                    "gold_std": float(gold_amount.std()),
                }

        self.report["summary_statistics"] = stats
        logger.info("Summary statistics computed")

    def save_report(self) -> None:
        """Save pipeline report to JSON file."""
        report_path = Path(self.args.report_dir) / f"tvae_pipeline_{self.args.partition}.json"

        # Add pipeline duration
        pipeline_duration = time.time() - self.pipeline_start_time
        self.report["pipeline_duration_seconds"] = pipeline_duration

        # Add phase summary
        completed_phases = sum(1 for p in self.phases.values() if p.status == "completed")
        skipped_phases = sum(1 for p in self.phases.values() if p.status == "skipped")
        failed_phases = sum(1 for p in self.phases.values() if p.status == "failed")

        self.report["phase_summary"] = {
            "total": len(self.phases),
            "completed": completed_phases,
            "skipped": skipped_phases,
            "failed": failed_phases,
        }

        with open(report_path, "w") as f:
            json.dump(self.report, f, indent=2, default=str)

        logger.info(f"Pipeline report saved to {report_path}")
        logger.info(f"Total pipeline duration: {pipeline_duration:.2f}s")

    def run(self) -> None:
        """Run the complete pipeline."""
        logger.info("=" * 60)
        logger.info("TVAE Hybrid Pipeline Starting")
        logger.info(f"Partition: {self.args.partition}")
        logger.info(f"Samples: {self.args.n_samples}")
        logger.info(f"Anomaly ratio: {self.args.anomaly_ratio}")
        logger.info("=" * 60)

        try:
            self.phase_1_monte_carlo_baseline()
            self.phase_2_tvae_training()
            self.phase_3_tvae_sampling()
            self.phase_4_balance_reconstruction()
            self.phase_5_feature_engineering()
            self.phase_6_anomaly_injection()
            self.validate_final_schema()
            self.compute_summary_statistics()
            self.save_report()

            logger.info("=" * 60)
            logger.info("TVAE Hybrid Pipeline Completed Successfully")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.save_report()
            raise


def main() -> None:
    args = parse_args()
    pipeline = TVAEHybridPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
