"""Synthetic clean-data generation and anomaly injection pipeline (library only)."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from src.data.anomaly_injector import FinancialAnomalyInjector, InjectorConfig
from src.data.lineage_decorator import emit_transformation_metadata, lineage_trace
from src.data.synthetic_generator import CleanDataGenerator, GeneratorConfig

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@lineage_trace(
    job_name="generate_clean_synthetic_dataset",
    input_datasets=["synthetic_generation_config"],
    output_datasets=["synthetic_clean.parquet"],
    namespace="sentinai.data",
)
def generate_clean_data(
    generator: CleanDataGenerator,
    output_path: Path,
    run_id: str | None = None,
) -> pl.DataFrame:
    """Generate and persist the clean synthetic dataset with lineage metadata."""
    clean_df = generator.generate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.write_parquet(output_path)

    emit_transformation_metadata(
        job_name="generate_clean_synthetic_dataset",
        run_id=run_id or str(uuid.uuid4()),
        transformation_python="Generate clean synthetic financial rows",
        input_rows=len(clean_df),
        output_rows=len(clean_df),
    )

    logger.info("Generated %s clean records at %s", len(clean_df), output_path)
    return clean_df


@lineage_trace(
    job_name="inject_anomalies_into_synthetic_dataset",
    input_datasets=["synthetic_clean.parquet"],
    output_datasets=["synthetic_anomalous.parquet"],
    namespace="sentinai.data",
)
def inject_anomalies(
    injector: FinancialAnomalyInjector,
    clean_df: pl.DataFrame,
    output_path: Path,
    run_id: str | None = None,
) -> pl.DataFrame:
    """Inject anomalies into the clean dataset and persist the result."""
    anomalous_df = injector.inject(clean_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anomalous_df.write_parquet(output_path)

    emit_transformation_metadata(
        job_name="inject_anomalies_into_synthetic_dataset",
        run_id=run_id or str(uuid.uuid4()),
        transformation_python="Inject anomaly flags and anomaly type labels into synthetic data",
        input_rows=len(clean_df),
        output_rows=len(anomalous_df),
    )

    logger.info(
        "Anomalies injected: %s records, ratio=%.2f%%",
        len(anomalous_df),
        anomalous_df["anomaly_flag"].mean() * 100,
    )
    return anomalous_df


@lineage_trace(
    job_name="run_audit_and_synth_pipeline",
    input_datasets=["synthetic_generation_config"],
    output_datasets=[
        "synthetic_clean.parquet",
        "synthetic_anomalous.parquet",
        "pipeline_summary.json",
    ],
    namespace="sentinai.data",
)
def run_pipeline(
    config: GeneratorConfig | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run clean synthetic generation and anomaly injection end to end."""
    logger.info("Starting two-stage synthetic data pipeline")

    if config is None:
        config = GeneratorConfig(num_records=1_000_000, num_entities=100_000, seed=42)

    output_root = output_dir or (PROJECT_ROOT / "data")
    output_root.mkdir(parents=True, exist_ok=True)

    clean_output_path = output_root / "synthetic_clean.parquet"
    anomalous_output_path = output_root / "synthetic_anomalous.parquet"
    summary_path = output_root / f"pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    generator = CleanDataGenerator(config)
    clean_df = generate_clean_data(generator, clean_output_path)

    injector = FinancialAnomalyInjector(InjectorConfig(anomaly_ratio=0.015, seed=42))
    anomalous_df = inject_anomalies(injector, clean_df, anomalous_output_path)

    summary = injector.get_anomaly_summary(anomalous_df)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("Pipeline complete. Summary saved to %s", summary_path)

    return {
        "clean_output_path": clean_output_path,
        "anomalous_output_path": anomalous_output_path,
        "summary_path": summary_path,
        "summary": summary,
    }
