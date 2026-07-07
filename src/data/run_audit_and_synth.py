import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import polars as pl

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.anomaly_injector import FinancialAnomalyInjector, InjectorConfig
from src.data.lineage_decorator import emit_transformation_metadata, lineage_trace
from src.data.synthetic_generator import CleanDataGenerator, GeneratorConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@lineage_trace(
    job_name="generate_clean_synthetic_dataset",
    input_datasets=["synthetic_generation_config"],
    output_datasets=["synthetic_clean.parquet"],
    namespace="sentinai.data",
)
def generate_clean_data(
    generator: CleanDataGenerator,
    output_path: Path,
    run_id: Optional[str] = None,
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

    logger.info(f"Generated {len(clean_df)} clean records")
    logger.info(f"Shape: {clean_df.shape}")
    logger.info(f"Columns: {clean_df.columns}")
    logger.info(f"Clean data saved to {output_path}")
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
    run_id: Optional[str] = None,
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

    logger.info(f"Anomalies injected. Total records: {len(anomalous_df)}")
    logger.info(f"Anomaly ratio: {anomalous_df['anomaly_flag'].mean():.2%}")
    logger.info(f"Anomalous data saved to {output_path}")
    return anomalous_df


@lineage_trace(
    job_name="run_audit_and_synth_pipeline",
    input_datasets=["synthetic_generation_config"],
    output_datasets=["synthetic_clean.parquet", "synthetic_anomalous.parquet", "pipeline_summary.json"],
    namespace="sentinai.data",
)
def run_pipeline(
    config: Optional[GeneratorConfig] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the synthetic data generation and anomaly injection pipeline end to end."""
    logger.info("Starting two-stage synthetic data pipeline")

    if config is None:
        config = GeneratorConfig(num_records=1_000_000, num_entities=100_000, seed=42)

    output_root = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent.parent.parent / "data"
    output_root.mkdir(parents=True, exist_ok=True)

    clean_output_path = output_root / "synthetic_clean.parquet"
    anomalous_output_path = output_root / "synthetic_anomalous.parquet"
    summary_path = output_root / f"pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    generator = CleanDataGenerator(config)
    clean_df = generate_clean_data(generator, clean_output_path)

    injector_config = InjectorConfig(anomaly_ratio=0.015, seed=42)
    injector = FinancialAnomalyInjector(injector_config)
    anomalous_df = inject_anomalies(injector, clean_df, anomalous_output_path)

    summary = injector.get_anomaly_summary(anomalous_df)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logger.info(f"Pipeline complete. Anomaly summary: {summary}")
    logger.info(f"Summary saved to {summary_path}")

    return {
        "clean_output_path": clean_output_path,
        "anomalous_output_path": anomalous_output_path,
        "summary_path": summary_path,
        "summary": summary,
    }


def main() -> None:
    """CLI entrypoint for the lineage-aware synthetic audit/synthesis pipeline."""
    run_pipeline()


if __name__ == "__main__":
    main()
