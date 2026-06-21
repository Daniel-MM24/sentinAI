"""
Bronze Layer Module for Medallion Architecture

This module implements the Bronze layer transformation that ingests raw, immutable logs
from PostgreSQL into Parquet/Delta format. All transformations are instrumented with
the lineage_trace decorator for MRM compliance and OpenLineage integration.

MRM Compliance: Bronze layer is immutable - once written, data is never modified.
"""

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.lineage_decorator import lineage_trace, emit_transformation_metadata
from src.data.schemas import BronzeTransactionRecord, DataSourceType

logger = logging.getLogger(__name__)


class BronzeLayer:
    """
    BronzeLayer orchestrates the ingestion of raw data from PostgreSQL into immutable
    Parquet/Delta format for the Bronze layer of the Medallion architecture.
    """

    def __init__(
        self,
        bronze_base_path: str = "data/bronze",
        namespace: str = "sentinai.bronze",
    ):
        """
        Initialize the Bronze Layer.

        Args:
            bronze_base_path: Base path for Bronze layer storage
            namespace: Namespace for OpenLineage tracking
        """
        self.bronze_base_path = Path(bronze_base_path)
        self.namespace = namespace
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create Bronze layer directory structure if it doesn't exist."""
        directories = [
            self.bronze_base_path / "transactions",
            self.bronze_base_path / "customers",
            self.bronze_base_path / "metadata",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Bronze directories ensured at {self.bronze_base_path}")

    @lineage_trace(
        job_name="ingest_postgres_to_bronze",
        input_datasets=["postgresql_transactions"],
        output_datasets=["bronze_transactions"],
        namespace="sentinai.bronze",
    )
    def ingest_postgres_transactions(
        self,
        raw_data: pl.DataFrame,
        source_table: str = "transactions",
        partition_key: Optional[str] = None,
    ) -> str:
        """
        Ingest raw PostgreSQL transaction data into Bronze layer as immutable Parquet.

        This method is decorated with lineage_trace to ensure OpenLineage metadata
        emission for MRM compliance.

        Args:
            raw_data: Raw dataframe from PostgreSQL
            source_table: Source PostgreSQL table name
            partition_key: Optional partition key (e.g., date) for storage organization

        Returns:
            Path to the written Parquet file
        """
        run_id = str(uuid.uuid4())
        logger.info(f"Starting Bronze ingestion for {source_table} (run_id={run_id})")

        try:
            # Add Bronze layer metadata
            enriched_df = self._add_bronze_metadata(
                raw_data, source_table, DataSourceType.POSTGRESQL
            )

            # Convert to PyArrow for Parquet writing
            arrow_table = enriched_df.to_arrow()

            # Determine output path
            partition_path = partition_key or datetime.now(timezone.utc).strftime("%Y-%m-%d")
            output_dir = self.bronze_base_path / "transactions" / partition_path
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"bronze_transactions_{run_id}.parquet"

            # Write to Parquet (immutable - once written, never modified)
            pq.write_table(arrow_table, output_path)

            # Emit transformation metadata
            emit_transformation_metadata(
                job_name="ingest_postgres_to_bronze",
                run_id=run_id,
                transformation_python="bronze_ingest_postgres",
                input_rows=raw_data.height,
                output_rows=enriched_df.height,
            )

            # Write metadata file
            self._write_bronze_metadata(
                output_path, enriched_df, source_table, run_id, partition_key
            )

            logger.info(f"Bronze ingestion complete: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Bronze ingestion failed: {e}")
            raise

    @lineage_trace(
        job_name="ingest_synthetic_to_bronze",
        input_datasets=["synthetic_generator"],
        output_datasets=["bronze_synthetic_transactions"],
        namespace="sentinai.bronze",
    )
    def ingest_synthetic_data(
        self,
        synthetic_data: pl.DataFrame,
        source_table: str = "synthetic_transactions",
        partition_key: Optional[str] = None,
    ) -> str:
        """
        Ingest synthetic data into Bronze layer, treating it exactly like real data.

        This ensures that lineage tracking works for all data types. Synthetic data
        is tagged with synthetic_flag=True for auditor distinguishability.

        Args:
            synthetic_data: Synthetic dataframe from generator
            source_table: Source table name for tracking
            partition_key: Optional partition key for storage organization

        Returns:
            Path to the written Parquet file
        """
        run_id = str(uuid.uuid4())
        logger.info(f"Starting synthetic Bronze ingestion (run_id={run_id})")

        try:
            # Map synthetic data columns to bronze schema
            mapped_df = synthetic_data.rename({
                "transaction_id": "customer_id",
                "sender_id": "customer_name", 
                "transaction_amount": "amount"
            })
            
            # Add Bronze layer metadata with synthetic flag
            enriched_df = self._add_bronze_metadata(
                mapped_df, source_table, DataSourceType.SYNTHETIC, synthetic_flag=True
            )

            # Determine output path
            partition_path = partition_key or datetime.now(timezone.utc).strftime("%Y-%m-%d")
            output_dir = self.bronze_base_path / "transactions" / partition_path
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"bronze_synthetic_{run_id}.parquet"

            # Debug: Log dataframe schema before writing
            logger.info(f"DataFrame schema: {enriched_df.schema}")
            logger.info(f"DataFrame shape: {enriched_df.shape}")
            
            # Convert Object types to String for Parquet compatibility using map_elements
            for col in enriched_df.columns:
                if enriched_df[col].dtype == pl.Object:
                    enriched_df = enriched_df.with_columns(
                        pl.col(col).map_elements(lambda x: str(x) if x is not None else None, return_dtype=pl.String)
                    )
            
            # Write to Parquet directly using Polars (avoids Arrow conversion issues)
            enriched_df.write_parquet(output_path)

            # Emit transformation metadata
            emit_transformation_metadata(
                job_name="ingest_synthetic_to_bronze",
                run_id=run_id,
                transformation_python="bronze_ingest_synthetic",
                input_rows=synthetic_data.height,
                output_rows=enriched_df.height,
            )

            # Write metadata file
            self._write_bronze_metadata(
                output_path, enriched_df, source_table, run_id, partition_key
            )

            logger.info(f"Synthetic Bronze ingestion complete: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Synthetic Bronze ingestion failed: {e}")
            raise

    def _add_bronze_metadata(
        self,
        df: pl.DataFrame,
        source_table: str,
        source_type: DataSourceType,
        synthetic_flag: bool = False,
    ) -> pl.DataFrame:
        """
        Add Bronze layer metadata columns to the dataframe.

        Args:
            df: Input dataframe
            source_table: Source table name
            source_type: Type of data source
            synthetic_flag: Whether this is synthetic data

        Returns:
            Dataframe with Bronze metadata columns added
        """
        # Ensure base columns exist with proper data types
        base_cols = [
            ("customer_id", pl.String),
            ("customer_name", pl.String),
            ("email", pl.String),
            ("tax_id", pl.String),
            ("currency", pl.String),
            ("amount", pl.Float64),
            ("timestamp", pl.String),
        ]
        for col, dtype in base_cols:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None, dtype=dtype).alias(col))

        # Add Bronze layer metadata with explicit types
        enriched_df = df.with_columns([
            pl.lit(source_table, dtype=pl.String).alias("source_table"),
            pl.lit(datetime.now(timezone.utc).isoformat(), dtype=pl.String).alias("ingestion_date"),
            pl.lit(source_type.value, dtype=pl.String).alias("source_type"),
            pl.lit(synthetic_flag, dtype=pl.Boolean).alias("synthetic_flag"),
        ])

        return enriched_df

    def _write_bronze_metadata(
        self,
        parquet_path: Path,
        df: pl.DataFrame,
        source_table: str,
        run_id: str,
        partition_key: Optional[str],
    ) -> None:
        """
        Write metadata file for Bronze layer dataset.

        Args:
            parquet_path: Path to the Parquet file
            df: Dataframe that was written
            source_table: Source table name
            run_id: Run identifier
            partition_key: Partition key used
        """
        import json

        metadata = {
            "layer": "bronze",
            "source_table": source_table,
            "run_id": run_id,
            "partition_key": partition_key,
            "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
            "row_count": df.height,
            "schema": {col: str(dtype) for col, dtype in df.schema.items()},
            "parquet_path": str(parquet_path),
            "immutable": True,  # Bronze layer is immutable
        }

        metadata_path = (
            self.bronze_base_path / "metadata" / f"bronze_metadata_{run_id}.json"
        )
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Bronze metadata written to {metadata_path}")

    def read_bronze_partition(self, partition_key: str) -> pl.DataFrame:
        """
        Read all Bronze data for a specific partition.

        Args:
            partition_key: Partition key (e.g., date)

        Returns:
            Combined dataframe from all files in the partition
        """
        partition_dir = self.bronze_base_path / "transactions" / partition_key
        if not partition_dir.exists():
            logger.warning(f"Partition directory does not exist: {partition_dir}")
            return pl.DataFrame()

        # Read all parquet files in the partition
        parquet_files = list(partition_dir.glob("*.parquet"))
        if not parquet_files:
            logger.warning(f"No parquet files found in partition: {partition_dir}")
            return pl.DataFrame()

        dataframes = []
        for file_path in parquet_files:
            try:
                df = pl.read_parquet(file_path)
                dataframes.append(df)
            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")

        if not dataframes:
            return pl.DataFrame()

        combined_df = pl.concat(dataframes)
        logger.info(f"Read {combined_df.height} rows from Bronze partition {partition_key}")
        return combined_df
