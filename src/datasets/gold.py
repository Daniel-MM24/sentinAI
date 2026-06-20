import logging
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import polars as pl
from openlineage.client import OpenLineageClient
from openlineage.client.run import RunEvent, RunState, Dataset

from src.datasets.registry import FeatureRegistry

logger = logging.getLogger(__name__)

class GoldLayer:
    """
    GoldLayer orchestrates the transformation from Silver datasets to a denormalized,
    feature-rich schema (Feature Store).
    """

    def __init__(self, version: str, ol_client: Optional[OpenLineageClient] = None, registry: Optional[FeatureRegistry] = None):
        self.version = version
        self.ol_client = ol_client or OpenLineageClient()
        self.registry = registry or FeatureRegistry()
        self._namespace = "sentinai.features"
        self._job_name = "silver_to_gold_transform"

    def create_feature_store(self, transactions_df: pl.DataFrame, customers_df: pl.DataFrame) -> str:
        """
        Materializes the Gold layer feature store by joining fact and dimension tables.
        Returns the URI/Path of the versioned feature dataset.
        """
        run_id = str(uuid.uuid4())
        self._emit_lineage(run_id, RunState.START, inputs=[
            self._create_ol_dataset("silver_transactions"),
            self._create_ol_dataset("silver_customers")
        ])

        try:
            logger.info("Starting Gold Layer Transformation...")

            local_dir = f"/tmp/sentinai/gold/features/v{self.version}/" 
            os.makedirs(local_dir, exist_ok=True)

            # 1. Join logic (Fact + Dimension)
            # Identify orphans (transactions without matching customers)
            orphans_df = transactions_df.join(customers_df, on="customer_id", how="anti")
            if orphans_df.height > 0:
                logger.warning(f"Anomaly Detected: {orphans_df.height} orphan transactions found (no matching customer).")
                # Ensure we capture this for the Audit trail
                orphan_path = os.path.join(local_dir, "anomaly_orphans.parquet")
                orphans_df.write_parquet(orphan_path)

            # Inner join to create the denormalized feature set
            feature_df = transactions_df.join(customers_df, on="customer_id", how="inner")

            # Extract date for partitioning
            if "timestamp" in feature_df.columns:
                if feature_df.schema["timestamp"] == pl.String:
                    feature_df = feature_df.with_columns(
                        pl.col("timestamp").str.to_datetime().alias("timestamp")
                    )
                feature_df = feature_df.with_columns(
                    pl.col("timestamp").dt.date().cast(pl.String).alias("partition_date")
                )
            else:
                feature_df = feature_df.with_columns(
                    pl.lit(datetime.utcnow().date().isoformat()).alias("partition_date")
                )

            # Validate against GoldFeatureSchema
            from src.datasets.schemas import GoldFeatureSchema
            feature_df = GoldFeatureSchema.validate(feature_df)

            # 2. Schema enforcement / extraction
            schema_dict = {col: str(dtype) for col, dtype in feature_df.schema.items()}

            # 3. Snapshotting & Versioning
            # In production, this would go to S3. We use a local mock for development/testing if s3 is not configured
            output_uri = f"s3://sentinai/gold/features/v{self.version}/"
            
            # Write parquet partitioned by date
            feature_df.write_parquet(local_dir, use_pyarrow=True, partition_by="partition_date")

            # MRM Standards: Manifest file capturing lineage and schema
            manifest = {
                "version": self.version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input_rows_fact": transactions_df.height,
                "input_rows_dim": customers_df.height,
                "output_rows": feature_df.height,
                "orphan_rows": orphans_df.height,
                "schema": schema_dict
            }

            manifest_path = os.path.join(local_dir, "manifest.json")
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            # 4. Registration
            self.registry.register_feature_set(
                feature_uri=output_uri,
                version=self.version,
                schema=schema_dict,
                metadata=manifest
            )

            # 5. OpenLineage registration
            self._emit_lineage(run_id, RunState.COMPLETE, outputs=[self._create_ol_dataset("gold_feature_store")])
            
            logger.info("Gold Layer Transformation complete.")
            return output_uri

        except Exception as e:
            self._emit_lineage(run_id, RunState.FAIL)
            logger.error(f"Transformation failed: {str(e)}")
            raise e

    def _emit_lineage(self, run_id: str, state: RunState, inputs: list = None, outputs: list = None) -> None:
        """
        Emits OpenLineage metadata to track feature transformations (Feature Contract).
        """
        event = RunEvent(
            eventType=state,
            eventTime=datetime.now(timezone.utc).isoformat(),
            run={"runId": run_id},
            job={"namespace": self._namespace, "name": self._job_name},
            inputs=inputs or [],
            outputs=outputs or []
        )
        self.ol_client.emit(event)

    def _create_ol_dataset(self, name: str) -> Dataset:
        return Dataset(namespace=self._namespace, name=name)
