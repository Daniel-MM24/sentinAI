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
from src.data.lineage_decorator import lineage_trace, emit_transformation_metadata
from src.data.anomaly_engine import finalize_and_partition_gold
import numpy as np

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

    @lineage_trace(
        job_name="silver_to_gold_transform",
        input_datasets=["silver_transactions", "silver_customers"],
        output_datasets=["gold_feature_store"],
        namespace="sentinai.features",
    )
    def create_feature_store(self, transactions_df: pl.DataFrame, customers_df: pl.DataFrame) -> str:
        """
        Materializes the Gold layer feature store by joining fact and dimension tables.
        Returns the URI/Path of the versioned feature dataset.
        
        This method is decorated with lineage_trace to ensure OpenLineage metadata
        emission for MRM compliance.
        """
        run_id = str(uuid.uuid4())

        try:
            logger.info("Starting Gold Layer Transformation...")

            local_dir = f"data/gold/features/v{self.version}/" 
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

            # Ensure timestamp is datetime type
            if "timestamp" in feature_df.columns:
                timestamp_type = feature_df.schema["timestamp"]
                if timestamp_type == pl.String:
                    feature_df = feature_df.with_columns(
                        pl.col("timestamp").str.to_datetime(time_zone="UTC").alias("timestamp")
                    )
                elif isinstance(timestamp_type, pl.Datetime) and timestamp_type.time_zone != "UTC":
                    feature_df = feature_df.with_columns(
                        pl.col("timestamp").dt.replace_time_zone("UTC").alias("timestamp")
                    )
            else:
                feature_df = feature_df.with_columns(
                    pl.lit(datetime.now(timezone.utc)).alias("timestamp")
                )

            # Extract date for partitioning
            feature_df = feature_df.with_columns(
                pl.col("timestamp").dt.date().cast(pl.String).alias("partition_date")
            )

            # Feature Engineering
            feature_df = feature_df.with_columns(
                # Temporal features (cast to Int64 to match schema)
                pl.col("timestamp").dt.day().cast(pl.Int64).alias("day_of_month"),
                pl.col("timestamp").dt.weekday().cast(pl.Int64).alias("day_of_week"),
                pl.col("timestamp").dt.hour().cast(pl.Int64).alias("hour_of_day"),
                # Base model card features
                (pl.col("timestamp").dt.weekday() >= 5).cast(pl.Boolean).alias("is_weekend"),
                pl.col("amount").log1p().alias("log_amount"),
                # Handle anomaly_case_id - use anomaly_type if available, otherwise generate synthetic ID
                pl.when(pl.col("anomaly_type").is_not_null())
                .then(pl.col("anomaly_type"))
                .otherwise(pl.lit(None, dtype=pl.String))
                .alias("anomaly_case_id")
            )
            
            # Key Engineered Features from Model Card
            feature_df = feature_df.with_columns(
                # amount_near_threshold (1 if amount is KES 140,000–149,999)
                ((pl.col("amount") >= 140000) & (pl.col("amount") < 150000)).cast(pl.Int64).alias("amount_near_threshold"),
                # is_round_number_100k
                ((pl.col("amount") % 100000 == 0) & (pl.col("amount") > 0)).cast(pl.Int64).alias("is_round_number_100k"),
                # Channel type flags
                (pl.col("currency") == "C2B").cast(pl.Int64).alias("is_stk_push"),
                (pl.col("currency") == "B2C").cast(pl.Int64).alias("is_b2c")
            )

            # Calculate transaction velocity (transactions per customer)
            if feature_df.height > 0:
                velocity_df = feature_df.group_by("customer_id").agg(
                    pl.len().alias("transaction_count")
                )
                feature_df = feature_df.join(velocity_df, on="customer_id", how="left", coalesce=True)
                feature_df = feature_df.with_columns(
                    pl.col("transaction_count").cast(pl.Float64).alias("transaction_velocity")
                )
            else:
                feature_df = feature_df.with_columns(
                    pl.lit(None).cast(pl.Float64).alias("transaction_velocity")
                )

            # Calculate mean transaction amount per customer
            if feature_df.height > 0:
                mean_amount_df = feature_df.group_by("customer_id").agg(
                    pl.col("amount").mean().alias("mean_transaction_amount")
                )
                feature_df = feature_df.join(mean_amount_df, on="customer_id", how="left", coalesce=True)
            else:
                feature_df = feature_df.with_columns(
                    pl.lit(None).cast(pl.Float64).alias("mean_transaction_amount")
                )

            # Calculate CLV (Customer Lifetime Value) - simplified as sum of amounts
            if feature_df.height > 0:
                clv_df = feature_df.group_by("customer_id").agg(
                    pl.col("amount").sum().alias("clv")
                )
                feature_df = feature_df.join(clv_df, on="customer_id", how="left", coalesce=True)
            else:
                feature_df = feature_df.with_columns(
                    pl.lit(None).cast(pl.Float64).alias("clv")
                )

            # High risk amount flag (amount > 10000)
            feature_df = feature_df.with_columns(
                (pl.col("amount") > 10000).cast(pl.Int64).alias("high_risk_amount")
            )

            # Z-score deviation (simplified - using amount deviation from mean)
            if feature_df.height > 0:
                global_mean = feature_df.select(pl.col("amount").mean()).item()
                global_std = feature_df.select(pl.col("amount").std()).item()
                if global_std and global_std > 0:
                    feature_df = feature_df.with_columns(
                        ((pl.col("amount") - global_mean) / global_std).alias("z_score_deviation")
                    )
                else:
                    feature_df = feature_df.with_columns(
                        pl.lit(0.0).alias("z_score_deviation")
                    )
            else:
                feature_df = feature_df.with_columns(
                    pl.lit(0.0).alias("z_score_deviation")
                )

            # Validate against GoldFeatureSchema
            from src.datasets.schemas import GoldFeatureSchema
            feature_df = GoldFeatureSchema.validate(feature_df)

            # 2. Schema enforcement / extraction
            schema_dict = {col: str(dtype) for col, dtype in feature_df.schema.items()}

            # 3. Snapshotting & Versioning
            # Return the local path for the materialized dataset
            output_uri = local_dir
            
            # Write partitioned dataset
            finalize_and_partition_gold(feature_df, local_dir)

            # MRM Standards: Manifest file capturing lineage and schema
            manifest = {
                "version": self.version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input_rows_fact": transactions_df.height,
                "input_rows_dim": customers_df.height,
                "output_rows": feature_df.height,
                "orphan_rows": orphans_df.height,
                "schema": schema_dict,
                "differential_privacy_budget": {
                    "epsilon": 0.0833,
                    "delta": 2.0e-7
                }
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

            # Emit transformation metadata for auditability
            emit_transformation_metadata(
                job_name="silver_to_gold_transform",
                run_id=run_id,
                transformation_python="gold_feature_store",
                input_rows=transactions_df.height + customers_df.height,
                output_rows=feature_df.height,
            )
            
            logger.info("Gold Layer Transformation complete.")
            return output_uri

        except Exception as e:
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
