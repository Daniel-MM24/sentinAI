import json
from typing import Any, Dict, Optional

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.base import BaseEstimator, TransformerMixin

# Try importing Great Expectations and OpenLineage
try:
    import great_expectations as gx
    GX_AVAILABLE = True
except ImportError:
    GX_AVAILABLE = False

try:
    from openlineage.client import OpenLineageClient
    from openlineage.client.dataset import Dataset
    from openlineage.client.run import Job, Run, RunEvent, RunState
    OL_AVAILABLE = True
except ImportError:
    OL_AVAILABLE = False


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transforms Silver layer data into Gold layer high-signal feature vectors.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.output_path = self.config.get("output_path", "gold_features.parquet")
        self.feature_metadata: Dict[str, str] = {}
        self.is_fitted = False

    def fit(self, X: pl.DataFrame, y: Any = None) -> "FeatureEngineer":
        """
        Fit the transformer. Currently stateless, but reserved for future
        stateful operations like computing scalers or means.
        """
        self.is_fitted = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Orchestrates the feature engineering pipeline.
        """
        # 1. Temporal Features
        df = self.create_temporal_features(df)

        # 2. Risk Indicators
        df = self.create_risk_indicators(df)

        # 3. Data Quality Validation
        if GX_AVAILABLE:
            self.validate_features(df)

        # 4. Lineage Tracking
        if OL_AVAILABLE:
            self.emit_lineage(df)

        # 5. Export with metadata
        self.export_to_parquet(df)

        return df

    def create_temporal_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Generates cyclical and rolling time-based features.
        """
        if "timestamp" not in df.columns:
            return df

        df = df.with_columns(
            [
                pl.col("timestamp").dt.day().alias("day_of_month"),
                pl.col("timestamp").dt.weekday().alias("day_of_week"),
                pl.col("timestamp").dt.hour().alias("hour_of_day"),
            ]
        )

        self.feature_metadata.update(
            {
                "day_of_month": "Extracted from timestamp to capture monthly patterns.",
                "day_of_week": "Extracted from timestamp to capture weekly patterns.",
                "hour_of_day": "Extracted from timestamp to capture intra-day patterns.",
            }
        )
        return df

    def create_risk_indicators(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Derives features for fraud/risk models.
        Includes meta-tags for SHAP/LIME explanation mapping.
        """
        if "amount" in df.columns:
            df = df.with_columns(
                [(pl.col("amount") > 10000).cast(pl.Int8).alias("high_risk_amount")]
            )
            self.feature_metadata["high_risk_amount"] = (
                "Binary flag for amounts > 10000 (Reg 12.B AML threshold check)."
            )

        if (
            "amount" in df.columns
            and "customer_id" in df.columns
            and "timestamp" in df.columns
        ):
            # Sort by timestamp for proper rolling calculations
            std_dev = pl.col("amount").std().over("customer_id")
            mean = pl.col("amount").mean().over("customer_id")
            
            z_score = pl.when(std_dev.is_not_null() & (std_dev > 0)).then(
                (pl.col("amount") - mean) / std_dev
            ).otherwise(0.0)

            df = df.sort("timestamp").with_columns(
                [
                    pl.col("amount")
                    .rolling_sum(window_size=3, min_periods=1)
                    .over("customer_id")
                    .alias("transaction_velocity"),
                    
                    pl.col("amount")
                    .cum_sum()
                    .over("customer_id")
                    .alias("clv"),
                    
                    mean.alias("mean_transaction_amount"),
                    
                    z_score.alias("z_score_deviation")
                ]
            )
            
            self.feature_metadata.update({
                "transaction_velocity": "Rolling 3-transaction amount sum per account to capture velocity risk.",
                "clv": "Cumulative Lifetime Value (total transaction amount per customer).",
                "mean_transaction_amount": "Historical mean transaction amount for the customer.",
                "z_score_deviation": "Deviation of current transaction from customer's historical mean in std devs. (Reason code mapping: >3 signifies anomaly)."
            })

        return df

    def validate_features(self, df: pl.DataFrame) -> None:
        """
        Validates the generated features using Great Expectations to prevent data drift.
        """
        try:
            # Convert to Pandas for ephemeral GX validation
            pdf = df.to_pandas()
            context = gx.get_context(mode="ephemeral")
            validator = context.sources.pandas_default.read_dataframe(pdf)

            if "high_risk_amount" in df.columns:
                validator.expect_column_values_to_be_in_set("high_risk_amount", [0, 1])

            if "transaction_velocity" in df.columns:
                validator.expect_column_values_to_not_be_null("transaction_velocity")

            results = validator.validate()
            if not results.success:
                # Log or raise an alert depending on production needs
                print("Warning: Feature validation failed. Check expectations.")

        except Exception as e:
            print(f"Validation step failed or skipped: {e}")

    def emit_lineage(self, df: pl.DataFrame) -> None:
        """
        Emits OpenLineage events tagging the Gold dataset and its columns.
        """
        try:
            from datetime import datetime, timezone

            client = OpenLineageClient(
                url=self.config.get("openlineage_url", "http://localhost:5000")
            )

            dataset = Dataset(
                namespace="sentinai.gold",
                name="feature_vectors",
                facets={
                    "schema": {
                        "_producer": "https://sentinai.com",
                        "_schemaURL": "",
                        "fields": [
                            {"name": col, "type": str(df.schema[col])}
                            for col in df.columns
                        ],
                    }
                },
            )

            event = RunEvent(
                eventType=RunState.COMPLETE,
                eventTime=datetime.now(timezone.utc).isoformat(),
                run=Run(runId="feature-engineer-run-id"),
                job=Job(namespace="sentinai.pipelines", name="feature_engineering"),
                outputs=[dataset],
            )
            client.emit(event)
        except Exception as e:
            print(f"Failed to emit lineage: {e}")

    def export_to_parquet(self, df: pl.DataFrame) -> None:
        """
        Exports the Polars DataFrame to Parquet, embedding explainability metadata.
        """
        table = df.to_arrow()

        # Prepare custom metadata
        custom_metadata = {
            b"sentinai.feature_importance": json.dumps(self.feature_metadata).encode(
                "utf-8"
            )
        }

        # Merge with existing metadata
        existing_metadata = table.schema.metadata or {}
        merged_metadata = {**existing_metadata, **custom_metadata}

        # Apply new metadata
        new_schema = table.schema.with_metadata(merged_metadata)
        table = table.cast(new_schema)

        # Write to Parquet
        pq.write_table(table, self.output_path)
