import os
import pyarrow as pa
import pyarrow.dataset as ds
import polars as pl
import logging

logger = logging.getLogger(__name__)

def finalize_and_partition_gold(feature_df: pl.DataFrame, output_dir: str):
    """
    Implements PyArrow explicit dataset serialization for the Gold tier.
    Partitions the dataset by anomaly_case_id.
    """
    logger.info(f"Finalizing and partitioning Gold data to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    table = feature_df.to_arrow()
    
    # Write partitioned dataset
    ds.write_dataset(
        data=table,
        base_dir=output_dir,
        format="parquet",
        partitioning=["anomaly_case_id"],
        existing_data_behavior="overwrite_or_ignore"
    )
    logger.info("Gold partitioning complete.")
