"""
# Data Connectors Module

This module provides data extraction capabilities from various sources (PostgreSQL, S3)
while enforcing Medallion Architecture standards, MRM compliance, and OpenLineage observability.

## How to Extend for New Data Sources

To add a new data source:
1. Define a Pydantic schema for the expected data format (similar to `TransactionSchema`).
2. Add a new extraction method to `DataConnector` (e.g., `extract_market_data`).
3. Ensure the method begins by instantiating an OpenLineage `RunEvent`.
4. Use `tenacity` for retries if the network interaction is flaky.
5. Validate the incoming data against your schema.
6. Return or write the validated data to the Bronze layer with attached MRM metadata.
"""

import logging
import datetime
import json
import os
import uuid
import asyncio
from typing import List, Any

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import text
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

import boto3
from openlineage.client import OpenLineageClient
from openlineage.client.run import RunEvent, RunState, Run
from openlineage.client.job import Job
from openlineage.client.transport.http import HttpConfig, HttpTransport

from src.core.config import config


class TransactionSchema(BaseModel):
    """Base schema for transactions."""
    transaction_id: str
    amount: float
    currency: str
    
    # MRM (Model Risk Management) fields for Bronze layer
    source_origin: str = Field(description="The origin source of the data.")
    ingestion_timestamp: datetime.datetime = Field(
        default_factory=datetime.datetime.utcnow, 
        description="Timestamp of ingestion."
    )
    data_governance_version: str = Field(
        default="v1.0", 
        description="Governance version applied during ingestion."
    )


class DataConnector:
    def __init__(self):
        self.logger = logging.getLogger("sentinai.data.connectors")
        
        # Configure logging
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Database async engine
        self.engine = create_async_engine(config.DB_URL)
        
        # S3 client (IAM role-based auth default or credentials from config)
        if config.AWS_ACCESS_KEY_ID and config.AWS_SECRET_ACCESS_KEY:
            self.s3_client = boto3.client(
                's3',
                region_name=config.AWS_REGION,
                aws_access_key_id=config.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY
            )
        else:
            self.s3_client = boto3.client('s3', region_name=config.AWS_REGION)

        # OpenLineage Client Setup
        # If OPENLINEAGE_URL is not set, we use a console transport for local debugging,
        # but in production it should point to Marquez or another OL backend.
        if config.OPENLINEAGE_URL:
            transport = HttpTransport(HttpConfig(url=config.OPENLINEAGE_URL))
            self.ol_client = OpenLineageClient(transport=transport)
        else:
            self.ol_client = OpenLineageClient() # Defaults to console if no env vars are set
            
        self.namespace = config.OPENLINEAGE_NAMESPACE

    def _emit_lineage(self, run_id: str, job_name: str, state: RunState, error_msg: str = None):
        """Helper to emit OpenLineage events."""
        event_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
        run = Run(runId=run_id)
        job = Job(namespace=self.namespace, name=job_name)
        
        event = RunEvent(
            eventType=state,
            eventTime=event_time,
            run=run,
            job=job,
            inputs=[],
            outputs=[]
        )
        try:
            self.ol_client.emit(event)
            self.logger.info(f"Emitted OpenLineage event: {job_name} {state}")
        except Exception as e:
            self.logger.error(f"Failed to emit OpenLineage event: {str(e)}")

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def extract_transactions(self, start_date: str, end_date: str) -> List[TransactionSchema]:
        """
        Extracts transactional data from PostgreSQL with integrated OpenLineage emission
        and resilient retry logic.
        """
        job_name = "extract_postgres_transactions"
        run_id = str(uuid.uuid4())
        
        self.logger.info(f"Starting transaction extraction [{start_date} to {end_date}]. Run ID: {run_id}")
        
        # 1. Start OpenLineage Run
        self._emit_lineage(run_id, job_name, RunState.START)
        
        extracted_data = []
        try:
            # 2. Execute Async SQL Query
            query = text(
                "SELECT transaction_id, amount, currency FROM transactions "
                "WHERE date >= :start_date AND date <= :end_date"
            )
            
            async with self.engine.connect() as conn:
                result = await conn.execute(query, {"start_date": start_date, "end_date": end_date})
                rows = result.fetchall()
                
            # 3. Validate against TransactionSchema
            for row in rows:
                tx = TransactionSchema(
                    transaction_id=str(row.transaction_id),
                    amount=float(row.amount),
                    currency=str(row.currency),
                    source_origin="PostgreSQL",
                    # ingestion_timestamp and data_governance_version are auto-populated
                )
                extracted_data.append(tx)
                
            # 4. Write to Bronze Layer
            self._write_to_bronze_layer(extracted_data, job_name, run_id)
                
            # 5. Log success with metadata and emit complete
            self.logger.info(f"Successfully extracted and wrote {len(extracted_data)} transactions to Bronze. Run ID: {run_id}")
            self._emit_lineage(run_id, job_name, RunState.COMPLETE)
            
            return extracted_data

        except Exception as e:
            # Log failure and emit fail event
            self.logger.error(f"Extraction failed for Run ID {run_id}: {str(e)}")
            self._emit_lineage(run_id, job_name, RunState.FAIL, error_msg=str(e))
            raise e

    def _write_to_bronze_layer(self, data: List[TransactionSchema], source: str, run_id: str):
        """
        Writes validated data to the Bronze layer (local filesystem for now).
        In a production scenario, this might write to S3 as Parquet.
        """
        bronze_dir = os.path.join("data", "bronze", source)
        os.makedirs(bronze_dir, exist_ok=True)
        
        file_path = os.path.join(bronze_dir, f"run_{run_id}.jsonl")
        
        with open(file_path, "w", encoding="utf-8") as f:
            for record in data:
                f.write(record.model_dump_json() + "\n")
                
        self.logger.info(f"Wrote {len(data)} records to Bronze layer at {file_path}")

    def extract_from_s3(self, bucket_name: str, object_key: str) -> bytes:
        """
        Synchronous example of extracting data from S3 using Boto3.
        """
        job_name = f"extract_s3_{bucket_name}"
        run_id = str(uuid.uuid4())
        
        self.logger.info(f"Starting S3 extraction [s3://{bucket_name}/{object_key}]. Run ID: {run_id}")
        self._emit_lineage(run_id, job_name, RunState.START)
        
        try:
            response = self.s3_client.get_object(Bucket=bucket_name, Key=object_key)
            data = response['Body'].read()
            
            self.logger.info(f"Successfully extracted {len(data)} bytes from S3. Run ID: {run_id}")
            self._emit_lineage(run_id, job_name, RunState.COMPLETE)
            return data
            
        except Exception as e:
            self.logger.error(f"S3 Extraction failed for Run ID {run_id}: {str(e)}")
            self._emit_lineage(run_id, job_name, RunState.FAIL, error_msg=str(e))
            raise e
