import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import httpx
import numpy as np
from dotenv import load_dotenv
from datetime import datetime, timezone
import uuid

# Load environment variables from .env file
load_dotenv()

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from src.retrieval.schemas import ChunkModel, ChunkMetadata
from openlineage.client import OpenLineageClient
from openlineage.client.run import RunEvent, RunState, Run, Job, Dataset

logger = logging.getLogger(__name__)

class RateLimitError(Exception):
    """Custom exception for rate limiting."""
    pass

class EmbeddingServiceAPI:
    """Embedding service using OpenRouter API."""
    def __init__(
        self,
        model_name: str = "openai/text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1"
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not provided. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts using OpenRouter API.
        Returns list of embedding vectors.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "SentinAI",
            "HTTP-Referer": "https://github.com/Daniel-MM24/sentinAI"
        }
        
        # OpenRouter uses OpenAI-compatible embedding API
        payload = {
            "model": self.model_name,
            "input": texts
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=payload,
                timeout=60.0
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract embeddings from response
            embeddings = [item["embedding"] for item in data["data"]]
            return embeddings

class DocumentIngestor:
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        embedding_api: Optional[EmbeddingServiceAPI] = None,
        embedding_model: str = "openai/text-embedding-3-small"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_api = embedding_api or EmbeddingServiceAPI(model_name=embedding_model)
        
        # Semantic Chunking: context-aware, prevents splitting critical headers
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=False
        )
        
        # Initialize OpenLineage client for document ingestion operations
        self.ol_client = OpenLineageClient()
        self.namespace = "sentinai.retrieval"
        self.job_name = "document_ingestion"

    def process_pdf(
        self, 
        file_path: str, 
        retention_policy_id: str = "default_retention", 
        sensitivity_label: str = "Internal"
    ) -> List[ChunkModel]:
        """
        Extracts content while maintaining audit-traceable metadata.
        """
        run_id = str(uuid.uuid4())
        
        # Emit START event for PDF processing
        try:
            start_event = RunEvent(
                eventType=RunState.START,
                eventTime=datetime.now(timezone.utc).isoformat(),
                run=Run(runId=run_id),
                job=Job(namespace=self.namespace, name=f"{self.job_name}_process_pdf"),
                producer="sentinai",
                inputs=[Dataset(namespace="sentinai.files", name=file_path)],
                outputs=[]
            )
            self.ol_client.emit(start_event)
            logger.info(f"Lineage START emitted: document_ingestion_process_pdf (run_id={run_id})")
        except Exception as e:
            logger.warning(f"Failed to emit START event: {e}")
        
        # Load PDF using Langchain's PyPDFLoader which extracts page numbers
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        chunk_models = []
        
        for doc in documents:
            page_content = doc.page_content
            metadata = doc.metadata
            page_number = metadata.get("page", 0) + 1  # 1-indexed
            
            # Split into chunks
            chunks = self.text_splitter.split_text(page_content)
            
            for chunk_text in chunks:
                chunk_model = ChunkModel.create(
                    content=chunk_text,
                    source_origin=file_path,
                    retention_policy_id=retention_policy_id,
                    sensitivity_label=sensitivity_label,
                    page_number=page_number,
                    document_version="1.0"
                )
                chunk_models.append(chunk_model)
        
        # Emit COMPLETE event for PDF processing
        try:
            complete_event = RunEvent(
                eventType=RunState.COMPLETE,
                eventTime=datetime.now(timezone.utc).isoformat(),
                run=Run(runId=run_id),
                job=Job(namespace=self.namespace, name=f"{self.job_name}_process_pdf"),
                producer="sentinai",
                inputs=[Dataset(namespace="sentinai.files", name=file_path)],
                outputs=[Dataset(namespace=self.namespace, name=f"chunks_{file_path}")]
            )
            self.ol_client.emit(complete_event)
            logger.info(f"Lineage COMPLETE emitted: document_ingestion_process_pdf (run_id={run_id})")
        except Exception as e:
            logger.warning(f"Failed to emit COMPLETE event: {e}")
                
        return chunk_models

    def load_markdown_seed(
        self,
        seed_path: str,
        retention_policy_id: str = "regulatory_retention",
        sensitivity_label: str = "Confidential",
        source_origin: Optional[str] = None
    ) -> List[ChunkModel]:
        """
        Load pre-chunked regulatory seed data from seed_data.json and wrap each
        entry in a ChunkModel with audit-compliant metadata and a SHA-256 hash.

        Expects JSON format:
            {"documents": [{"id": str, "document": str, "metadata": {...}}, ...]}
        """
        run_id = str(uuid.uuid4())
        
        # Emit START event for seed loading
        try:
            start_event = RunEvent(
                eventType=RunState.START,
                eventTime=datetime.now(timezone.utc).isoformat(),
                run=Run(runId=run_id),
                job=Job(namespace=self.namespace, name=f"{self.job_name}_load_seed"),
                producer="sentinai",
                inputs=[Dataset(namespace="sentinai.files", name=seed_path)],
                outputs=[]
            )
            self.ol_client.emit(start_event)
            logger.info(f"Lineage START emitted: document_ingestion_load_seed (run_id={run_id})")
        except Exception as e:
            logger.warning(f"Failed to emit START event: {e}")
        
        path = Path(seed_path)
        if not path.exists():
            logger.error("Seed file not found: %s", seed_path)
            return []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_docs = data.get("documents", [])
        if not raw_docs:
            logger.warning("No documents found in seed file: %s", seed_path)
            return []

        chunk_models: List[ChunkModel] = []
        for doc in raw_docs:
            chunk_models.append(
                ChunkModel.create(
                    content=doc["document"],
                    source_origin=source_origin or doc["metadata"].get("source", seed_path),
                    retention_policy_id=retention_policy_id,
                    sensitivity_label=sensitivity_label,
                    document_version=doc.get("metadata", {}).get("version", "1.0")
                )
            )

        logger.info(
            "Loaded %d seed chunks from %s",
            len(chunk_models), seed_path
        )
        
        # Emit COMPLETE event for seed loading
        try:
            complete_event = RunEvent(
                eventType=RunState.COMPLETE,
                eventTime=datetime.now(timezone.utc).isoformat(),
                run=Run(runId=run_id),
                job=Job(namespace=self.namespace, name=f"{self.job_name}_load_seed"),
                producer="sentinai",
                inputs=[Dataset(namespace="sentinai.files", name=seed_path)],
                outputs=[Dataset(namespace=self.namespace, name=f"seed_chunks_{seed_path}")]
            )
            self.ol_client.emit(complete_event)
            logger.info(f"Lineage COMPLETE emitted: document_ingestion_load_seed (run_id={run_id})")
        except Exception as e:
            logger.warning(f"Failed to emit COMPLETE event: {e}")
        
        return chunk_models

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((RateLimitError, ConnectionError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Rate limit hit. Retrying in {retry_state.next_action.sleep} seconds..."
        )
    )
    async def embed_chunks_resilient(self, chunks: List[ChunkModel], batch_size: int = 100) -> List[Dict[str, Any]]:
        """
        Resilient embeddings with batching logic and exponential backoff.
        """
        all_embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [chunk.content for chunk in batch]
            
            # Calling the external embedding API
            embeddings = await self.embedding_api.embed_batch(texts)
            
            for chunk, emb in zip(batch, embeddings):
                all_embeddings.append({
                    "chunk": chunk,
                    "embedding": emb
                })
        
        return all_embeddings
