import asyncio
import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from src.retrieval.schemas import ChunkModel, ChunkMetadata

logger = logging.getLogger(__name__)

class RateLimitError(Exception):
    """Custom exception for rate limiting."""
    pass

class EmbeddingServiceAPI:
    """Real embedding service using sentence-transformers for local inference."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L-6-v2"):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts using sentence-transformers.
        Returns list of embedding vectors.
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, 
            self.model.encode,
            texts
        )
        return embeddings.tolist()

class DocumentIngestor:
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        embedding_api: Optional[EmbeddingServiceAPI] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L-6-v2"
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

    def process_pdf(
        self, 
        file_path: str, 
        retention_policy_id: str = "default_retention", 
        sensitivity_label: str = "Internal"
    ) -> List[ChunkModel]:
        """
        Extracts content while maintaining audit-traceable metadata.
        """
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
