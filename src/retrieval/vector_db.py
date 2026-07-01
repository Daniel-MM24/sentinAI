import asyncio
import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict
import math
import numpy as np
import chromadb
from chromadb.config import Settings
from datetime import datetime, timezone
import uuid

from src.retrieval.schemas import SearchResult, ChunkMetadata
from openlineage.client import OpenLineageClient
from openlineage.client.run import RunEvent, RunState, Run, Job, Dataset

logger = logging.getLogger(__name__)

class BM25Engine:
    """Real BM25 Sparse Search Engine for keyword-based retrieval."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_freqs = defaultdict(int)
        self.idf = {}
        self.doc_lengths = []
        self.avg_doc_length = 0
        
    def index_documents(self, documents: List[Dict[str, Any]]):
        """Build BM25 index from documents."""
        self.corpus = documents
        self.doc_lengths = []
        
        # Tokenize and build document frequency
        for doc in documents:
            tokens = self._tokenize(doc['content'])
            self.doc_lengths.append(len(tokens))
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1
                
        # Calculate average document length
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        # Calculate IDF
        n_docs = len(documents)
        for token, freq in self.doc_freqs.items():
            self.idf[token] = math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1)
            
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization - lowercase and split on whitespace/punctuation."""
        text = text.lower()
        tokens = []
        for word in text.split():
            # Remove punctuation
            cleaned = ''.join(c for c in word if c.isalnum())
            if cleaned:
                tokens.append(cleaned)
        return tokens
        
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search using BM25 scoring."""
        if not self.corpus:
            return []
            
        query_tokens = self._tokenize(query)
        scores = []
        
        for idx, doc in enumerate(self.corpus):
            doc_tokens = self._tokenize(doc['content'])
            token_freqs = defaultdict(int)
            for token in doc_tokens:
                token_freqs[token] += 1
                
            score = 0
            for token in query_tokens:
                if token in token_freqs:
                    tf = token_freqs[token]
                    idf = self.idf.get(token, 0)
                    doc_length = self.doc_lengths[idx]
                    
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                    score += idf * (numerator / denominator)
            
            scores.append((score, idx, doc))
        
        # Sort by score descending and return top k
        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, idx, doc in scores[:k]:
            results.append({
                'id': doc.get('id', str(idx)),
                'content': doc['content'],
                'metadata': doc.get('metadata', {}),
                'score': score
            })
        
        return results

class CrossEncoderReranker:
    """Cross-Encoder Reranker - disabled when using OpenRouter."""
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", enabled: bool = False):
        self.enabled = enabled
        self.model_name = model_name
        if enabled:
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(model_name)
            except ImportError:
                logger.warning("sentence-transformers not available, reranking disabled")
                self.enabled = False
        
    async def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank documents using cross-encoder model if enabled, otherwise return as-is."""
        if not documents:
            return []
            
        if not self.enabled:
            # Return documents with default confidence scores
            for doc in documents:
                doc['confidence_score'] = 0.5
            return documents
            
        # Prepare query-document pairs
        pairs = [[query, doc['content']] for doc in documents]
        
        # Run inference in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            self.model.predict,
            pairs
        )
        
        # Assign scores and sort
        for doc, score in zip(documents, scores):
            doc['confidence_score'] = float(score)
        
        # Sort by confidence_score descending
        return sorted(documents, key=lambda x: x['confidence_score'], reverse=True)

class VectorStore:
    def __init__(
        self, 
        persist_directory: str = "./chroma_data", 
        collection_name: str = "sentinai_docs",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_persistent_client: bool = True
    ):
        """
        Designed for persistence in a distributed environment.
        ChromaDB is used here, but can be swapped with Pinecone via similar interfaces.
        
        Args:
            persist_directory: Directory for ChromaDB persistent storage
            collection_name: Name of the collection
            cross_encoder_model: Model name for cross-encoder reranker
            use_persistent_client: If True, uses PersistentClient for local-only mode.
                                   If False, uses AsyncHttpClient for server mode.
        """
        if use_persistent_client:
            # Use persistent client for local-only mode (no server required)
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(allow_reset=True)
            )
        else:
            # Use async HTTP client for server mode
            self.client = chromadb.AsyncHttpClient(
                host="localhost",
                port=8000,
                settings=Settings(allow_reset=True)
            )
        self.collection_name = collection_name
        self.sparse_search = BM25Engine()
        self.reranker = CrossEncoderReranker(model_name=cross_encoder_model, enabled=False)
        self._indexed = False
        self.use_persistent_client = use_persistent_client
        
        # Initialize OpenLineage client for vector store operations
        self.ol_client = OpenLineageClient()
        self.namespace = "sentinai.retrieval"
        self.job_name = "vector_store_operations"
        
    async def get_collection(self):
        # Handle both PersistentClient (sync) and AsyncHttpClient (async)
        if self.use_persistent_client:
            # PersistentClient is synchronous, wrap in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.client.get_or_create_collection,
                self.collection_name
            )
        else:
            # AsyncHttpClient is asynchronous
            return await self.client.get_or_create_collection(name=self.collection_name)

    async def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to both ChromaDB and BM25 index.
        """
        run_id = str(uuid.uuid4())
        
        # Emit START event for document ingestion
        try:
            start_event = RunEvent(
                eventType=RunState.START,
                eventTime=datetime.now(timezone.utc).isoformat(),
                run=Run(runId=run_id),
                job=Job(namespace=self.namespace, name=f"{self.job_name}_add_documents"),
                producer="sentinai",
                inputs=[],
                outputs=[]
            )
            self.ol_client.emit(start_event)
            logger.info(f"Lineage START emitted: vector_store_add_documents (run_id={run_id})")
        except Exception as e:
            logger.warning(f"Failed to emit START event: {e}")
        
        collection = await self.get_collection()

        # Prepare documents for ChromaDB
        ids = [doc.get('id', str(i)) for i, doc in enumerate(documents)]
        texts = [doc['content'] for doc in documents]

        # Filter metadata to remove None values (ChromaDB doesn't accept None)
        metadatas = []
        for doc in documents:
            meta = doc.get('metadata', {})
            filtered_meta = {k: v for k, v in meta.items() if v is not None}
            metadatas.append(filtered_meta)

        # Extract embeddings if provided
        embeddings = [doc.get('embedding') for doc in documents if doc.get('embedding')]
        # If all documents have embeddings, use them; otherwise let ChromaDB generate them
        use_embeddings = len(embeddings) == len(documents)

        # Add to ChromaDB
        if self.use_persistent_client:
            # PersistentClient is synchronous
            loop = asyncio.get_event_loop()
            if use_embeddings:
                await loop.run_in_executor(
                    None,
                    lambda: collection.add(
                        ids=ids,
                        documents=texts,
                        metadatas=metadatas,
                        embeddings=embeddings
                    )
                )
            else:
                await loop.run_in_executor(
                    None,
                    lambda: collection.add(
                        ids=ids,
                        documents=texts,
                        metadatas=metadatas
                    )
                )
        else:
            # AsyncHttpClient is asynchronous
            if use_embeddings:
                await collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
            else:
                await collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas
                )
        
        # Index for BM25
        self.sparse_search.index_documents(documents)
        self._indexed = True
        
        logger.info(f"Added {len(documents)} documents to vector store")
        
        # Emit COMPLETE event for document ingestion
        try:
            complete_event = RunEvent(
                eventType=RunState.COMPLETE,
                eventTime=datetime.now(timezone.utc).isoformat(),
                run=Run(runId=run_id),
                job=Job(namespace=self.namespace, name=f"{self.job_name}_add_documents"),
                producer="sentinai",
                inputs=[],
                outputs=[Dataset(namespace=self.namespace, name=self.collection_name)]
            )
            self.ol_client.emit(complete_event)
            logger.info(f"Lineage COMPLETE emitted: vector_store_add_documents (run_id={run_id})")
        except Exception as e:
            logger.warning(f"Failed to emit COMPLETE event: {e}")
        
    async def hybrid_search(self, query: str, filters: Optional[Dict] = None, k: int = 5) -> List[SearchResult]:
        """
        Performs hybrid search followed by cross-encoder reranking.
        """
        collection = await self.get_collection()
        
        # 1. Sparse Search (BM25)
        sparse_results = self.sparse_search.search(query, k=k)
        
        # 2. Vector Search (Dense)
        # Assuming embedding function is handled by Chroma or passed explicitly.
        if self.use_persistent_client:
            # PersistentClient is synchronous
            loop = asyncio.get_event_loop()
            dense_response = await loop.run_in_executor(
                None,
                collection.query,
                query_texts=[query],
                n_results=k,
                where=None  # where parameter
            )
        else:
            # AsyncHttpClient is asynchronous
            dense_response = await collection.query(
                query_texts=[query],
                n_results=k,
                where=filters
            )
        
        dense_results = []
        if dense_response and dense_response['documents'] and len(dense_response['documents']) > 0:
            docs = dense_response['documents'][0]
            metas = dense_response['metadatas'][0]
            ids = dense_response['ids'][0]
            for d, m, i in zip(docs, metas, ids):
                dense_results.append({
                    "id": i,
                    "content": d,
                    "metadata": m
                })
        
        # Merge results for RRF (Reciprocal Rank Fusion)
        # For simplicity, we just combine and deduplicate based on IDs
        merged_map = {res['id']: res for res in dense_results}
        for res in sparse_results:
            if res['id'] not in merged_map:
                merged_map[res['id']] = res
                
        merged_results = list(merged_map.values())
        
        # 3. Rerank
        reranked_results = await self.reranker.rerank(query, merged_results)
        
        # Format as SearchResult Pydantic models for Explainability & Auditability
        final_results = []
        for doc in reranked_results[:k]:
            meta = doc['metadata']
            
            # Reconstruct ChunkMetadata
            chunk_metadata = ChunkMetadata(
                source_origin=meta.get('source_origin', 'unknown'),
                page_number=meta.get('page_number', None),
                document_version=meta.get('document_version', '1.0'),
                retention_policy_id=meta.get('retention_policy_id', 'unknown'),
                sensitivity_label=meta.get('sensitivity_label', 'unknown'),
                metadata_hash=meta.get('metadata_hash', 'unknown')
            )
            
            # Retrieved Source IDs are tracked via source_origin and the doc id
            source_id = f"{chunk_metadata.source_origin}_{doc['id']}"
            
            result = SearchResult(
                content=doc['content'],
                metadata=chunk_metadata,
                confidence_score=doc['confidence_score'],
                source_id=source_id
            )
            final_results.append(result)
            
        # Log for Explainability
        source_ids = [res.source_id for res in final_results]
        logger.info(f"Hybrid Search completed. Retrieved Source IDs: {source_ids}")
        
        return final_results
