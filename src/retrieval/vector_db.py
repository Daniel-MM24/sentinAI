import asyncio
import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict
import math
import numpy as np
from sentence_transformers import CrossEncoder
import chromadb
from chromadb.config import Settings

from src.retrieval.schemas import SearchResult, ChunkMetadata

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
    """Real Cross-Encoder Reranker using sentence-transformers."""
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
        
    async def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank documents using cross-encoder model."""
        if not documents:
            return []
            
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
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Designed for persistence in a distributed environment.
        ChromaDB is used here, but can be swapped with Pinecone via similar interfaces.
        """
        self.client = chromadb.AsyncHttpClient(
            host="localhost",
            port=8000,
            settings=Settings(allow_reset=True)
        )
        self.collection_name = collection_name
        self.sparse_search = BM25Engine()
        self.reranker = CrossEncoderReranker(model_name=cross_encoder_model)
        self._indexed = False
        
    async def get_collection(self):
        # Fallback to get_or_create_collection using AsyncHttpClient
        return await self.client.get_or_create_collection(name=self.collection_name)

    async def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to both ChromaDB and BM25 index.
        """
        collection = await self.get_collection()
        
        # Prepare documents for ChromaDB
        ids = [doc.get('id', str(i)) for i, doc in enumerate(documents)]
        texts = [doc['content'] for doc in documents]
        metadatas = [doc.get('metadata', {}) for doc in documents]
        
        # Add to ChromaDB
        await collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        
        # Index for BM25
        self.sparse_search.index_documents(documents)
        self._indexed = True
        
        logger.info(f"Added {len(documents)} documents to vector store")
        
    async def hybrid_search(self, query: str, filters: Optional[Dict] = None, k: int = 5) -> List[SearchResult]:
        """
        Performs hybrid search followed by cross-encoder reranking.
        """
        collection = await self.get_collection()
        
        # 1. Sparse Search (BM25)
        sparse_results = self.sparse_search.search(query, k=k)
        
        # 2. Vector Search (Dense)
        # Assuming embedding function is handled by Chroma or passed explicitly.
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
