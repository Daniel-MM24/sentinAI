"""Tools for agents to interact with external systems."""
import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from src.retrieval.vector_db import VectorStore

logger = logging.getLogger(__name__)

# Global vector store instance (initialized in graph.py)
_vector_store: Optional[VectorStore] = None


def set_vector_store(store: VectorStore) -> None:
    """Set the global vector store instance for tools."""
    global _vector_store
    _vector_store = store


@tool
async def query_vector_db(query: str, k: int = 3) -> str:
    """
    Query the vector database for relevant compliance documents.

    Args:
        query: The search query for retrieving relevant documents.
        k: Number of results to return (default: 3).

    Returns:
        A formatted string containing the retrieved context with source citations.
    """
    if _vector_store is None:
        logger.warning("Vector store not initialized")
        return "Error: Vector store not initialized."

    try:
        results = await _vector_store.hybrid_search(query=query, k=k)

        if not results:
            return "No relevant documents found."

        formatted_context = []
        for idx, result in enumerate(results, 1):
            formatted_context.append(
                f"[{idx}] Source: {result.source_id}\n"
                f"Content: {result.content}\n"
                f"Confidence: {result.confidence_score:.3f}\n"
                f"Metadata: {result.metadata.source_origin} (page {result.metadata.page_number})"
            )

        return "\n\n".join(formatted_context)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Vector DB query failed: %s", exc)
        return f"Error querying vector database: {exc}"
