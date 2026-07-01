"""
Verification script for SentinAI Retrieval Ingestion Pipeline.

This script should be run AFTER installing dependencies to verify:
1. ChromaDB collection contains 12 documents
2. Embeddings were successfully persisted
3. Hybrid search functionality works correctly

Prerequisites:
- pip install chromadb sentence-transformers langchain-text-splitters langchain-community tenacity

Usage:
    python src/retrieval/verify_ingestion.py
"""
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval.vector_db import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def verify_collection_count():
    """Verify the ChromaDB collection contains 12 documents."""
    persist_directory = "./src/retrieval/chroma_storage"
    collection_name = "mpesa_risk_knowledge"
    
    logger.info("=" * 60)
    logger.info("Verifying ChromaDB Collection")
    logger.info("=" * 60)
    
    vector_store = VectorStore(
        persist_directory=persist_directory,
        collection_name=collection_name,
        use_persistent_client=True
    )
    
    try:
        collection = await vector_store.get_collection()
        count = collection.count()
        
        logger.info(f"Collection '{collection_name}' contains {count} documents")
        
        if count == 12:
            logger.info("✓ Document count matches expected (12)")
            return True
        else:
            logger.warning(f"✗ Expected 12 documents, found {count}")
            return False
    except Exception as e:
        logger.error(f"Failed to verify collection: {e}")
        return False


async def test_hybrid_search():
    """Test hybrid search functionality with a sample query."""
    persist_directory = "./src/retrieval/chroma_storage"
    collection_name = "mpesa_risk_knowledge"
    
    logger.info("=" * 60)
    logger.info("Testing Hybrid Search")
    logger.info("=" * 60)
    
    vector_store = VectorStore(
        persist_directory=persist_directory,
        collection_name=collection_name,
        use_persistent_client=True
    )
    
    test_queries = [
        "What are the AML transaction limits for Tier 1 customers?",
        "How is cuckoo smurfing detected in M-Pesa transactions?",
        "What are the penalties for non-compliance with CBK regulations?"
    ]
    
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        try:
            results = await vector_store.hybrid_search(query, k=3)
            logger.info(f"Retrieved {len(results)} results")
            
            for i, result in enumerate(results, 1):
                logger.info(f"  Result {i}:")
                logger.info(f"    Confidence: {result.confidence_score:.4f}")
                logger.info(f"    Source: {result.metadata.source_origin}")
                logger.info(f"    Content preview: {result.content[:100]}...")
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return False
    
    logger.info("\n✓ Hybrid search tests completed")
    return True


async def verify_chroma_storage():
    """Verify ChromaDB storage directory contains expected files."""
    persist_directory = Path("./src/retrieval/chroma_storage")
    
    logger.info("=" * 60)
    logger.info("Verifying ChromaDB Storage Directory")
    logger.info("=" * 60)
    
    if not persist_directory.exists():
        logger.error(f"Storage directory does not exist: {persist_directory}")
        return False
    
    logger.info(f"Storage directory exists: {persist_directory}")
    
    # List contents
    items = list(persist_directory.iterdir())
    logger.info(f"Directory contains {len(items)} items")
    
    for item in items:
        logger.info(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
    
    if len(items) > 0:
        logger.info("✓ Storage directory contains data")
        return True
    else:
        logger.warning("✗ Storage directory is empty")
        return False


async def main():
    """Run all verification checks."""
    logger.info("Starting SentinAI Retrieval Verification")
    logger.info("")
    
    # Verify storage directory
    storage_ok = await verify_chroma_storage()
    logger.info("")
    
    # Verify collection count
    count_ok = await verify_collection_count()
    logger.info("")
    
    # Test hybrid search
    search_ok = await test_hybrid_search()
    logger.info("")
    
    # Summary
    logger.info("=" * 60)
    logger.info("Verification Summary")
    logger.info("=" * 60)
    logger.info(f"Storage Directory: {'✓ PASS' if storage_ok else '✗ FAIL'}")
    logger.info(f"Document Count: {'✓ PASS' if count_ok else '✗ FAIL'}")
    logger.info(f"Hybrid Search: {'✓ PASS' if search_ok else '✗ FAIL'}")
    logger.info("=" * 60)
    
    if storage_ok and count_ok and search_ok:
        logger.info("All verification checks passed!")
        return 0
    else:
        logger.warning("Some verification checks failed")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Verification interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
