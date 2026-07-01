import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval.ingestion import DocumentIngestor
from src.retrieval.vector_db import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main ingestion pipeline for M-Pesa risk knowledge base."""
    
    # Configuration
    seed_path = "./src/retrieval/seed_data.json"
    persist_directory = "./src/retrieval/chroma_storage"
    collection_name = "mpesa_risk_knowledge"
    cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    logger.info("=" * 60)
    logger.info("Starting SentinAI Retrieval Ingestion Pipeline")
    logger.info("=" * 60)
    
    # Validate seed file exists
    seed_file = Path(seed_path)
    if not seed_file.exists():
        logger.error(f"Seed file not found: {seed_path}")
        sys.exit(1)
    
    # Initialize components
    logger.info("Initializing DocumentIngestor...")
    ingestor = DocumentIngestor()
    
    logger.info(f"Initializing VectorStore with persist_directory={persist_directory}")
    logger.info("Using PersistentClient (local-only mode)")
    vector_store = VectorStore(
        persist_directory=persist_directory,
        collection_name=collection_name,
        cross_encoder_model=cross_encoder_model,
        use_persistent_client=True  # Use local persistent mode
    )
    
    # Load seed data
    logger.info(f"Loading seed data from {seed_path}...")
    chunks = ingestor.load_markdown_seed(
        seed_path=seed_path,
        retention_policy_id="regulatory_retention",
        sensitivity_label="Confidential"
    )
    
    if not chunks:
        logger.error("No chunks loaded from seed data. Aborting.")
        sys.exit(1)
    
    logger.info(f"Successfully loaded {len(chunks)} document chunks")
    
    # Generate embeddings
    logger.info(f"Generating embeddings for {len(chunks)} documents...")
    try:
        embedded_docs = await ingestor.embed_chunks_resilient(chunks, batch_size=100)
        logger.info(f"Successfully generated embeddings for {len(embedded_docs)} documents")
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        sys.exit(1)
    
    # Prepare documents for ChromaDB
    logger.info("Preparing documents for ChromaDB insertion...")
    documents = []
    for item in embedded_docs:
        chunk = item["chunk"]
        embedding = item["embedding"]
        documents.append({
            "id": chunk.metadata.metadata_hash,
            "content": chunk.content,
            "metadata": chunk.metadata.model_dump(),
            "embedding": embedding
        })
    
    logger.info(f"Prepared {len(documents)} documents for insertion")
    
    # Add to vector store
    logger.info(f"Persisting {len(documents)} documents to ChromaDB collection '{collection_name}'...")
    try:
        await vector_store.add_documents(documents)
        logger.info("Successfully persisted documents to ChromaDB")
    except Exception as e:
        logger.error(f"Failed to persist documents to ChromaDB: {e}")
        sys.exit(1)
    
    # Completion summary
    logger.info("=" * 60)
    logger.info("Ingestion completed successfully!")
    logger.info(f"Successfully vectorized {len(documents)} documents to '{collection_name}' collection")
    logger.info(f"ChromaDB storage location: {persist_directory}")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Ingestion interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
