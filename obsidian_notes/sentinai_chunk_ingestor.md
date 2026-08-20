# SentinAI Chunk Ingestor

The Chunk Ingestor is the pipeline component responsible for ingesting regulatory documents (CBK guidelines, FATF recommendations) and splitting them into semantically meaningful chunks for embedding and retrieval.

## Features
- Markdown-aware chunking preserving document structure
- Overlapping chunk strategy for context continuity
- Vector embedding storage for semantic search
- Metadata tagging per chunk (source, section, page)

## Processing Pipeline
1. Load raw document text
2. Split into chunks by heading/section boundaries
3. Generate embeddings via sentence-transformers
4. Store in vector database with metadata
5. Enable retrieval-augmented SAR generation
