import hashlib
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator

class ChunkMetadata(BaseModel):
    """
    Metadata for a document chunk to ensure auditability and governance.
    """
    source_origin: str = Field(..., description="The original source of the document (e.g., S3 URI, file path).")
    page_number: Optional[int] = Field(None, description="The page number where the chunk originated.")
    document_version: str = Field(default="1.0", description="Version of the document for audit trails.")
    retention_policy_id: str = Field(..., description="Governance tag for data retention.")
    sensitivity_label: str = Field(..., description="Data sensitivity level (e.g., Public, Internal, Confidential).")
    metadata_hash: str = Field(..., description="SHA-256 hash of the content and metadata for integrity verification.")

class ChunkModel(BaseModel):
    """
    Represents a semantically chunked piece of text with its associated metadata.
    """
    content: str = Field(..., description="The actual text content of the chunk.")
    metadata: ChunkMetadata = Field(..., description="Audit and governance metadata.")
    
    @classmethod
    def create(
        cls, 
        content: str, 
        source_origin: str, 
        retention_policy_id: str, 
        sensitivity_label: str,
        page_number: Optional[int] = None, 
        document_version: str = "1.0"
    ) -> "ChunkModel":
        """Helper to create a chunk model and automatically compute its metadata hash."""
        # Compute a simple hash based on content and source for integrity
        hash_input = f"{content}|{source_origin}|{page_number}|{document_version}".encode('utf-8')
        metadata_hash = hashlib.sha256(hash_input).hexdigest()
        
        metadata = ChunkMetadata(
            source_origin=source_origin,
            page_number=page_number,
            document_version=document_version,
            retention_policy_id=retention_policy_id,
            sensitivity_label=sensitivity_label,
            metadata_hash=metadata_hash
        )
        return cls(content=content, metadata=metadata)

class SearchResult(BaseModel):
    """
    Represents a retrieved document chunk after ranking.
    """
    content: str = Field(..., description="The text content retrieved.")
    metadata: ChunkMetadata = Field(..., description="The original chunk metadata.")
    confidence_score: float = Field(..., description="The reranking confidence score.")
    source_id: str = Field(..., description="The retrieved source ID (usually source_origin or a composite key).")
