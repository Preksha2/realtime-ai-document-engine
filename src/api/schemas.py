"""
Pydantic models for API request/response validation.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for document query endpoint."""
    question: str = Field(..., min_length=1, max_length=2000, description="Natural language question")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of chunks to retrieve")


class SourceInfo(BaseModel):
    """Metadata about a retrieved source chunk."""
    filename: str
    chunk_index: int
    score: float
    preview: str


class QueryResponse(BaseModel):
    """Response body for document query endpoint."""
    answer: str
    sources: List[SourceInfo]
    chunks_retrieved: int
    chunks_used: int
    model: str


class HealthResponse(BaseModel):
    """Response body for health check endpoint."""
    status: str
    index_loaded: bool
    total_chunks: int


class IndexRequest(BaseModel):
    """Request body for triggering document indexing."""
    source_dir: str = Field(..., description="Path to directory containing documents")
    save_path: Optional[str] = Field("data/faiss_index.bin", description="Path to save the built index")


class IndexResponse(BaseModel):
    """Response body for indexing endpoint."""
    status: str
    chunks_indexed: int
    index_path: str
