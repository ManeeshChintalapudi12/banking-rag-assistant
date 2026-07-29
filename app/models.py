from typing import List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: Optional[int] = Field(default=None, ge=1, le=10)


class SourceChunk(BaseModel):
    chunk_id: str
    source: str
    excerpt: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    grounded: bool
    confidence: float
    sources: List[SourceChunk]


class HealthResponse(BaseModel):
    status: str
    provider: str
    index_ready: bool
