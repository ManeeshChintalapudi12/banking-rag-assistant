from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: Optional[int] = Field(default=None, ge=1, le=10)

    @field_validator("question", mode="before")
    @classmethod
    def _strip_question(cls, value):
        # Strip before the length constraints run, so a question made
        # only of whitespace (e.g. "   ") fails min_length instead of
        # reaching retrieval, and padded questions are answered as-is.
        if isinstance(value, str):
            return value.strip()
        return value


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
