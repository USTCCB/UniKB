"""Chat / Q&A schemas."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    kb_id: str = "default"
    session_id: Optional[str] = None
    mode: Literal["rag", "agent"] = "rag"
    top_k: int = Field(default=5, ge=1, le=20)
    stream: bool = False


class SourceItem(BaseModel):
    doc_id: str
    chunk_id: str
    content: str
    score: float
    metadata: dict = {}


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[SourceItem] = []
    agent_trace: Optional[list[dict]] = None
    usage: dict = {}


class ChatChunk(BaseModel):
    """SSE chunk."""
    type: Literal["token", "source", "done", "error"] = "token"
    data: dict = {}
