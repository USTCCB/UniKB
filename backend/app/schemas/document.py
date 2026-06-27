"""Document schemas."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    doc_id: str
    filename: str
    kb_id: str
    chunks: int
    status: str
    created_at: datetime
    metadata: dict = {}


class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunks: int
    status: str = "indexed"
    message: Optional[str] = None
