"""Chat API: 智能问答 + 流式 SSE。"""
from __future__ import annotations

import json
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from loguru import logger

from app.agents.graph import build_agent_graph
from app.agents.llm_router import get_llm
from app.api.deps import get_current_user
from app.rag.retriever import HybridRetriever
from app.rag.reranker import CrossEncoderReranker
from app.schemas.chat import ChatRequest, ChatResponse, SourceItem

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


def _build_prompt(question: str, contexts: list[dict]) -> str:
    ctx_text = "\n\n".join(
        f"[{i+1}] {c.get(