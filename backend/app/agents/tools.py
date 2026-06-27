"""Tools available to agents: retrieval + web search (mock) + calculator."""
from __future__ import annotations

from typing import List

from langchain_core.tools import tool

from app.rag.embedding import get_embedding_service
from app.rag.vector_store import ChromaStore
from app.rag.bm25_store import BM25Store
from app.rag.reranker import CrossEncoderReranker
from app.rag.retriever import rrf_fuse
from app.core.config import settings


def build_tools(kb_id: str = "default"):
    """Build tool set bound to a specific knowledge base."""
    embedding = get_embedding_service()
    vector_store = ChromaStore(collection_name=f"kb_{kb_id}")
    bm25_store = BM25Store(persist_path=f"./data/bm25_{kb_id}.pkl")
    try:
        bm25_store.load()
    except Exception:
        pass
    reranker = CrossEncoderReranker()

    @tool
    def hybrid_search(query: str, top_k: int = 5) -> str:
        """从企业知识库混合检索（BM25+向量+RRF 重排）最相关的内容。"""
        qv = embedding.embed_query(query)
        vec_hits = vector_store.query(qv, top_k=settings.top_k_vector)
        bm25_hits = bm25_store.query(query, top_k=settings.top_k_bm25)
        fused = rrf_fuse([vec_hits, bm25_hits], k=60)
        candidates = fused[: max(10, top_k * 2)]
        reranked = reranker.rerank(query, candidates, top_k=top_k)
        if not reranked:
            return "未检索到相关文档。"
        lines = []
        for i, r in enumerate(reranked, 1):
            snippet = (r.get("document") or "")[:300].replace("\n", " ")
            lines.append(f"[{i}] (score={r.get(