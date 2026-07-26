# -*- coding: utf-8 -*-
"""Tools available to agents: retrieval + calculator + current_date."""
from __future__ import annotations
from langchain_core.tools import tool

from app.rag.embedding import get_embedding_service
from app.rag.vector_store import ChromaStore
from app.rag.bm25_store import BM25Store
from app.rag.reranker import CrossEncoderReranker
from app.rag.retriever import rrf_fuse
from app.core.config import settings


def build_tools(kb_id="default"):
    embedding = get_embedding_service()
    vector_store = ChromaStore(collection_name="kb_" + kb_id)
    bm25_store = BM25Store(persist_path="./data/bm25_" + kb_id + ".pkl")
    try:
        bm25_store.load()
    except Exception:
        pass
    reranker = CrossEncoderReranker()

    @tool
    def hybrid_search(query: str, top_k: int = 5) -> str:
        "从企业知识库混合检索（BM25+向量+RRF 重排）最相关的内容。"
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
            doc = r.get("document", "")
            snippet = doc[:300].replace("\n", " ")
            score = r.get("rerank_score", 0)
            lines.append("[" + str(i) + "] (score=" + format(score, ".3f") + ") " + snippet)
        return "\n".join(lines)

    @tool
    def calculator(expression: str) -> str:
        "安全计算数学表达式。"
        import math
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})
        try:
            return str(eval(expression, {"__builtins__": {}}, allowed))
        except Exception as e:
            return "计算失败: " + str(e)

    @tool
    def current_date() -> str:
        "获取当前日期时间。"
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return [hybrid_search, calculator, current_date]
