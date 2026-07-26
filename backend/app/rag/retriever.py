"""混合检索：BM25 + 向量 + RRF 融合。"""
from __future__ import annotations

import asyncio
from typing import List

from loguru import logger

from app.core.config import settings
from app.rag.bm25_store import BM25Store
from app.rag.embedding import get_embedding_service
from app.rag.vector_store import ChromaStore


def rrf_fuse(rank_lists: List[List[dict]], k: int = 60) -> List[dict]:
    """Reciprocal Rank Fusion 融合多路召回。
    公式：score(d) = sum( 1 / (k + rank_i(d)) )，k 默认 60（标准做法）。"""
    scores: dict = {}
    meta: dict = {}
    for rl in rank_lists:
        for rank, item in enumerate(rl, start=1):
            d_id = item["id"]
            scores[d_id] = scores.get(d_id, 0.0) + 1.0 / (k + rank)
            if d_id not in meta:
                meta[d_id] = item
            else:
                # 保留文本/metadata，score 取最大
                if item.get("rerank_score", 0) > meta[d_id].get("rerank_score", 0):
                    meta[d_id].update(item)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    out = []
    for d_id, s in ranked:
        item = dict(meta[d_id])
        item["rrf_score"] = s
        out.append(item)
    return out


class HybridRetriever:
    def __init__(self, kb_id: str = "default"):
        self.kb_id = kb_id
        self.vector_store = ChromaStore(collection_name=f"kb_{kb_id}")
        self.bm25_store = BM25Store(persist_path=f"./data/bm25_{kb_id}.pkl")
        # 启动时尝试恢复 BM25
        try:
            self.bm25_store.load()
        except Exception:
            pass
        self.embedding = get_embedding_service()

    def add_documents(self, ids: List[str], documents: List[str], metadatas: List[dict]):
        # CPU 密集: embedding + 向量写入 + BM25 重建. 同步接口.
        embeddings = self.embedding.embed(documents)
        self.vector_store.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        self.bm25_store.add(ids=ids, documents=documents, metadatas=metadatas)

    async def add_documents_async(self, ids: List[str], documents: List[str], metadatas: List[dict]) -> None:
        """async 包装: 在 async 上下文里调用, 不会阻塞事件循环.

        用 asyncio.to_thread 把 CPU 密集型 embedding/序列化
        卸到默认 threadpool, 让 FastAPI 的事件循环能继续处理其他请求.
        """
        return await asyncio.to_thread(self.add_documents, ids, documents, metadatas)

    def retrieve(self, query: str, top_k: int | None = None) -> List[dict]:
        # CPU 密集: embedding + 两次召回 + RRF. 同步接口.
        top_k = top_k or settings.top_k_final
        qv = self.embedding.embed_query(query)
        vec_hits = self.vector_store.query(qv, top_k=settings.top_k_vector)
        bm25_hits = self.bm25_store.query(query, top_k=settings.top_k_bm25)
        fused = rrf_fuse([vec_hits, bm25_hits], k=60)
        return fused[: top_k * 2]

    async def retrieve_async(self, query: str, top_k: int | None = None) -> List[dict]:
        """async 包装: 在 async 上下文里调用, 不会阻塞事件循环."""
        return await asyncio.to_thread(self.retrieve, query, top_k)
