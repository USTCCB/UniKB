"""混合检索：BM25 + 向量 + RRF 融合。"""
from __future__ import annotations

import asyncio
import threading
from typing import Dict, List

from app.core.config import settings
from app.rag.bm25_store import BM25Store
from app.rag.embedding import get_embedding_service
from app.rag.vector_store import ChromaStore


# 进程内 HybridRetriever 实例缓存 (按 kb_id).
# 同一 kb 多次请求复用同一 ChromaStore/BM25Store, 避免每次新建 collection/重建 BM25.
_INSTANCE_CACHE: Dict[str, "HybridRetriever"] = {}
_INSTANCE_LOCK = threading.Lock()


def get_retriever(kb_id: str = "default") -> "HybridRetriever":
    """获取/创建 HybridRetriever 实例. 同 kb_id 永远返回同一实例.

    线程安全: 用 _INSTANCE_LOCK 串行化创建, 避免 double-init race.
    测试时可调 reset_retriever_cache() 清空.
    """
    with _INSTANCE_LOCK:
        r = _INSTANCE_CACHE.get(kb_id)
        if r is not None:
            return r
        r = HybridRetriever(kb_id=kb_id)
        _INSTANCE_CACHE[kb_id] = r
        return r


def reset_retriever_cache() -> None:
    """仅供测试: 清空实例缓存."""
    with _INSTANCE_LOCK:
        _INSTANCE_CACHE.clear()


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
        # 纵深防御: kb_id 会拼进文件路径与 collection 名, API 层已经用
        # kb_registry.validate_kb_id 拦过一次, 这里再兜一次, 保证 Agent / 评估脚本
        # 等非 HTTP 调用方也不能构造出 "../../x" 这种路径.
        from app.core.kb_registry import is_valid_kb_id

        if not is_valid_kb_id(kb_id):
            raise ValueError(
                f"非法的 kb_id: {kb_id!r} (只允许字母、数字、下划线和连字符, 长度 1-64)"
            )
        kb_id = kb_id.strip()
        self.kb_id = kb_id
        self.vector_store = ChromaStore(collection_name=f"kb_{kb_id}")
        self.bm25_store = BM25Store(persist_path=f"./data/bm25_{kb_id}.pkl")
        # 启动时尝试恢复 BM25
        try:
            self.bm25_store.load()
        except Exception:
            pass
        self.embedding = get_embedding_service()
        # 写并发锁: 上传 (add_documents) 与删除 (delete_documents) 都涉及
        # embedding + 向量 upsert + BM25 add/delete, 串行化避免数据竞争.
        # 读 (retrieve) 不需要这把锁 (Chroma query 自身线程安全; BM25 _ensure_index
        # 内部有 _rebuild_lock).
        self._write_lock = threading.Lock()

    def add_documents(self, ids: List[str], documents: List[str], metadatas: List[dict]):
        # CPU 密集: embedding + 向量写入 + BM25 add. 同步接口.
        # 整个流程加锁, 避免两个上传线程同时改 BM25+Chroma.
        with self._write_lock:
            embeddings = self.embedding.embed(documents)
            self.vector_store.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
            self.bm25_store.add(ids=ids, documents=documents, metadatas=metadatas)

    async def add_documents_async(self, ids: List[str], documents: List[str], metadatas: List[dict]) -> None:
        """async 包装: 在 async 上下文里调用, 不会阻塞事件循环.

        用 asyncio.to_thread 把 CPU 密集型 embedding/序列化
        卸到默认 threadpool, 让 FastAPI 的事件循环能继续处理其他请求.
        """
        return await asyncio.to_thread(self.add_documents, ids, documents, metadatas)

    def delete_documents(self, ids: List[str]) -> int:
        """按 doc_id 删除 (含其下所有 chunk). 返回实际删除的 chunk 数."""
        with self._write_lock:
            n_vec = self.vector_store.delete(ids)
            n_bm25 = self.bm25_store.delete(ids)
        return max(n_vec, n_bm25)

    def delete_by_doc_id(self, doc_id: str) -> int:
        """按 metadata.doc_id 删除整个文档的所有 chunk.

        先从 BM25 内存 docs 中收集 chunk id (O(N) 但无需重建索引),
        再同步删除 vector store 中同 doc_id 的 chunk.
        删除后 BM25 dirty=True, 下次 query 按需惰性重建.
        """
        with self._write_lock:
            ids_to_delete = [d["id"] for d in self.bm25_store.docs if d.get("metadata", {}).get("doc_id") == doc_id]
            if not ids_to_delete:
                # BM25 可能为空或尚未加载, 回退到 vector store 按 metadata 查询
                ids_to_delete = self.vector_store.get_ids_by_doc_id(doc_id)
            if ids_to_delete:
                self.vector_store.delete(ids_to_delete)
                self.bm25_store.delete(ids_to_delete)
        return len(ids_to_delete)

    def _sibling_chunks(self, doc_id: str, exclude_ids: set) -> List[dict]:
        """从 BM25 内存 docs 取同 doc_id 的兄弟 chunk (排除已召回的).

        BM25Store.docs 始终持有 {id, text, metadata}, 真实/ Fake 实现通用.
        返回带 rrf_score 占位 None 的 dict, 由调用方决定排序权重.
        """
        out = []
        for d in self.bm25_store.docs:
            md = d.get("metadata", {}) or {}
            if md.get("doc_id") != doc_id:
                continue
            if d["id"] in exclude_ids:
                continue
            out.append({
                "id": d["id"],
                "document": d.get("text", ""),
                "metadata": md,
                "rrf_score": None,  # 占位, 后续按父 chunk 分数继承
            })
        return out

    def expand_parent_recall(
        self,
        fused: List[dict],
        max_docs: int = 4,
        max_extra_per_doc: int = 6,
    ) -> List[dict]:
        """父文档召回 (连坐召回).

        当 top chunk 被召回时, 把**同一文档**的兄弟 chunk 一并带上, 保证
        喂给 LLM 的上下文是完整的段落/章节, 而不是被切碎的孤立句子.

        实现:
          - 仅对 fused 前 `max_docs` 个 doc_id 做连坐 (控成本).
          - 每个 doc_id 最多补 `max_extra_per_doc` 个兄弟 chunk.
          - 兄弟 chunk 继承其父 chunk 的 rrf_score, 排在其后、其他文档之前.
        """
        if not fused:
            return fused
        recalled_ids = {c["id"] for c in fused}
        expanded: List[dict] = list(fused)
        seen: set = set(recalled_ids)
        for parent in fused[:max_docs]:
            doc_id = (parent.get("metadata", {}) or {}).get("doc_id")
            if not doc_id:
                continue
            parent_score = parent.get("rrf_score", 0.0) or 0.0
            sibs = self._sibling_chunks(doc_id, seen)
            for s in sibs[:max_extra_per_doc]:
                s["rrf_score"] = parent_score * 0.99  # 紧随父 chunk 之后
                expanded.append(s)
                seen.add(s["id"])
        return expanded

    def retrieve(self, query: str, top_k: int | None = None) -> List[dict]:
        # CPU 密集: embedding + 两次召回 + RRF. 同步接口.
        # 读不需要锁; BM25 内部 _ensure_index 有 _rebuild_lock.
        top_k = top_k or settings.top_k_final
        qv = self.embedding.embed_query(query)
        vec_hits = self.vector_store.query(qv, top_k=settings.top_k_vector)
        bm25_hits = self.bm25_store.query(query, top_k=settings.top_k_bm25)
        fused = rrf_fuse([vec_hits, bm25_hits], k=60)
        # 候选池: 先取 top_k 个做 RRF 融合结果.
        fused_top = fused[: top_k * 2]
        # 父文档召回 (连坐): 把同文档兄弟 chunk 补回, 提升上下文完整度.
        if getattr(settings, "parent_recall_enabled", False):
            fused_top = self.expand_parent_recall(
                fused_top, max_docs=4, max_extra_per_doc=settings.parent_recall_extra
            )
        return fused_top

    async def retrieve_async(self, query: str, top_k: int | None = None) -> List[dict]:
        """async 包装: 在 async 上下文里调用, 不会阻塞事件循环."""
        return await asyncio.to_thread(self.retrieve, query, top_k)
