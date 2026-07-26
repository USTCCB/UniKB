"""In-memory BM25 store (rank_bm25). 适合小到中等规模。生产可换 Elasticsearch / OpenSearch。"""
from __future__ import annotations

import pickle
import re
import threading
from pathlib import Path
from typing import List

from loguru import logger


def _tokenize(text: str) -> List[str]:
    # 中英混合简单切分：按非字母数字汉字切
    text = text.lower()
    tokens = re.findall(r"[一-龥]+|[a-z0-9]+", text)
    return tokens


class BM25Store:
    def __init__(self, persist_path: str | None = None):
        self.persist_path = persist_path
        self._bm25 = None
        self.docs: List[dict] = []  # 每条: {id, text, metadata}
        # 惰性重建: 写入/删除只动 docs + 置 dirty=True, 不立即 rebuild.
        # 首次 query() 时若 dirty 才重建. 大幅降低 N 次上传的 CPU 开销.
        self._dirty: bool = False
        # 重建成本较高, 用锁串行化 (只允许一个线程在 rebuild).
        self._rebuild_lock = threading.Lock()
        # 重建节流: 阈值以下不重建; 让 query 直接返回空 (避免 1 条文档都触发重建).
        # 设为 1 (默认有内容才建).
        self._min_docs_to_build: int = 1

    def add(self, ids: List[str], documents: List[str], metadatas: List[dict]):
        """追加 doc + 标 dirty. 不立即重建, query 时按需重建."""
        for i, d in enumerate(documents):
            self.docs.append({"id": ids[i], "text": d, "metadata": metadatas[i]})
        self._dirty = True
        self._maybe_persist()

    def delete(self, ids: List[str]) -> int:
        """按 id 删除. 返回实际删除的条数. 不存在的不报错."""
        if not ids:
            return 0
        ids_set = set(ids)
        kept: List[dict] = []
        removed = 0
        for d in self.docs:
            if d["id"] in ids_set:
                removed += 1
                continue
            kept.append(d)
        if removed:
            self.docs = kept
            self._dirty = True
            self._maybe_persist()
        return removed

    def query(self, text: str, top_k: int = 20) -> List[dict]:
        if not self.docs:
            return []
        self._ensure_index()
        if not self._bm25:
            return []
        tokens = _tokenize(text)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        out = []
        for idx, score in ranked:
            d = self.docs[idx]
            out.append({
                "id": d["id"],
                "document": d["text"],
                "metadata": d["metadata"],
                "score": float(score),
            })
        return out

    def count(self) -> int:
        return len(self.docs)

    def rebuild(self, force: bool = False) -> None:
        """显式重建 (例如大批量入库后想马上能 query). 强制覆盖 dirty."""
        with self._rebuild_lock:
            self._do_build()
            self._dirty = False

    def _ensure_index(self) -> None:
        """仅在 dirty 时重建, 锁内串行."""
        if not self._dirty:
            return
        with self._rebuild_lock:
            # 双重检查: 上一个等待者可能已经重建完.
            if not self._dirty:
                return
            if len(self.docs) < self._min_docs_to_build:
                self._bm25 = None
                self._dirty = False
                return
            self._do_build()
            self._dirty = False

    def _do_build(self) -> None:
        from rank_bm25 import BM25Okapi
        tokenized_corpus = [_tokenize(d["text"]) for d in self.docs]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index built: {len(self.docs)} docs")

    def _maybe_persist(self):
        if not self.persist_path:
            return
        try:
            Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
            # 只持久化 docs + dirty 标记; _bm25 反序列化时按需重建.
            with open(self.persist_path, "wb") as f:
                pickle.dump({"docs": self.docs}, f)
        except Exception as e:
            logger.warning(f"BM25 persist failed: {e}")

    def load(self):
        if not self.persist_path or not Path(self.persist_path).exists():
            return
        with open(self.persist_path, "rb") as f:
            data = pickle.load(f)
        self.docs = data["docs"]
        self._dirty = True  # 启动后第一次 query 触发重建