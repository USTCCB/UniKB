"""Cross-Encoder 重排序，抑制幻觉。"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Tuple

from loguru import logger

from app.core.config import settings


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.rerank_model
        self._model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading rerank model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: List[dict], top_k: int = 5) -> List[dict]:
        """candidates: [{id, document, metadata, score|distance}]"""
        if not candidates:
            return []
        try:
            m = self._ensure()
            pairs = [(query, c.get("document", "")) for c in candidates]
            scores = m.predict(pairs, show_progress_bar=False)
            for c, s in zip(candidates, scores):
                c["rerank_score"] = float(s)
            ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
            return ranked[:top_k]
        except Exception as e:
            logger.warning(f"Rerank failed, fallback to original order: {e}")
            return candidates[:top_k]


@lru_cache(maxsize=4)
def get_reranker(model_name: Optional[str] = None) -> CrossEncoderReranker:
    """获取 CrossEncoderReranker 单例.

    `model_name` 是缓存 key 的一部分, 换模型才会新建实例; 同一模型下始终
    复用同一对象, 避免每次请求都从磁盘/HuggingFace 重新加载. 第一次调用
    时会触发 `CrossEncoderReranker._ensure()`, 之后 hot path 就是普通的
    forward (`m.predict`).

    类似 `embedding.get_embedding_service` 和 `retriever.get_retriever`,
    整个 codebase 都用 module-level 单例. 测试时可调 `reset_reranker_cache()`
    或直接 monkey-patch `CrossEncoderReranker`.
    """
    return CrossEncoderReranker(model_name)


def reset_reranker_cache() -> None:
    """仅供测试: 清空单例缓存."""
    get_reranker.cache_clear()
