"""Embedding service: sentence-transformers (default) + OpenAI (optional)."""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from typing import List

from loguru import logger

from app.core.config import settings


class EmbeddingService:
    """维度：bge-small-zh-v1.5 -> 512, bge-base-zh-v1.5 -> 768, OpenAI text-embedding-3-small -> 1536"""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self._model = None
        self.dim = 512  # default for bge-small

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self.dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        m = self._ensure_model()
        vecs = m.encode(texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        return vecs.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]


class FakeEmbeddingService:
    """deterministic embedding: 字符哈希到 64 维, 不依赖 torch / sentence-transformers.

    用在 CI / 沙箱 / 评估脚本验证链路通断. 行为类似 bge-small, 但语义质量远不如真模型.
    """

    dim = 64

    def _vec(self, text: str) -> List[float]:
        v = [0.0] * self.dim
        for ch in text:
            idx = int(hashlib.md5(ch.encode()).hexdigest(), 16) % self.dim
            v[idx] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)


@lru_cache
def get_embedding_service() -> "EmbeddingService | FakeEmbeddingService":
    if os.environ.get("UNIKB_FAKE_EMBEDDING", "").lower() in ("1", "true", "yes"):
        logger.info("Using FakeEmbeddingService (UNIKB_FAKE_EMBEDDING=1)")
        return FakeEmbeddingService()
    return EmbeddingService()
