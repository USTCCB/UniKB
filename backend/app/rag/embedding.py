"""Embedding service: sentence-transformers (default) + OpenAI (optional)."""
from __future__ import annotations

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


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
