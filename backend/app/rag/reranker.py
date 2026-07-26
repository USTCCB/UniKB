"""Cross-Encoder 重排序，抑制幻觉。"""
from __future__ import annotations

from typing import List, Tuple

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
