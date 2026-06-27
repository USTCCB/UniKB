"""In-memory BM25 store (rank_bm25). 适合小到中等规模。生产可换 Elasticsearch / OpenSearch。"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import List

from loguru import logger


def _tokenize(text: str) -> List[str]:
    # 中英混合简单切分：按非字母数字汉字切
    text = text.lower()
    tokens = re.findall(r"[\u4e00-\u9fa5]+|[a-z0-9]+", text)
    return tokens


class BM25Store:
    def __init__(self, persist_path: str | None = None):
        self.persist_path = persist_path
        self._bm25 = None
        self.docs: List[dict] = []  # 每条: {id, text, metadata}

    def add(self, ids: List[str], documents: List[str], metadatas: List[dict]):
        from rank_bm25 import BM25Okapi
        for i, d in enumerate(documents):
            self.docs.append({"id": ids[i], "text": d, "metadata": metadatas[i]})
        tokenized_corpus = [_tokenize(d["text"]) for d in self.docs]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index built: {len(self.docs)} docs")
        self._maybe_persist()

    def query(self, text: str, top_k: int = 20) -> List[dict]:
        if not self._bm25 or not self.docs:
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

    def _maybe_persist(self):
        if not self.persist_path:
            return
        try:
            Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
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
        from rank_bm25 import BM25Okapi
        tokenized_corpus = [_tokenize(d["text"]) for d in self.docs]
        self._bm25 = BM25Okapi(tokenized_corpus)
