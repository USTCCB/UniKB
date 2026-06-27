"""Chroma vector store wrapper."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from loguru import logger

from app.core.config import settings
from app.rag.embedding import get_embedding_service


class ChromaStore:
    def __init__(self, persist_dir: str | None = None, collection_name: str = "default"):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._client = None
        self._col = None

    def _ensure(self):
        if self._client is None:
            import chromadb
            from chromadb.config import Settings as Cfg
            self._client = chromadb.PersistentClient(path=self.persist_dir, settings=Cfg(anonymized_telemetry=False))
            self._col = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._col

    def add(self, ids: List[str], documents: List[str], embeddings: List[List[float]], metadatas: List[dict]):
        col = self._ensure()
        col.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        logger.info(f"Chroma upsert: {len(ids)} vectors into {self.collection_name}")

    def query(self, query_embedding: List[float], top_k: int = 20, where: Optional[dict] = None):
        col = self._ensure()
        kwargs = {"query_embeddings": [query_embedding], "n_results": top_k}
        if where:
            kwargs["where"] = where
        res = col.query(**kwargs)
        # 返回统一结构
        out = []
        if not res or not res.get("ids"):
            return out
        for i, _id in enumerate(res["ids"][0]):
            out.append({
                "id": _id,
                "document": res["documents"][0][i] if res.get("documents") else "",
                "metadata": res["metadatas"][0][i] if res.get("metadatas") else {},
                "distance": res["distances"][0][i] if res.get("distances") else 0.0,
            })
        return out

    def count(self) -> int:
        col = self._ensure()
        return col.count()

    def reset(self):
        col = self._ensure()
        # 删除再重建
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._col = None
