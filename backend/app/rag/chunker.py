"""Recursive text chunker (LangChain-style)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.core.config import settings


@dataclass
class Chunk:
    text: str
    metadata: dict


class TextChunker:
    """按段落 / 句子递归切分，保留 overlap。"""

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.separators = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""]

    def split(self, text: str, doc_id: str = "") -> List[Chunk]:
        text = (text or "").strip()
        if not text:
            return []
        splits = self._recursive_split(text, self.separators)
        chunks: List[Chunk] = []
        buf = ""
        for s in splits:
            if len(buf) + len(s) <= self.chunk_size:
                buf = (buf + s).strip()
            else:
                if buf:
                    chunks.append(Chunk(text=buf, metadata={"doc_id": doc_id}))
                if len(s) > self.chunk_size:
                    # 单段过长，硬切
                    for i in range(0, len(s), self.chunk_size - self.chunk_overlap):
                        chunks.append(Chunk(text=s[i : i + self.chunk_size], metadata={"doc_id": doc_id}))
                    buf = ""
                else:
                    buf = s.strip()
        if buf:
            chunks.append(Chunk(text=buf, metadata={"doc_id": doc_id}))
        # 注入 chunk_id
        for i, c in enumerate(chunks):
            c.metadata["chunk_id"] = f"{doc_id}_c{i}"
            c.metadata["chunk_index"] = i
        return chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if not separators or len(text) <= self.chunk_size:
            return [text]
        sep = separators[0]
        rest = separators[1:]
        if sep == "":
            # fallback: 按字符切
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        parts = text.split(sep)
        out: List[str] = []
        for p in parts:
            p = (p + sep) if sep and not p.endswith(sep) else p
            if len(p) <= self.chunk_size:
                out.append(p)
            else:
                out.extend(self._recursive_split(p, rest))
        return out
