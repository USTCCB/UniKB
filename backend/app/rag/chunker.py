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
    """按段落 / 句子递归切分, 保留 overlap.

    overlap 应用规则:
    1. 主打包路径 (`buf` 累积) -- 切出当前 chunk 后, 下一个 chunk 的 buf 从
       上一 chunk 的末尾 `chunk_overlap` 个字符开始, 而不是清空, 让相邻
       chunk 之间有真正重叠, 改善边界附近的召回.
    2. "硬切" (单段超过 chunk_size) 路径 -- 循环用 `chunk_size - chunk_overlap`
       作为 step, 保证硬切的相邻片之间也有 overlap.

    注: `_recursive_split` 输出的每段都带 separator 尾巴, 例如 "句子。".
    在主打包路径上, overlap 边界可能落在 separator 中间, 但相邻 chunk 仍会
    有几个字符重叠. 硬切路径(没有 separator)严格 overlap.
    """

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        # overlap 必须 < chunk_size, 否则循环不收敛 (硬切路径).
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = max(0, self.chunk_size // 4)
        self.separators = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""]

    def split(self, text: str, doc_id: str = "") -> List[Chunk]:
        text = (text or "").strip()
        if not text:
            return []
        splits = self._recursive_split(text, self.separators)
        # 过滤纯分隔符的"幽灵段" (递归时会产生 "。" 这种空壳).
        splits = [s for s in splits if s.strip()]
        chunks: List[Chunk] = []
        buf = ""
        for s in splits:
            if len(buf) + len(s) <= self.chunk_size:
                # 1) buf + s 装得下: 累积.
                buf = (buf + s).strip()
                continue
            # 2) 装不下了, 先把 buf 切出.
            if buf:
                chunks.append(Chunk(text=buf, metadata={"doc_id": doc_id}))
                # 关键修复: 下一轮 buf 从上一 chunk 的尾部 overlap 开始, 而不是清空.
                buf = self._tail(buf, self.chunk_overlap)
            # 3) 现在 buf 是上一 chunk 的尾部 overlap. 尝试再吃 s.
            if len(s) <= self.chunk_size - len(buf):
                buf = (buf + s).strip()
                continue
            # 4) s 自己就超长: 硬切. 把 buf 拼到 s 前面, 让"上一 chunk 末尾"
            #    与"s 的开头"也重叠.
            piece = (buf + s) if buf else s
            buf = ""
            step = max(1, self.chunk_size - self.chunk_overlap)
            # 一片一片切, 每片都 <= chunk_size.
            for i in range(0, len(piece), step):
                slice_text = piece[i : i + self.chunk_size]
                if not slice_text.strip():
                    continue
                chunks.append(Chunk(text=slice_text, metadata={"doc_id": doc_id}))
            # 硬切结束后, 把最后一片的尾部 overlap 留给下一段作为 buf 起点.
            if chunks:
                buf = self._tail(chunks[-1].text, self.chunk_overlap)
        if buf and buf.strip():
            # 防 hard-cut 残留: 上一 chunk 末尾已经覆盖了 buf (buf 是其尾部 overlap),
            # 不要重复 emit 一个只含 overlap 内容的 chunk.
            if chunks and chunks[-1].text.endswith(buf):
                pass
            else:
                chunks.append(Chunk(text=buf.strip(), metadata={"doc_id": doc_id}))
        # 注入 chunk_id
        for i, c in enumerate(chunks):
            c.metadata["chunk_id"] = f"{doc_id}_c{i}"
            c.metadata["chunk_index"] = i
        return chunks

    @staticmethod
    def _tail(s: str, n: int) -> str:
        """返回 s 的末尾 n 个字符. n<=0 时返回空串."""
        if n <= 0 or not s:
            return ""
        return s[-n:]

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
        for idx, p in enumerate(parts):
            # 保留分隔符 (除最后一段, 因为它没有后续分隔符), 这样 buf 里能恢复原文.
            is_last = idx == len(parts) - 1
            if is_last:
                p_with_sep = p  # 不补 sep, 避免递归时把 sep 重复加进去产生幽灵段.
            else:
                p_with_sep = p + sep
            if not p_with_sep:
                continue
            if len(p_with_sep) <= self.chunk_size:
                out.append(p_with_sep)
            else:
                out.extend(self._recursive_split(p_with_sep, rest))
        return out