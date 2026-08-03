"""元数据增强 (Metadata Enhancement) / HyDE.

RAG 工程里, 给 chunk 追加**结构化元数据**能显著提升检索命中率:
  - 抽取式元数据: 字符数、关键词 (TF)、首句摘要、章节标题、语言 (中/英).
  - HyDE (Hypothetical Document Embeddings): 为 chunk 生成一段
    "假设性答案/问题" 作为额外检索锚点. 查询时把用户问题也映射到同一
    假设空间, 语义召回更稳 (尤其问句与文档陈述句式差异大时).

设计:
  - `enrich(text)` 永远可用 (抽取式, 不依赖 LLM), 保证入库链路零外部依赖.
  - `enrich_with_hyde(text, llm)` 在提供 LLM 时生成 `hyde_question`, 否则
    回退到抽取式首句摘要 (metadata 仍完整, 检索不降级).
  - 所有字段写入 chunk.metadata, 由 retriever 在召回时拼回上下文.
"""
from __future__ import annotations

import hashlib
import re
from typing import Callable, Dict, List, Optional

# 中文/英文/数字 token, 与 bm25_store 的 _tokenize 保持一致口径.
_TOKEN_RE = re.compile(r"[一-龥]+|[a-zA-Z][a-zA-Z0-9]*|[0-9]+")
_STOPWORDS = set(
    "的 了 和 是 在 我 有 也 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 "
    "好 自己 这 那 与 及 或 等 中 为 以 对 从 把 被 让 给 向 于 之 其 该 各 类 们 它 它们 "
    "the a an of to and or in on for with as by from at is are be this that these those "
    "we you they i he she it can will should may might must do does did has have had "
    "not no yes if then else when where what which who how why".split()
)
_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;\.\n])")


class MetadataEnhancer:
    """为 chunk 文本生成结构化元数据 + 可选 HyDE 假设问题."""

    def __init__(self, hyde_enabled: bool = True, max_keywords: int = 6):
        self.hyde_enabled = hyde_enabled
        self.max_keywords = max_keywords

    # ------------------------------------------------------------------
    # 抽取式 (永远可用)
    # ------------------------------------------------------------------
    def _keywords(self, text: str) -> List[str]:
        tf: Dict[str, int] = {}
        for tok in _TOKEN_RE.findall(text.lower()):
            if len(tok) < 2:
                continue
            if tok in _STOPWORDS:
                continue
            tf[tok] = tf.get(tok, 0) + 1
        ranked = sorted(tf.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
        return [w for w, _ in ranked[: self.max_keywords]]

    @staticmethod
    def _language(text: str) -> str:
        cn = len(re.findall(r"[一-龥]", text))
        en = len(re.findall(r"[a-zA-Z]+", text))
        if cn == 0 and en == 0:
            return "unknown"
        return "zh" if cn >= en else "en"

    @staticmethod
    def _first_sentence(text: str, limit: int = 120) -> str:
        sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
        if not sents:
            return text[:limit]
        return sents[0][:limit]

    @staticmethod
    def _stable_hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]

    def enrich(self, text: str) -> Dict[str, object]:
        """抽取式元数据. 不调用任何外部模型."""
        text = text or ""
        meta: Dict[str, object] = {
            "char_count": len(text),
            "language": self._language(text),
            "keywords": self._keywords(text),
            "summary": self._first_sentence(text),
            "content_hash": self._stable_hash(text),
        }
        return meta

    # ------------------------------------------------------------------
    # HyDE (可选 LLM)
    # ------------------------------------------------------------------
    def enrich_with_hyde(
        self,
        text: str,
        llm: Optional[Callable[[str], str]] = None,
    ) -> Dict[str, object]:
        """在抽取式元数据基础上追加 `hyde_question`.

        - 若传入 `llm(callable)`: 用 LLM 为 chunk 生成一句"该段落能回答的问题".
        - 否则回退到首句摘要, 保证 metadata 完整.
        """
        meta = self.enrich(text)
        if self.hyde_enabled and llm is not None:
            try:
                prompt = (
                    "阅读下面这段知识库内容, 只输出一句它能回答的用户问题 "
                    "(不要解释, 不要标点以外的符号):\n\n" + text[:800]
                )
                hyp = (llm(prompt) or "").strip()
                if hyp:
                    meta["hyde_question"] = hyp
                    return meta
            except Exception:
                pass
        # 回退: 用首句摘要作为 HyDE 锚点 (仍然有效, 只是弱一些).
        meta["hyde_question"] = meta["summary"]
        return meta

    # ------------------------------------------------------------------
    # 查询侧 HyDE: 把问题映射成"假设文档"再做向量检索
    # ------------------------------------------------------------------
    @staticmethod
    def hyde_query(
        query: str,
        llm: Optional[Callable[[str], str]] = None,
    ) -> str:
        """返回用于 embedding 的查询文本.

        有 LLM 时: 让 LLM 把问题扩写成一段"假设性答案文档", embedding 它去检索,
        比直接用短问句召回更稳. 无 LLM 时原样返回.
        """
        if llm is None:
            return query
        try:
            prompt = (
                "假设你要回答下面这个问题, 写一段简洁的参考答案文档 "
                "(2-3 句, 直接写内容, 不要前缀):\n\n" + query
            )
            hyp = (llm(prompt) or "").strip()
            return hyp or query
        except Exception:
            return query
