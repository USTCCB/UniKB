"""查询改写 (Query Rewriting).

用户问题常常口语化、有歧义、或与文档陈述句式不一致, 直接拿去检索召回差。
改写的目标:
  - **扩展 (Expansion)**: 补上同义/上下位词, 让 BM25 也能命中.
  - **拆解 (Decomposition)**: 把复合问题拆成多个子问题, 分别检索再融合.
  - **消歧 (Disambiguation)**: 把指代/缩写展开成文档里的标准说法.

实现:
  - 有 LLM 时: 让 LLM 输出 N 个改写变体 (一行一个).
  - 无 LLM (测试/离线): 走**确定性回退** —— 按中英文标点做子句切分,
    每个子句作为一条检索 query, 与原问题一起融合, 仍然能提升召回.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional

_SPLIT_PUNCT = re.compile(r"[；;。！!？?\n]+")
_BRACKET = re.compile(r"[\[\]【】()（）<>《》\"'\"'`,，、]")
_WS = re.compile(r"\s+")


class QueryRewriter:
    """把一个问题改写成多条检索 query.

    `rewrite(query, llm=None) -> List[str]` 总是包含原始 query,
    追加 0~max_variants 条改写.
    """

    def __init__(self, enabled: bool = True, max_variants: int = 3):
        self.enabled = enabled
        self.max_variants = max_variants

    def rewrite(
        self,
        query: str,
        llm: Optional[Callable[[str], str]] = None,
    ) -> List[str]:
        query = (query or "").strip()
        if not query:
            return []
        if not self.enabled:
            return [query]
        # 去括号类噪声与多余空白, 但保留句末标点 (问号等是 query 的一部分).
        cleaned = _WS.sub(" ", _BRACKET.sub(" ", query)).strip()
        if llm is not None:
            try:
                variants = self._llm_rewrite(cleaned, llm)
                if variants:
                    out = [cleaned] + variants
                    # 去重保序
                    seen = set()
                    uniq = []
                    for v in out:
                        if v and v not in seen:
                            seen.add(v)
                            uniq.append(v)
                    return uniq[: 1 + self.max_variants]
            except Exception:
                pass
        # 确定性回退: 子句切分 (复合问题拆解)
        return self._fallback_split(cleaned)

    @staticmethod
    def _llm_rewrite(query: str, llm: Callable[[str], str]) -> List[str]:
        prompt = (
            "你是检索系统的 Query Rewriter. 给定用户问题, 生成最多 3 条"
            "检索友好的改写 (同义扩展 / 子问题拆解 / 标准说法), 每行一条, "
            "不要编号, 不要解释:\n\n" + query
        )
        resp = (llm(prompt) or "").strip()
        out: List[str] = []
        for line in resp.splitlines():
            line = line.strip().lstrip("0123456789.-、) ").strip()
            if line:
                out.append(line)
        return out

    def _fallback_split(self, query: str) -> List[str]:
        """无 LLM 回退: 按标点拆子句, 原句 + 各子句分别检索后融合."""
        parts = [p.strip() for p in _SPLIT_PUNCT.split(query) if p.strip()]
        # 单个子句就没必要拆了
        if len(parts) <= 1:
            return [query]
        out = [query]
        for p in parts:
            if len(p) >= 4 and p not in out:  # 太短的子句意义不大
                out.append(p)
        return out[: 1 + self.max_variants]
