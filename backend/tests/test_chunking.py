"""测试 1: 文档切片 (Recursive chunker)."""
from __future__ import annotations

from app.rag.chunker import TextChunker


def _chunker():
    return TextChunker(chunk_size=32, chunk_overlap=8)


def test_chunk_basic_chinese():
    chunks = _chunker().split(
        "UniKB 是一个面向企业知识管理场景的 RAG 平台。"
        "核心目标是把文档/网页/图片等多源知识, 通过多 Agent 协作 + MCP 工具协议"
        "+ 混合检索 + 重排序 + 引用溯源, 最终以流式、可溯源的方式回答用户问题。",
        doc_id="t1",
    )
    assert len(chunks) >= 2
    assert all(hasattr(c, "text") for c in chunks)


def test_chunk_empty_input_returns_empty_list():
    assert _chunker().split("", doc_id="t") == []
    assert _chunker().split("   \n\t  ", doc_id="t") == []


def test_chunk_respects_size_upper_bound():
    chunks = _chunker().split("句子。" * 500, doc_id="t")
    for c in chunks:
        assert len(c.text) <= 32 + 8  # chunk_size + overlap 边界