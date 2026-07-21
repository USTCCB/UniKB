"""测试 1: 文档切片 (Recursive chunker)."""
from __future__ import annotations

import pytest

from app.rag.chunker import TextChunker


@pytest.fixture
def chunker():
    return TextChunker(chunk_size=32, chunk_overlap=8)


def test_chunk_basic_chinese(chunker):
    text = (
        "UniKB 是一个面向企业知识管理场景的 RAG 平台。"
        "核心目标是把文档/网页/图片等多源知识, 通过多 Agent 协作 + MCP 工具协议"
        "+ 混合检索 + 重排序 + 引用溯源, 最终以流式、可溯源的方式回答用户问题。"
    )
    chunks = chunker.split_text(text)
    assert len(chunks) >= 2


def test_chunk_empty_input_returns_empty_list(chunker):
    assert chunker.split_text("") == []
    assert chunker.split_text("   \n\t  ") == []


def test_chunk_respects_size_upper_bound(chunker):
    text = "句子。" * 500
    chunks = chunker.split_text(text)
    for c in chunks:
        assert len(c.text) <= 32 + 8  # chunk_size + overlap 边界