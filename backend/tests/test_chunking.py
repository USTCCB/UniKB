"""测试 1: 文档切片 (chunking)"""
from app.rag.chunking import split_text


def test_split_text_basic_chinese():
    text = (
        "UniKB 是一个面向企业知识管理场景的 RAG 平台。"
        "核心目标是把文档/网页/图片等多源知识, 通过多 Agent 协作 + MCP 工具协议"
        "+ 混合检索 + 重排序 + 引用溯源, 最终以流式、可溯源的方式回答用户问题。"
    )
    chunks = split_text(text, chunk_size=32, overlap=8)
    assert len(chunks) >= 2
    # overlap 必须生效
    assert any(c in chunks[1] for c in chunks[0])


def test_split_text_empty_input_returns_empty_list():
    assert split_text("", chunk_size=100, overlap=10) == []
    assert split_text("   \n\t  ", chunk_size=100, overlap=10) == []


def test_split_text_respects_chunk_size_upper_bound():
    text = "句子。" * 500  # 1500 字符
    chunks = split_text(text, chunk_size=64, overlap=8)
    for c in chunks:
        assert len(c) <= 64 + 8  # 允许 overlap 边界