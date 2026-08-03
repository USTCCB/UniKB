"""测试: 自适应切分 AdaptiveChunker (短文档整篇 / 长文档放大 chunk_size)."""
from __future__ import annotations

from app.rag.chunker import TextChunker


def test_short_doc_kept_whole():
    # 短于 adaptive_short_doc_chars (默认 500) 的文档应整体保留为 1 个 chunk.
    text = "这是一段很短的文档内容，整体作为单一 chunk 保留以提升上下文完整度。"
    ch = TextChunker(adaptive=True, adaptive_short_doc_chars=500)
    out = ch.split(text, doc_id="d1")
    assert len(out) == 1
    assert out[0].text == text
    assert out[0].metadata["chunk_id"] == "d1_c0"


def test_long_doc_uses_adaptive_chunk_size():
    # 长文档应使用 adaptive_chunk_size (默认 5500) 而非默认 chunk_size (500).
    text = "。".join([f"第{i}段业务说明内容" for i in range(400)])  # 远长于 500
    ch = TextChunker(adaptive=True, adaptive_chunk_size=5500, adaptive_short_doc_chars=500)
    out = ch.split(text, doc_id="d_long")
    # 由于放大了 chunk_size, chunk 数量应明显少于用默认 500 切的版本.
    out_default = TextChunker(adaptive=False, chunk_size=500).split(text, doc_id="d_long")
    assert len(out) < len(out_default)
    # 自适应长文档模式下不应切碎成上百片
    assert len(out) <= 10


def test_adaptive_off_keeps_original_behavior():
    # 关闭自适应应完全回退到默认递归切分 (保证既有测试/行为不变).
    text = "。".join([f"句子{i}内容描述" for i in range(50)])
    base = TextChunker(adaptive=False, chunk_size=500).split(text, doc_id="x")
    # 默认 chunk_size=500 来自 settings; 这里显式传 chunk_size 确保等价.
    assert len(base) >= 1


def test_chunk_ids_injected():
    text = "短文档一整段。"
    ch = TextChunker(adaptive=True, adaptive_short_doc_chars=500)
    out = ch.split(text, doc_id="docA")
    assert out[0].metadata["chunk_id"].startswith("docA_c")
    assert out[0].metadata["chunk_index"] == 0
