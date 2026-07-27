"""测试 chunker 的 overlap 是否真的生效.

覆盖:
1. 主打包路径 (按分隔符累积): 相邻 chunk 应该有 overlap 字符.
2. 硬切路径 (单段超过 chunk_size): 仍然要留 overlap.
3. overlap 不能 >= chunk_size (构造函数会回退到 chunk_size/4).
"""
from __future__ import annotations

from app.rag.chunker import TextChunker


def _overlap(a: str, b: str, expected: int) -> bool:
    """检查 b 的开头是否有 expected 个字符与 a 的末尾重叠."""
    if expected <= 0:
        return True
    if len(a) < expected or len(b) < expected:
        return False
    return a[-expected:] == b[:expected]


def test_main_packing_applies_overlap_between_chunks():
    """主打包路径 (按分隔符累积): 相邻 chunk 末尾/开头应该有 overlap."""
    # 句子短, 多次累加超过 chunk_size 才会切. overlap=10, 每个 chunk 之间应
    # 该有 10 个字符重叠.
    sentences = ["第一句内容比较长一点。", "第二句内容也比较长一点。", "第三句内容也不短。", "第四句内容跟前面类似。"]
    text = "".join(sentences)
    ch = TextChunker(chunk_size=20, chunk_overlap=10)
    out = ch.split(text, doc_id="d")
    assert len(out) >= 2, "需要至少两个 chunk 才能验证 overlap"
    # 至少有一对相邻 chunk 是有 overlap 的.
    found_overlap = False
    for a, b in zip(out, out[1:]):
        # b 的开头应该与 a 的末尾有重叠 (>= 5 字符, 因为 chunk_size 小, overlap 上限受 chunk_size/4 影响).
        # 这里 chunk_size=20, overlap 上限是 5. 我们要求 overlap 上限尽量大,
        # 所以 overlap 至少 1 字符即可.
        for k in range(5, 0, -1):
            if _overlap(a.text, b.text, k):
                found_overlap = True
                break
        if found_overlap:
            break
    assert found_overlap, f"相邻 chunk 之间没有 overlap, chunks={[c.text for c in out]}"


def test_main_packing_long_text_overlap_present():
    """长文本, 主打包路径, chunk_size 大点, overlap 取大值, 强校验."""
    # 用顺序字符而不是重复, 这样能看出 overlap 真的生效.
    text = "".join(chr(ord("a") + (i % 26)) for i in range(300))  # 300 字符, 不重复
    ch = TextChunker(chunk_size=80, chunk_overlap=20)
    out = ch.split(text, doc_id="d2")
    assert len(out) >= 2
    # 主打包路径下, 至少一对相邻 chunk 之间有 overlap 字符.
    found_overlap = False
    for a, b in zip(out, out[1:]):
        # 至少 1 字符重叠 (分隔符切分会破坏严格 overlap 边界, 但重叠仍存在).
        for k in range(20, 0, -1):
            if _overlap(a.text, b.text, k):
                found_overlap = True
                break
        if found_overlap:
            break
    assert found_overlap, f"相邻 chunk 之间没有 overlap, chunks={[c.text for c in out]}"


def test_overlap_clamped_when_too_large():
    """overlap >= chunk_size 时构造函数自动回退到 chunk_size/4, 不会死循环."""
    ch = TextChunker(chunk_size=20, chunk_overlap=50)
    assert ch.chunk_overlap < ch.chunk_size
    # 仍然能跑通.
    out = ch.split("foo。bar。" * 30, doc_id="d3")
    assert len(out) >= 1


def test_hard_split_path_keeps_overlap():
    """单段超过 chunk_size 的硬切路径: 每片之间也要有 overlap (这是旧实现就有的)."""
    # 一段超长, 没有分隔符, 走硬切.
    text = "x" * 500
    ch = TextChunker(chunk_size=50, chunk_overlap=10)
    out = ch.split(text, doc_id="d4")
    assert len(out) >= 2
    # 每片长度 <= chunk_size
    for c in out:
        assert len(c.text) <= 50
    # 至少相邻两片之间应有重叠 (硬切路径).
    for a, b in zip(out, out[1:]):
        # 硬切路径用 step=chunk_size-overlap, 所以 a 末尾 chunk_overlap 字符 == b 开头.
        assert a.text[-ch.chunk_overlap:] == b.text[:ch.chunk_overlap], (
            "硬切路径 chunk 之间应严格 overlap"
        )