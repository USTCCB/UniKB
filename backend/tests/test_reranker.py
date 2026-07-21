"""测试 3: Cross-Encoder 重排 (mock 分数, 不下载模型)."""
from __future__ import annotations

import pytest

from app.rag.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    """不下载真实模型, 直接返回传入的假分数."""

    def __init__(self, scores):
        self._scores = scores
        self.idx = 0

    def predict(self, pairs):
        # 简单按 idx 切分数
        n = len(pairs)
        out = self._scores[self.idx : self.idx + n]
        self.idx += n
        if len(out) < n:
            out = out + [0.0] * (n - len(out))
        return out


@pytest.fixture
def reranker(monkeypatch):
    r = CrossEncoderReranker(model_name="fake-model")
    return r


def test_rerank_top_n_in_descending_order(reranker, monkeypatch):
    monkeypatch.setattr(
        "app.rag.reranker.CrossEncoder",
        lambda *_a, **_kw: FakeCrossEncoder([0.9, 0.1, 0.7, 0.05]),
    )
    query = "RAG 检索"
    docs = [
        "BM25 + 向量混合检索",  # 期望第一
        "数据库索引优化",       # 期望靠后
        "Cross-Encoder 重排抑制幻觉",  # 期望第二
        "前端 UI 设计",          # 最差
    ]
    ranked = reranker.rerank(query, docs, top_n=2)
    assert len(ranked) == 2
    assert ranked[0][0] == docs[0]
    assert ranked[1][0] == docs[2]


def test_rerank_returns_score_tuples(reranker, monkeypatch):
    monkeypatch.setattr(
        "app.rag.reranker.CrossEncoder",
        lambda *_a, **_kw: FakeCrossEncoder([0.5, 0.9]),
    )
    ranked = reranker.rerank("q", ["a", "b"], top_n=2)
    for item in ranked:
        assert isinstance(item, tuple) and len(item) == 2


def test_rerank_top_n_larger_than_docs_returns_all(reranker, monkeypatch):
    monkeypatch.setattr(
        "app.rag.reranker.CrossEncoder",
        lambda *_a, **_kw: FakeCrossEncoder([0.5]),
    )
    ranked = reranker.rerank("q", ["only"], top_n=10)
    assert len(ranked) == 1