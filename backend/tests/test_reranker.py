"""测试 3: Cross-Encoder 重排 (mock, 不下载模型).

app.rag.reranker.CrossEncoderReranker.rerank(query, candidates, top_k)
的 candidates 是 List[dict], 不是 List[str].
"""
from __future__ import annotations

import pytest

from app.rag.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    """假装是 sentence_transformers.CrossEncoder, 直接返回传入的假分数."""

    def __init__(self, scores):
        self._scores = scores
        self.idx = 0

    def predict(self, pairs):
        n = len(pairs)
        out = self._scores[self.idx : self.idx + n]
        self.idx += n
        if len(out) < n:
            out = list(out) + [0.0] * (n - len(out))
        return out


@pytest.fixture
def reranker(monkeypatch):
    monkeypatch.setattr(
        "sentence_transformers.CrossEncoder",
        lambda *_a, **_kw: FakeCrossEncoder([0.9, 0.1, 0.7, 0.05]),
    )
    return CrossEncoderReranker(model_name="fake-model")


def _docs_as_dicts(texts):
    return [{"document": t} for t in texts]


def test_rerank_top_n_in_descending_order(reranker):
    docs = _docs_as_dicts([
        "BM25 + 向量混合检索",       # 期望第一
        "数据库索引优化",            # 期望靠后
        "Cross-Encoder 重排抑制幻觉",  # 期望第二
        "前端 UI 设计",              # 最差
    ])
    ranked = reranker.rerank("RAG 检索", docs, top_k=2)
    assert len(ranked) == 2
    # 第一条期望分数最高
    assert ranked[0]["document"] == "BM25 + 向量混合检索"
    assert ranked[1]["document"] == "Cross-Encoder 重排抑制幻觉"


def test_rerank_returns_dict_list_with_score(reranker):
    docs = _docs_as_dicts(["a", "b"])
    ranked = reranker.rerank("q", docs, top_n=2)  # type: ignore[arg-type]
    assert isinstance(ranked, list)
    # 每条 result 应有 rerank_score
    assert all("rerank_score" in r for r in ranked)


def test_rerank_top_n_larger_than_docs_returns_all(reranker):
    ranked = reranker.rerank("q", _docs_as_dicts(["only"]), top_k=10)
    assert len(ranked) == 1