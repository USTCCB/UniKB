"""测试: 重排候选池优化 RerankOptimizer (Top-50 候选 -> Top-20 精排)."""
from __future__ import annotations

from app.rag.reranker import CrossEncoderReranker, RerankOptimizer


class _ScoringReranker(CrossEncoderReranker):
    """用确定性打分替掉 CrossEncoder, 不打模型."""

    def __init__(self, scores: list[float]):
        super().__init__(model_name="fake")
        self._scores = scores

    def _ensure(self):
        return self

    def rerank(self, query, candidates, top_k=5):
        out = []
        for i, c in enumerate(candidates):
            c2 = dict(c)
            c2["rerank_score"] = self._scores[i] if i < len(self._scores) else 0.0
            out.append(c2)
        out.sort(key=lambda x: x["rerank_score"], reverse=True)
        return out[:top_k]


def _cands(n: int, start_score: float = 1.0):
    return [
        {"id": f"c{i}", "document": f"doc {i}", "rrf_score": start_score - i * 0.01}
        for i in range(n)
    ]


def test_optimizer_truncates_pool_before_rerank():
    # 100 个候选, 但 Cross-Encoder 只应看到 pool=50 个.
    cands = _cands(100)
    seen = []

    class _Tracked(_ScoringReranker):
        def rerank(self, query, candidates, top_k=5):
            seen.extend(c["id"] for c in candidates)
            return super().rerank(query, candidates, top_k=top_k)

    opt = RerankOptimizer(reranker=_Tracked([0.9] * 50))
    opt.optimize("q", cands, pool=50, top_n=20)
    # Cross-Encoder 输入被截断到 50
    assert len(seen) == 50
    assert "c50" not in seen


def test_optimizer_returns_top_n():
    cands = _cands(80)
    scores = [0.1, 0.9, 0.5] + [0.0] * 77  # 期望 c1 排第一
    opt = RerankOptimizer(reranker=_ScoringReranker(scores))
    out = opt.optimize("q", cands, pool=50, top_n=20)
    assert len(out) == 20
    # 分数最高的是 c1 (score 0.9)
    assert out[0]["id"] == "c1"


def test_optimizer_empty_candidates():
    opt = RerankOptimizer(reranker=_ScoringReranker([]))
    assert opt.optimize("q", [], pool=50, top_n=20) == []
