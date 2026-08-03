"""测试: 离线召回评估 Recall@K + RecallEvaluator."""
from __future__ import annotations

from app.rag.evaluate import RecallEvaluator, recall_at_k


def test_recall_at_k_perfect():
    assert recall_at_k(["a", "b"], ["a", "b", "c"], k=3) == 1.0


def test_recall_at_k_partial():
    # 相关集 2 个, 前 3 只命中 1 个 -> 0.5
    assert recall_at_k(["a", "b"], ["a", "x", "y"], k=3) == 0.5


def test_recall_at_k_truncated_by_k():
    # 相关集 2 个, 但 k=1 只取第一, 命中 1 -> 0.5
    assert recall_at_k(["a", "b"], ["a", "b"], k=1) == 0.5


def test_recall_at_k_empty_relevant():
    # 无相关项, 定义 Recall=1.0
    assert recall_at_k([], ["a"], k=5) == 1.0


def test_evaluator_runs_over_dataset():
    dataset = [
        {"question": "q1", "relevant_ids": ["d1", "d2"]},
        {"question": "q2", "relevant_ids": ["d3"]},
    ]

    def fake_retrieve(q: str) -> list[dict]:
        # 模拟检索: q1 召回 d1,d2 ; q2 召回 d3,x
        if q == "q1":
            return [{"id": "d1"}, {"id": "d2"}, {"id": "x"}]
        return [{"id": "d3"}, {"id": "x"}]

    ev = RecallEvaluator(ks=(1, 3, 5))
    res = ev.evaluate(dataset, fake_retrieve)
    assert res["n"] == 2
    # q1 相关 2 项, 但 top1 只命中 d1 -> recall@1 = 0.5; q2 top1 命中 -> 1.0; 平均 0.75
    assert res["recall"]["@1"] == 0.75
    assert res["recall"]["@3"] == 1.0
    assert len(res["per_question"]) == 2


def test_evaluator_handles_misses():
    dataset = [{"question": "q", "relevant_ids": ["missing"]}]

    def fake_retrieve(q: str) -> list[dict]:
        return [{"id": "other"}]

    res = RecallEvaluator(ks=(1, 5)).evaluate(dataset, fake_retrieve)
    assert res["recall"]["@1"] == 0.0
    assert res["recall"]["@5"] == 0.0
