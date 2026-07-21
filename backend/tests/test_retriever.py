"""测试 2: RRF 融合 (Reciprocal Rank Fusion)."""
from __future__ import annotations

from app.rag.retriever import rrf_fuse


def test_rrf_merges_overlapping_results_higher_rank():
    bm25 = [{"id": "a", "score": 10.0}, {"id": "b", "score": 8.0}, {"id": "c", "score": 5.0}]
    vector = [{"id": "b", "score": 0.9}, {"id": "a", "score": 0.7}, {"id": "d", "score": 0.6}]
    fused = rrf_fuse([bm25, vector], k=60)
    top_ids = [item["id"] for item in fused[:2]]
    # a/b 在两路都出现, 应该排前
    assert set(top_ids) == {"a", "b"}


def test_rrf_single_source_preserves_input_order():
    bm25 = [{"id": "x"}, {"id": "y"}, {"id": "z"}]
    fused = rrf_fuse([bm25], k=60)
    assert [item["id"] for item in fused] == ["x", "y", "z"]


def test_rrf_handles_disjoint_results():
    bm25 = [{"id": "a"}, {"id": "b"}]
    vector = [{"id": "c"}, {"id": "d"}]
    fused = rrf_fuse([bm25, vector], k=60)
    assert len(fused) == 4
    assert {item["id"] for item in fused} == {"a", "b", "c", "d"}