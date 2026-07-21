"""测试 2: BM25 + 向量混合检索的 RRF 融合."""
from app.rag.retriever import reciprocal_rank_fusion


def test_rrf_merges_overlapping_results_higher_rank():
    bm25 = [("a", 10.0), ("b", 8.0), ("c", 5.0)]
    vector = [("b", 0.9), ("a", 0.7), ("d", 0.6)]
    fused = reciprocal_rank_fusion([bm25, vector], k=60)
    # b/a 在两路都出现, 应该排前
    top_ids = [doc_id for doc_id, _score in fused[:2]]
    assert set(top_ids) == {"a", "b"}


def test_rrf_single_source_returns_sorted_by_score():
    bm25 = [("x", 1.0), ("y", 2.0), ("z", 3.0)]
    fused = reciprocal_rank_fusion([bm25], k=60)
    assert [d for d, _ in fused] == ["z", "y", "x"]


def test_rrf_handles_disjoint_results():
    bm25 = [("a", 1.0), ("b", 2.0)]
    vector = [("c", 0.9), ("d", 0.8)]
    fused = reciprocal_rank_fusion([bm25, vector], k=60)
    assert len(fused) == 4
    assert {d for d, _ in fused} == {"a", "b", "c", "d"}