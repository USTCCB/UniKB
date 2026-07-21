"""测试 3: Cross-Encoder 重排 (用 mock 分数模拟, 不依赖真实模型下载)."""
from app.rag.reranker import rerank


def test_rerank_keeps_top_n_in_descending_order():
    query = "RAG 检索"
    docs = [
        "BM25 + 向量混合检索",  # 期望第一
        "数据库索引优化",       # 期望靠后
        "Cross-Encoder 重排抑制幻觉",  # 期望第二
        "前端 UI 设计",          # 最差
    ]
    scores = [0.9, 0.1, 0.7, 0.05]  # mock
    ranked = rerank(query, docs, scores, top_n=2)
    assert len(ranked) == 2
    assert ranked[0][0] == docs[0]
    assert ranked[1][0] == docs[2]


def test_rerank_returns_score_tuples():
    ranked = rerank("q", ["a", "b"], [0.5, 0.9], top_n=2)
    for item in ranked:
        assert isinstance(item, tuple)
        assert len(item) == 2


def test_rerank_top_n_larger_than_docs_returns_all():
    ranked = rerank("q", ["only"], [0.5], top_n=10)
    assert len(ranked) == 1