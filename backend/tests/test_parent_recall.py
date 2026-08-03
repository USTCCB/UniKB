"""测试: 父文档召回 (连坐召回) ParentDocRecall."""
from __future__ import annotations

from app.rag.retriever import HybridRetriever


def _make_retriever() -> HybridRetriever:
    """构造一个不依赖 chromadb/torch 的 retriever (不调用 retrieve)."""
    r = HybridRetriever(kb_id="test_parent")
    # 手动灌入 bm25_store.docs: 两个文档, 各 3 个 chunk.
    r.bm25_store.docs = [
        {"id": "docA_c0", "text": "A段落0", "metadata": {"doc_id": "docA"}},
        {"id": "docA_c1", "text": "A段落1", "metadata": {"doc_id": "docA"}},
        {"id": "docA_c2", "text": "A段落2", "metadata": {"doc_id": "docA"}},
        {"id": "docB_c0", "text": "B段落0", "metadata": {"doc_id": "docB"}},
        {"id": "docB_c1", "text": "B段落1", "metadata": {"doc_id": "docB"}},
    ]
    return r


def test_parent_recall_brings_siblings():
    r = _make_retriever()
    # 只召回了 docA_c0
    fused = [{"id": "docA_c0", "document": "A段落0", "metadata": {"doc_id": "docA"}, "rrf_score": 0.9}]
    expanded = r.expand_parent_recall(fused, max_docs=5, max_extra_per_doc=8)
    ids = [c["id"] for c in expanded]
    # docA 的兄弟 chunk 应被连坐召回
    assert "docA_c1" in ids
    assert "docA_c2" in ids
    # 不应混入其他文档的 chunk
    assert "docB_c0" not in ids
    # 父 chunk 仍在
    assert "docA_c0" in ids


def test_parent_recall_respects_extra_limit():
    r = _make_retriever()
    fused = [{"id": "docA_c0", "document": "A段落0", "metadata": {"doc_id": "docA"}, "rrf_score": 0.9}]
    expanded = r.expand_parent_recall(fused, max_docs=5, max_extra_per_doc=1)
    sib_ids = [c["id"] for c in expanded if c["id"] != "docA_c0"]
    # 最多补 1 个兄弟
    assert len(sib_ids) == 1


def test_parent_recall_inherits_parent_score():
    r = _make_retriever()
    fused = [{"id": "docA_c0", "document": "A段落0", "metadata": {"doc_id": "docA"}, "rrf_score": 0.9}]
    expanded = r.expand_parent_recall(fused, max_docs=5, max_extra_per_doc=8)
    # 兄弟 chunk 的分数应略低于父 (0.9 * 0.99), 排在其后.
    sib = next(c for c in expanded if c["id"] == "docA_c1")
    assert abs(sib["rrf_score"] - 0.9 * 0.99) < 1e-9
    # 父排第一
    assert expanded[0]["id"] == "docA_c0"


def test_parent_recall_empty_fused():
    r = _make_retriever()
    assert r.expand_parent_recall([]) == []
