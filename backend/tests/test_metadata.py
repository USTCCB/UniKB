"""测试: 元数据增强 MetadataEnhancer / HyDE."""
from __future__ import annotations

from app.rag.metadata import MetadataEnhancer


def test_enrich_extracts_keywords_summary_language():
    text = "混合检索结合 BM25 与向量语义检索以提升召回率。BM25 是一种经典的关键词检索算法。"
    m = MetadataEnhancer().enrich(text)
    assert m["char_count"] == len(text)
    assert "language" in m and m["language"] in ("zh", "en")
    assert isinstance(m["keywords"], list) and len(m["keywords"]) > 0
    # 拉丁术语应被抽出 (大小写归一), 中文按词串抽取.
    assert "bm25" in [k.lower() for k in m["keywords"]]
    assert m["summary"]  # 首句摘要非空


def test_hyde_falls_back_to_summary_without_llm():
    text = "父文档召回能在召回某 chunk 时一并返回同文档的兄弟 chunk。"
    m = MetadataEnhancer(hyde_enabled=True).enrich_with_hyde(text, llm=None)
    # 无 LLM 时 hyde_question 回退到首句摘要, 但字段必须存在.
    assert "hyde_question" in m
    assert m["hyde_question"]


def test_hyde_uses_llm_when_provided():
    text = "向量数据库用于存储 embedidng 并支持近似最近邻检索。"

    def fake_llm(prompt: str) -> str:
        return "向量数据库如何支持语义检索？"

    m = MetadataEnhancer().enrich_with_hyde(text, llm=fake_llm)
    assert m["hyde_question"] == "向量数据库如何支持语义检索？"


def test_hyde_query_expands_with_llm():
    called = {}

    def fake_llm(prompt: str) -> str:
        called["p"] = prompt
        return "假设答案是一段关于 RAG 检索的说明文档。"

    q = MetadataEnhancer.hyde_query("什么是 RAG 检索？", llm=fake_llm)
    assert "假设答案" in q
    assert "RAG" in called["p"]


def test_hyde_query_no_llm_returns_original():
    q = MetadataEnhancer.hyde_query("什么是 RAG 检索？", llm=None)
    assert q == "什么是 RAG 检索？"
