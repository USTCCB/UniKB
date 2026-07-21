"""测试 8: LangFuse 可观测 wrapper (关闭态 no-op)."""
from __future__ import annotations


def test_tracer_disabled_returns_noop():
    from app.core.observability import LangfuseTracer

    t = LangfuseTracer()
    # 默认 LANGFUSE_ENABLED=false, 应该 noop
    assert t.enabled is False


def test_traced_context_manager_no_throw_when_disabled():
    from app.core.observability import traced

    with traced("test", user_id="u") as ctx:
        assert ctx is not None


def test_timed_context_manager_measures():
    import time
    from app.core.observability import timed

    start = time.perf_counter()
    with timed("noop"):
        time.sleep(0.01)
    elapsed = time.perf_counter() - start
    assert elapsed >= 0.01


def test_pipeline_runs_with_mocked_dependencies(monkeypatch):
    """pipeline.run_rag 在外部依赖被 mock 时不应抛错."""
    import asyncio

    # mock EmbeddingService (避免真实下载模型)
    from app.rag import embedding as emb_mod

    class FakeEmbedding:
        def embed(self, texts):
            return [[0.0] * 4 for _ in texts]

        def embed_query(self, text):
            return [0.0] * 4

    monkeypatch.setattr(emb_mod, "get_embedding_service", lambda: FakeEmbedding())

    # mock HybridRetriever
    from app.rag import retriever as rt_mod

    class FakeRetriever:
        def __init__(self, kb_id="default"):
            self.kb_id = kb_id

        def retrieve(self, q, top_k=10):
            return [
                {"id": "c1", "document": "上下文 [1]", "metadata": {"doc_id": "d1"}, "rerank_score": 0.9},
                {"id": "c2", "document": "上下文 [2]", "metadata": {"doc_id": "d1"}, "rerank_score": 0.7},
            ]

    monkeypatch.setattr(rt_mod, "HybridRetriever", FakeRetriever)

    # mock reranker
    from app.rag.reranker import CrossEncoderReranker

    monkeypatch.setattr(
        CrossEncoderReranker,
        "rerank",
        lambda self, q, cands, top_k: cands[:top_k],
    )

    # mock LLM
    class FakeLLM:
        def invoke(self, _prompt):
            class R:
                content = "测试回答 [1] 引用"
            return R()

    from app.agents import llm_router as lr

    monkeypatch.setattr(lr, "get_llm", lambda: FakeLLM())

    from app.rag import pipeline as pl_mod

    out = asyncio.run(pl_mod.run_rag(question="什么是 UniKB?", kb_id="default", top_k=2))
    assert "answer" in out
    assert "测试回答" in out["answer"]
    assert isinstance(out["contexts"], list)
    assert len(out["contexts"]) == 2
    assert out["context_ids"] == ["c1", "c2"]
    assert out["metadata"]["mode"] == "rag"


def test_pipeline_agent_mode(monkeypatch):
    """agent 模式应该能跑通 (用 fake graph)."""
    import asyncio

    class FakeGraph:
        def invoke(self, state):
            return {**state, "final": "agent 模式回答", "retrieved": "上下文"}

    # pipeline 里是 lazy import `from app.agents.graph import build_agent_graph`,
    # 所以 monkeypatch 那个原模块
    from app.agents import graph as graph_mod

    monkeypatch.setattr(graph_mod, "build_agent_graph", lambda: FakeGraph())

    # mock LLM (agent 模式不用, 但保险起见)
    class FakeLLM:
        def invoke(self, _prompt):
            class R:
                content = "agent 模式回答"
            return R()

    from app.agents import llm_router as lr

    monkeypatch.setattr(lr, "get_llm", lambda: FakeLLM())

    from app.rag import pipeline as pl_mod

    out = asyncio.run(pl_mod.run_rag(question="q", use_agent=True))
    assert out["answer"] == "agent 模式回答"
    assert out["metadata"]["mode"] == "agent"
    assert out["contexts"] == ["上下文"]
