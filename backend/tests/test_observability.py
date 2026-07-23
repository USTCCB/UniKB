"""测试 8: LangFuse 可观测 wrapper (关闭态 no-op).

注: 原 test_pipeline_runs_with_mocked_dependencies / test_pipeline_agent_mode
已搬到 tests/test_integration_retrieval.py, 用统一的 _install_fakes fixture 跑.
这里只保留 observability 本身的单元测试.
"""
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
