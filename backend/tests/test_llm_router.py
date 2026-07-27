"""测试 get_llm 的按请求切换.

要求 (来自 P3-8):
1. `get_llm()` 不带参数时, 返回默认 provider/model 的 client (向后兼容).
2. `get_llm("qwen")` 返回 qwen 的 client, 不再要求进程重启换 provider.
3. `get_llm("openai", "gpt-4o")` 任意组合按 (provider, model) 缓存.
4. 不同组合返回不同 client (per-request 切换是真的).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.agents import llm_router


@pytest.fixture(autouse=True)
def _clear_cache():
    """每个测试前后清空缓存, 避免相互污染."""
    llm_router.reset_llm_cache()
    yield
    llm_router.reset_llm_cache()


def _fake_router_get(provider=None, model=None):
    """替换 LLMRouter.get, 用一个简单 dataclass 标记 provider/model."""
    from app.core.config import settings

    provider = (provider or settings.default_llm_provider).lower()
    model = model or settings.get_llm_model(provider)

    class _FakeLLM:
        def __init__(self):
            self.provider = provider
            self.model = model

    return _FakeLLM()


def test_get_llm_default_returns_provider_model():
    """get_llm() 返回 (default_provider, default_model) 组合."""
    with patch.object(llm_router, "LLMRouter") as MockRouter:
        MockRouter.return_value.get.side_effect = lambda *a, **kw: _fake_router_get(*a, **kw)
        a = llm_router.get_llm()
    from app.core.config import settings

    assert a.provider == settings.default_llm_provider.lower()
    assert a.model == settings.get_llm_model(settings.default_llm_provider.lower())


def test_get_llm_per_provider_returns_distinct_clients():
    """get_llm('qwen') 与 get_llm('deepseek') 返回不同 client."""
    with patch.object(llm_router, "LLMRouter") as MockRouter:
        MockRouter.return_value.get.side_effect = lambda *a, **kw: _fake_router_get(*a, **kw)
        a = llm_router.get_llm("qwen")
        b = llm_router.get_llm("deepseek")
    assert a.provider == "qwen"
    assert b.provider == "deepseek"
    assert a is not b


def test_get_llm_per_model_returns_distinct_clients():
    """get_llm('openai', 'gpt-4o') 与 get_llm('openai', 'gpt-4o-mini') 是不同的 cache key."""
    with patch.object(llm_router, "LLMRouter") as MockRouter:
        MockRouter.return_value.get.side_effect = lambda *a, **kw: _fake_router_get(*a, **kw)
        a = llm_router.get_llm("openai", "gpt-4o")
        b = llm_router.get_llm("openai", "gpt-4o-mini")
    assert a.model == "gpt-4o"
    assert b.model == "gpt-4o-mini"
    assert a is not b


def test_get_llm_same_args_returns_same_instance():
    """相同 (provider, model) 必须返回同一实例 (复用 client, 避免重复握手)."""
    with patch.object(llm_router, "LLMRouter") as MockRouter:
        MockRouter.return_value.get.side_effect = lambda *a, **kw: _fake_router_get(*a, **kw)
        a = llm_router.get_llm("qwen", "qwen-max")
        b = llm_router.get_llm("qwen", "qwen-max")
    assert a is b


def test_get_llm_resolve_default_uses_provider_default_model():
    """get_llm('qwen') 不指定 model 时, model 应是 qwen_provider 的默认 model."""
    from app.core.config import settings

    with patch.object(llm_router, "LLMRouter") as MockRouter:
        MockRouter.return_value.get.side_effect = lambda *a, **kw: _fake_router_get(*a, **kw)
        a = llm_router.get_llm("qwen")
    assert a.model == settings.get_llm_model("qwen")