# -*- coding: utf-8 -*-
"""统一 LLM 路由：DeepSeek / Qwen / OpenAI，自动按 provider 拼装 chat model。

支持按请求切换 provider / model: `get_llm(provider, model)` 会按 (provider, model)
作 key 缓存, 同一组合复用同一 client, 换 provider / model 才新建.

向后兼容: `get_llm()` 不带参数仍返回默认 provider 的默认 model (走 `settings`),
这样老的调用点 `get_llm().invoke(...)` 不需要改动.

测试里如果 patch 了 `_build_llm`, 也可以直接 monkey-patch 该函数实现自定义 fake.
"""
from __future__ import annotations
from functools import lru_cache
from typing import Callable, Literal, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import settings


Provider = Literal["deepseek", "qwen", "openai"]


class LLMRouter:
    def get(self, provider=None, model=None) -> BaseChatModel:
        provider = (provider or settings.default_llm_provider).lower()
        model = model or settings.default_llm_model
        api_key = settings.get_llm_api_key(provider)
        if not api_key:
            raise ValueError("API key for provider '" + provider + "' not set. Please fill " + provider.upper() + "_API_KEY in .env")
        base_url = settings.get_llm_base_url(provider)
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.2,
            streaming=True,
        )


def _resolve_default_pair() -> Tuple[str, str]:
    return (settings.default_llm_provider.lower(), settings.default_llm_model)


@lru_cache(maxsize=16)
def _build_llm(provider: str, model: str) -> BaseChatModel:
    """按 (provider, model) 缓存 client 实例.

    同样组合多次调用复用同一个对象, 既避免重复 handshake, 也方便做 per-request
    切换 (`get_llm("qwen", "qwen-max")` 和 `get_llm("deepseek", "deepseek-chat")`
    会得到两个不同的 client).
    """
    return LLMRouter().get(provider=provider, model=model)


def get_llm(provider: Optional[str] = None, model: Optional[str] = None) -> BaseChatModel:
    """获取 LLM client.

    - `get_llm()` -> 默认 provider/model (进程启动时由环境变量固定).
    - `get_llm("qwen")` -> 指定 provider, model 用对应 provider 的默认 model.
    - `get_llm("openai", "gpt-4o")` -> 任意组合, 按 (provider, model) 缓存.
    """
    if provider is None and model is None:
        provider, model = _resolve_default_pair()
    else:
        provider = (provider or settings.default_llm_provider).lower()
        if not model:
            model = settings.get_llm_model(provider)
    return _build_llm(provider, model)


def reset_llm_cache() -> None:
    """仅供测试: 清空 (provider, model) -> client 缓存."""
    _build_llm.cache_clear()


def get_llm_text_callable(
    provider: Optional[str] = None, model: Optional[str] = None
) -> Callable[[str], str]:
    """返回一个 (prompt: str) -> str 的纯文本适配器.

    场景: metadata/HyDE/查询改写等只需要模型吐出的"文本", 但 `get_llm()` 返回的是
    `BaseChatModel`, 直接 `llm(prompt).strip()` 拿到的是 `AIMessage`, 会抛
    `AttributeError` 被 except 吞掉, 导致 LLM 分支永远走不到 (这是之前查询改写/HyDE
    不生效的根因). 这里统一取出 `.content`.
    """
    llm = get_llm(provider, model)

    def _call(prompt: str) -> str:
        out = llm.invoke(prompt)
        if isinstance(out, str):
            return out
        return getattr(out, "content", "") or ""

    return _call