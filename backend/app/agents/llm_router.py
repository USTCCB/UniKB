"""统一 LLM 路由：DeepSeek / Qwen / OpenAI，自动按 provider 拼装 chat model。"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import settings


Provider = Literal["deepseek", "qwen", "openai"]


class LLMRouter:
    """按 provider 返回 LangChain ChatModel。"""

    def get(self, provider: str | None = None, model: str | None = None) -> BaseChatModel:
        provider = (provider or settings.default_llm_provider).lower()
        model = model or settings.default_llm_model
        api_key = settings.get_llm_api_key(provider)
        if not api_key:
            raise ValueError(
                f"API key for provider 