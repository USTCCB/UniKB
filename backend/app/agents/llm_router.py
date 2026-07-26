# -*- coding: utf-8 -*-
"""统一 LLM 路由：DeepSeek / Qwen / OpenAI，自动按 provider 拼装 chat model。"""
from __future__ import annotations
from functools import lru_cache
from typing import Literal

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


@lru_cache
def get_llm() -> BaseChatModel:
    return LLMRouter().get()
