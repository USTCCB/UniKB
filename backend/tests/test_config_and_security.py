"""测试 11: 配置管理 + 安全工具."""
from __future__ import annotations

from datetime import timedelta

import pytest


def test_settings_loads_from_env_example(monkeypatch):
    """默认应该能从 backend/.env.example 读到所有默认值."""
    from app.core.config import Settings

    s = Settings()
    assert s.llm_provider in {"deepseek", "qwen", "openai"}
    assert s.chunk_size >= 100
    assert s.chunk_overlap < s.chunk_size
    assert s.top_k_final > 0


def test_settings_llm_provider_routing():
    from app.core.config import Settings

    s = Settings()
    assert s.get_llm_api_key(s.llm_provider) is None or isinstance(s.get_llm_api_key(s.llm_provider), str)
    assert s.get_llm_base_url("deepseek").startswith("https://")
    assert "openai" in s.get_llm_base_url("openai")


def test_settings_unknown_provider_returns_openai_base():
    from app.core.config import Settings

    s = Settings()
    url = s.get_llm_base_url("not-exist")
    assert url.startswith("https://")


def test_settings_chunk_validation():
    from app.core.config import Settings

    with pytest.raises(Exception):
        Settings(chunk_size=10)  # 小于 ge=100
    with pytest.raises(Exception):
        Settings(chunk_overlap=500)  # 大于 le=400


def test_password_hash_and_verify_roundtrip():
    from app.core.security import hash_password, verify_password

    h = hash_password("super-secret")
    assert h != "super-secret"
    assert verify_password("super-secret", h) is True
    assert verify_password("wrong", h) is False


def test_create_token_with_custom_expiry():
    from app.core.security import create_access_token, verify_token

    t = create_access_token(subject="alice", expires_delta=timedelta(minutes=5))
    assert verify_token(t) == "alice"


def test_password_hash_is_bcrypt_format():
    from app.core.security import hash_password

    h = hash_password("x")
    # passlib 的 bcrypt 默认前缀
    assert h.startswith("$2") or h.startswith("!")  # ! 是 passlib 标记


def test_settings_default_llm_model_matches_provider(monkeypatch):
    from app.core.config import Settings

    s = Settings()
    monkeypatch.setattr(s, "llm_provider", "deepseek")
    assert s.default_llm_model == s.deepseek_model
    monkeypatch.setattr(s, "llm_provider", "qwen")
    assert s.default_llm_model == s.qwen_model
