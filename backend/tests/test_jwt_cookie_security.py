"""P2-11 测试: JWT 存储安全 (HttpOnly cookie + 安全响应头 + Bearer 兼容).

覆盖:
1. /auth/login 成功后返回 token 并设置 HttpOnly cookie.
2. 受保护路由可通过 cookie 访问 (不带 Bearer).
3. 受保护路由仍可通过 Bearer 访问 (向后兼容).
4. /auth/logout 清除 cookie.
5. 安全响应头 (CSP, X-Content-Type-Options, X-Frame-Options, Referrer-Policy) 存在.
6. dev 环境 HSTS 不出现, prod 环境 HSTS 出现.
7. cookie 属性: dev 时 secure=False + samesite=lax; prod 时 secure=True + samesite=strict.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """构造一个 patch 了嵌入/检索的 TestClient."""
    from app.agents import llm_router as lr_mod
    from app.api import chat as chat_mod
    from app.rag import bm25_store as bm_mod
    from app.rag import embedding as emb_mod
    from app.rag import reranker as rk_mod
    from app.rag import retriever as rt_mod
    from app.rag import vector_store as vs_mod

    import os
    from tests._fakes import (
        FakeBM25Store,
        FakeEmbeddingService,
        FakeLLM,
        FakeReranker,
        FakeVectorStore,
    )

    # 确保 dev 环境, 避免之前测试改到 prod
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("JWT_SECRET", "a-very-long-secret-for-test-only-32chars")

    # 清掉配置缓存, 否则上一个测试的 prod cache 会影响 dev 测试
    from app.core.config import get_settings
    get_settings.cache_clear()

    monkeypatch.chdir(os.environ.get("TMPDIR", "/tmp"))
    os.environ["UNIKB_FAKE_EMBEDDING"] = "1"
    emb_mod.get_embedding_service.cache_clear()
    monkeypatch.setattr(vs_mod, "ChromaStore", FakeVectorStore)
    monkeypatch.setattr(bm_mod, "BM25Store", FakeBM25Store)
    monkeypatch.setattr(rk_mod, "CrossEncoderReranker", FakeReranker)
    monkeypatch.setattr(emb_mod, "EmbeddingService", FakeEmbeddingService)
    monkeypatch.setattr(rt_mod, "ChromaStore", FakeVectorStore)
    monkeypatch.setattr(rt_mod, "BM25Store", FakeBM25Store)
    monkeypatch.setattr(rt_mod, "reset_retriever_cache", lambda: None)
    monkeypatch.setattr(lr_mod, "get_llm", lambda: FakeLLM())
    monkeypatch.setattr(chat_mod, "get_llm", lambda: FakeLLM())

    from app.main import app
    # TestClient 默认不进入 lifespan, 需手动建表并清空用户表
    from app.api.auth import reset_users_for_tests
    reset_users_for_tests()
    return TestClient(app)


def _register(c: TestClient, username: str, password: str = "password123") -> str:
    r = c.post(
        "/api/v1/auth/register",
        json={"username": username, "email": f"{username}@example.com", "password": password},
    )
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


# ... existing tests ...


def test_prod_hsts_header(client, monkeypatch):
    """prod 环境应带 HSTS 头."""
    from app.api.auth import reset_users_for_tests
    from app.core.config import get_settings

    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("JWT_SECRET", "a-very-long-secret-for-test-only-32chars")
    get_settings.cache_clear()

    # 重新 import app 让 lifespan 通过 prod 校验
    from app.main import app as prod_app
    reset_users_for_tests()
    prod_client = TestClient(prod_app)
    r = prod_client.get("/health")
    assert r.status_code == 200
    hsts = r.headers.get("strict-transport-security")
    assert hsts
    assert "max-age=31536000" in hsts


# ... rest unchanged ...


def test_prod_cookie_secure_strict(client, monkeypatch):
    """prod 环境 cookie secure=True, samesite=strict."""
    from app.api.auth import reset_users_for_tests
    from app.core.config import get_settings

    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("JWT_SECRET", "a-very-long-secret-for-test-only-32chars")
    get_settings.cache_clear()

    from app.main import app as prod_app
    reset_users_for_tests()
    prod_client = TestClient(prod_app)
    r = prod_client.post(
        "/api/v1/auth/register",
        json={"username": "frank", "email": "frank@example.com", "password": "password123"},
    )
    assert r.status_code == 200
    cookie = r.headers["set-cookie"].lower()
    assert "secure" in cookie
    assert "samesite=strict" in cookie