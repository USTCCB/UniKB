"""测试限流 (P0-5).

覆盖:
1. login 失败计数, 超过阈值后下一次请求被 429 锁定 (且带 Retry-After).
2. login 成功后清空失败计数.
3. chat 配额: 超过 rate_limit_chat_per_min 后下一次 429.
4. 不同 username/IP 互相隔离.
5. rate_limit_enabled=False 时, 所有限流都失效 (用于测试隔离).
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app.api import auth as auth_mod
from app.core import rate_limit as rl_mod
from app.core.config import get_settings


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """每个测试清空限流状态, 关掉 settings 缓存."""
    auth_mod.reset_users_for_tests()
    rl_mod.reset_for_tests()
    get_settings.cache_clear()
    # 让所有限流在测试里走小窗口以减少耗时
    monkeypatch.setenv("RATE_LIMIT_LOGIN_FAIL_MAX", "3")
    monkeypatch.setenv("RATE_LIMIT_LOGIN_WINDOW_SEC", "60")
    monkeypatch.setenv("RATE_LIMIT_LOGIN_LOCK_SEC", "60")
    monkeypatch.setenv("RATE_LIMIT_CHAT_PER_MIN", "3")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "1")
    get_settings.cache_clear()
    yield
    auth_mod.reset_users_for_tests()
    rl_mod.reset_for_tests()
    get_settings.cache_clear()
    os.environ.pop("RATE_LIMIT_LOGIN_FAIL_MAX", None)
    os.environ.pop("RATE_LIMIT_LOGIN_WINDOW_SEC", None)
    os.environ.pop("RATE_LIMIT_LOGIN_LOCK_SEC", None)
    os.environ.pop("RATE_LIMIT_CHAT_PER_MIN", None)
    os.environ.pop("RATE_LIMIT_ENABLED", None)


def _register(c: TestClient, username: str, password: str = "password123") -> str:
    r = c.post(
        "/api/v1/auth/register",
        json={"username": username, "email": f"{username}@example.com", "password": password},
    )
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


# ============ 单元: rate_limit 模块 ============

def test_login_lock_after_n_failures(monkeypatch):
    """失败 N 次后, 下一次请求被锁定 (allowed=False)."""
    monkeypatch.setenv("RATE_LIMIT_LOGIN_FAIL_MAX", "3")
    get_settings.cache_clear()
    rl_mod.reset_for_tests()

    for _ in range(3):
        decision = rl_mod.check_login_failure("alice", "1.2.3.4")
        rl_mod.record_login_failure("alice", "1.2.3.4")
        assert decision.allowed, "前 3 次不应被锁定"
    d = rl_mod.check_login_failure("alice", "1.2.3.4")
    assert not d.allowed
    assert d.retry_after > 0


def test_login_success_clears_failures(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_LOGIN_FAIL_MAX", "3")
    get_settings.cache_clear()
    rl_mod.reset_for_tests()
    rl_mod.record_login_failure("alice", "1.2.3.4")
    rl_mod.record_login_failure("alice", "1.2.3.4")
    rl_mod.record_login_success("alice", "1.2.3.4")
    d = rl_mod.check_login_failure("alice", "1.2.3.4")
    assert d.allowed


def test_login_lock_isolated_per_user_and_ip():
    """锁定对其他账号/IP 不影响."""
    for _ in range(5):
        rl_mod.check_login_failure("alice", "1.1.1.1")
        rl_mod.record_login_failure("alice", "1.1.1.1")
    d = rl_mod.check_login_failure("alice", "1.1.1.1")
    assert not d.allowed
    # 别的账号不受影响
    assert rl_mod.check_login_failure("bob", "1.1.1.1").allowed
    # 同账号其它 IP 也不受影响
    assert rl_mod.check_login_failure("alice", "2.2.2.2").allowed


def test_chat_quota_blocks_after_limit(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_CHAT_PER_MIN", "3")
    get_settings.cache_clear()
    rl_mod.reset_for_tests()
    for i in range(3):
        d = rl_mod.check_chat_quota("alice", "1.2.3.4")
        assert d.allowed, f"第 {i+1} 次不应被限流"
    d = rl_mod.check_chat_quota("alice", "1.2.3.4")
    assert not d.allowed
    assert d.retry_after > 0


def test_chat_quota_isolated_per_user():
    rl_mod.reset_for_tests()
    for _ in range(3):
        rl_mod.check_chat_quota("alice", "ip")
    assert not rl_mod.check_chat_quota("alice", "ip").allowed
    assert rl_mod.check_chat_quota("bob", "ip").allowed


def test_rate_limit_can_be_disabled(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "0")
    monkeypatch.setenv("RATE_LIMIT_CHAT_PER_MIN", "1")
    get_settings.cache_clear()
    rl_mod.reset_for_tests()
    for _ in range(10):
        d = rl_mod.check_chat_quota("alice", "ip")
        assert d.allowed


# ============ 集成: HTTP 端点 ============

def _client_with_fakes(monkeypatch):
    """构造一个 patch 了 embedding/vector/bm25/reranker/llm 的 TestClient."""
    from app.agents import llm_router as lr_mod
    from app.api import chat as chat_mod
    from app.rag import bm25_store as bm_mod
    from app.rag import embedding as emb_mod
    from app.rag import reranker as rk_mod
    from app.rag import retriever as rt_mod
    from app.rag import vector_store as vs_mod

    from tests._fakes import (
        FakeBM25Store,
        FakeEmbeddingService,
        FakeLLM,
        FakeReranker,
        FakeVectorStore,
    )

    os.environ["UNIKB_FAKE_EMBEDDING"] = "1"
    emb_mod.get_embedding_service.cache_clear()
    monkeypatch.setattr(vs_mod, "ChromaStore", FakeVectorStore)
    monkeypatch.setattr(bm_mod, "BM25Store", FakeBM25Store)
    monkeypatch.setattr(rk_mod, "CrossEncoderReranker", FakeReranker)
    monkeypatch.setattr(emb_mod, "EmbeddingService", FakeEmbeddingService)
    monkeypatch.setattr(rt_mod, "ChromaStore", FakeVectorStore)
    monkeypatch.setattr(rt_mod, "BM25Store", FakeBM25Store)
    monkeypatch.setattr(lr_mod, "get_llm", lambda: FakeLLM())
    monkeypatch.setattr(chat_mod, "get_llm", lambda: FakeLLM())

    from app.main import app

    return TestClient(app)


def test_http_login_lock_after_failures(monkeypatch):
    """真实 HTTP 路径: 错误密码连续失败, 第 N+1 次必须 429."""
    c = _client_with_fakes(monkeypatch)
    # 没注册的用户, /login 永远错, 用来模拟暴力破解
    for i in range(3):
        r = c.post("/api/v1/auth/login", json={"username": "alice", "password": "wrong"})
        assert r.status_code == 401, f"第 {i+1} 次应 401, got {r.status_code}"
    # 第 4 次应被锁定
    r = c.post("/api/v1/auth/login", json={"username": "alice", "password": "wrong"})
    assert r.status_code == 429, r.text
    assert r.headers.get("Retry-After") is not None


def test_http_login_success_resets_failure_counter(monkeypatch):
    c = _client_with_fakes(monkeypatch)
    _register(c, "alice")

    # 几次失败
    for _ in range(2):
        r = c.post("/api/v1/auth/login", json={"username": "alice", "password": "wrong"})
        assert r.status_code == 401
    # 登录成功 -> 清空计数
    r = c.post("/api/v1/auth/login", json={"username": "alice", "password": "password123"})
    assert r.status_code == 200
    # 再失败 2 次, 都不会被锁定
    for _ in range(2):
        r = c.post("/api/v1/auth/login", json={"username": "alice", "password": "wrong"})
        assert r.status_code == 401


def test_http_chat_quota_returns_429_after_limit(monkeypatch):
    """chat 触发 N 次后下一次 429."""
    c = _client_with_fakes(monkeypatch)
    token = _register(c, "alice")
    headers = {"Authorization": f"Bearer {token}"}

    # 配额是 3
    for i in range(3):
        r = c.post(
            "/api/v1/chat",
            json={"question": "hi", "kb_id": "default", "mode": "rag", "top_k": 2},
            headers=headers,
        )
        assert r.status_code == 200, f"第 {i+1} 次应 200, got {r.status_code}: {r.text}"
    r = c.post(
        "/api/v1/chat",
        json={"question": "hi", "kb_id": "default", "mode": "rag", "top_k": 2},
        headers=headers,
    )
    assert r.status_code == 429, r.text
    assert r.headers.get("Retry-After") is not None


def test_http_chat_quota_isolated_per_user(monkeypatch):
    c = _client_with_fakes(monkeypatch)
    token_a = _register(c, "alice")
    token_b = _register(c, "bob")

    for i in range(3):
        r = c.post(
            "/api/v1/chat",
            json={"question": "hi", "kb_id": "default", "mode": "rag", "top_k": 2},
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert r.status_code == 200
    # alice 被锁
    r = c.post(
        "/api/v1/chat",
        json={"question": "hi", "kb_id": "default", "mode": "rag", "top_k": 2},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert r.status_code == 429
    # bob 不受影响
    r = c.post(
        "/api/v1/chat",
        json={"question": "hi", "kb_id": "default", "mode": "rag", "top_k": 2},
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert r.status_code == 200