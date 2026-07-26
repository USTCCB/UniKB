"""测试知识库越权访问 (P0-1).

场景:
1. A 上传到自己的 kb_B, B 调 /documents/list 用 kb_A 必须返回 403.
2. B 上传到 kb_A 必须返回 403 (私有 kb 已被 A 占用).
3. B 上传到 kb_B (B 自己) 必须返回 200.
4. 任意登录用户访问公共 kb "default" 都能 list / chat (200), 但不能上传 (403).
5. B 在 /chat 上用 kb_A 必须 403.

测试用 FastAPI TestClient + fakes (ChromaStore / BM25Store / embedding / reranker / LLM),
保证不依赖真实重型模型.
"""
from __future__ import annotations


import pytest
from fastapi.testclient import TestClient

# 必须在 import app.main 之前先把测试隔离的环境变量设上, 否则 app.core.kb_registry
# 会读到上一轮测试遗留的 data/kb_registry.json.
@pytest.fixture(autouse=True)
def _isolate_registry(tmp_path, monkeypatch):
    """每个测试用独立的 cwd + KB registry, 不污染真实 data/."""
    monkeypatch.chdir(tmp_path)
    from app.api import auth as auth_mod
    from app.core import kb_registry

    auth_mod.reset_users_for_tests()
    kb_registry.reset_for_tests()
    # 清掉可能缓存的 settings (lru_cache), 让 env 生效.
    from app.core.config import get_settings

    get_settings.cache_clear()
    yield
    auth_mod.reset_users_for_tests()
    kb_registry.reset_for_tests()


@pytest.fixture
def client(monkeypatch):
    """一个装好所有 fake 的 FastAPI TestClient."""
    # 注入 fake embedding / vector / bm25 / rerank / llm.
    import os as _os

    from app.api import chat as chat_mod
    from app.agents import llm_router as lr_mod
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

    _os.environ["UNIKB_FAKE_EMBEDDING"] = "1"
    emb_mod.get_embedding_service.cache_clear()

    monkeypatch.setattr(vs_mod, "ChromaStore", FakeVectorStore)
    monkeypatch.setattr(bm_mod, "BM25Store", FakeBM25Store)
    monkeypatch.setattr(rk_mod, "CrossEncoderReranker", FakeReranker)
    monkeypatch.setattr(emb_mod, "EmbeddingService", FakeEmbeddingService)
    monkeypatch.setattr(rt_mod, "ChromaStore", FakeVectorStore)
    monkeypatch.setattr(rt_mod, "BM25Store", FakeBM25Store)
    # chat 路由直接 import 了 retriever 模块里的类, 已经在上面 patch.
    # chat 还用了 llm_router.get_llm: 既要 patch 源头, 也要 patch chat 模块命名空间.
    monkeypatch.setattr(lr_mod, "get_llm", lambda: FakeLLM())
    monkeypatch.setattr(chat_mod, "get_llm", lambda: FakeLLM())

    from app.main import app

    with TestClient(app) as c:
        yield c

    emb_mod.get_embedding_service.cache_clear()
    _os.environ.pop("UNIKB_FAKE_EMBEDDING", None)


def _register(client: TestClient, username: str, password: str = "password123") -> str:
    r = client.post(
        "/api/v1/auth/register",
        json={"username": username, "email": f"{username}@example.com", "password": password},
    )
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _make_upload_bytes(name: str = "note.txt") -> bytes:
    return ("hello world from " + name + "\n").encode("utf-8")


# ----- 1. A 创建 kb 后, B 访问应该 403 -----
def test_user_B_cannot_list_user_A_private_kb(client):
    token_a = _register(client, "alice")
    token_b = _register(client, "bob")

    # A 上传到 alice_kb: 应成功 (200)
    files = {"file": ("a.txt", _make_upload_bytes("a.txt"), "text/plain")}
    r = client.post(
        "/api/v1/documents/upload",
        files=files,
        data={"kb_id": "alice_kb"},
        headers=_auth(token_a),
    )
    assert r.status_code == 200, r.text

    # B 用 alice_kb 调 list: 应 403
    r = client.get(
        "/api/v1/documents/list",
        params={"kb_id": "alice_kb"},
        headers=_auth(token_b),
    )
    assert r.status_code == 403, f"expected 403, got {r.status_code}: {r.text}"


# ----- 2. B 不能往 alice 已经创建的 kb 写 -----
def test_user_B_cannot_upload_to_user_A_private_kb(client):
    token_a = _register(client, "alice")
    token_b = _register(client, "bob")

    files = {"file": ("a.txt", _make_upload_bytes(), "text/plain")}
    r = client.post(
        "/api/v1/documents/upload",
        files=files,
        data={"kb_id": "alice_kb"},
        headers=_auth(token_a),
    )
    assert r.status_code == 200, r.text

    # B 试图往 alice_kb 写
    files = {"file": ("b.txt", _make_upload_bytes("b.txt"), "text/plain")}
    r = client.post(
        "/api/v1/documents/upload",
        files=files,
        data={"kb_id": "alice_kb"},
        headers=_auth(token_b),
    )
    assert r.status_code == 403, f"expected 403, got {r.status_code}: {r.text}"


# ----- 3. B 上传到自己的私有 kb 应该成功 -----
def test_user_B_can_upload_to_own_private_kb(client):
    token_b = _register(client, "bob")
    files = {"file": ("b.txt", _make_upload_bytes(), "text/plain")}
    r = client.post(
        "/api/v1/documents/upload",
        files=files,
        data={"kb_id": "bob_kb"},
        headers=_auth(token_b),
    )
    assert r.status_code == 200, r.text


# ----- 4. 公共 kb "default": 任何用户可 list, 但不能上传 -----
def test_public_kb_is_readable_by_anyone_but_not_writable(client):
    token_a = _register(client, "alice")
    token_b = _register(client, "bob")

    # list (read): 两个用户都能访问 default
    r = client.get(
        "/api/v1/documents/list",
        params={"kb_id": "default"},
        headers=_auth(token_a),
    )
    assert r.status_code == 200, r.text
    r = client.get(
        "/api/v1/documents/list",
        params={"kb_id": "default"},
        headers=_auth(token_b),
    )
    assert r.status_code == 200, r.text

    # 上传 (write): 不允许写公共 kb
    files = {"file": ("d.txt", _make_upload_bytes(), "text/plain")}
    r = client.post(
        "/api/v1/documents/upload",
        files=files,
        data={"kb_id": "default"},
        headers=_auth(token_a),
    )
    assert r.status_code == 403, f"expected 403, got {r.status_code}: {r.text}"


# ----- 5. chat (一次性) 必须同样校验 kb 归属 -----
def test_chat_endpoint_enforces_kb_acl(client):
    token_a = _register(client, "alice")
    token_b = _register(client, "bob")

    # A 在 alice_kb 里有文档 (无所谓, B 访问也应该被拒).
    files = {"file": ("a.txt", _make_upload_bytes(), "text/plain")}
    r = client.post(
        "/api/v1/documents/upload",
        files=files,
        data={"kb_id": "alice_kb"},
        headers=_auth(token_a),
    )
    assert r.status_code == 200, r.text

    # B 调 /chat 问 alice_kb: 必须 403
    r = client.post(
        "/api/v1/chat",
        json={"question": "hi", "kb_id": "alice_kb", "mode": "rag", "top_k": 3},
        headers=_auth(token_b),
    )
    assert r.status_code == 403, f"expected 403, got {r.status_code}: {r.text}"

    # B 问公共 kb default: 应该 200
    r = client.post(
        "/api/v1/chat",
        json={"question": "hi", "kb_id": "default", "mode": "rag", "top_k": 3},
        headers=_auth(token_b),
    )
    assert r.status_code == 200, r.text


# ----- 6. chat/stream 同样校验 kb 归属 -----
def test_chat_stream_endpoint_enforces_kb_acl(client):
    token_a = _register(client, "alice")
    token_b = _register(client, "bob")

    files = {"file": ("a.txt", _make_upload_bytes(), "text/plain")}
    r = client.post(
        "/api/v1/documents/upload",
        files=files,
        data={"kb_id": "alice_kb"},
        headers=_auth(token_a),
    )
    assert r.status_code == 200, r.text

    # B 调 /chat/stream 用 alice_kb: 必须 403
    r = client.post(
        "/api/v1/chat/stream",
        json={"question": "hi", "kb_id": "alice_kb", "mode": "rag", "top_k": 3},
        headers=_auth(token_b),
    )
    assert r.status_code == 403, f"expected 403, got {r.status_code}: {r.text}"

    # 公共 kb: 200
    r = client.post(
        "/api/v1/chat/stream",
        json={"question": "hi", "kb_id": "default", "mode": "rag", "top_k": 3},
        headers=_auth(token_b),
    )
    assert r.status_code == 200, r.text


# ----- 7. 私有 kb 必须存在才能 list (404), 不能借此嗅探别人的 kb -----
def test_private_kb_missing_returns_404_not_403(client):
    """不应区分 404 vs 403 的 kb 是否存在, 避免泄露; 但已注册的私有 kb 仍应该是 403."""
    token_b = _register(client, "bob")
    # 一个从未被创建过的 kb_id
    r = client.get(
        "/api/v1/documents/list",
        params={"kb_id": "ghost_kb"},
        headers=_auth(token_b),
    )
    # 约定: 不存在的私有 kb → 404
    assert r.status_code == 404, f"expected 404, got {r.status_code}: {r.text}"


# ----- 8. kb_registry 单元测试 -----
def test_kb_registry_roundtrip():
    from app.core.kb_registry import (
        ensure_kb_for_read,
        ensure_kb_for_write,
        get_kb,
        is_public_kb,
    )

    # 私有 kb 不存在 → write 自动创建并归属; read 报 404
    kb = ensure_kb_for_write("alpha", "alice")
    assert kb.owner == "alice"
    assert kb.is_public is False
    assert get_kb("alpha").owner == "alice"

    # alice 自己 read alpha: OK
    ensure_kb_for_read("alpha", "alice")

    # bob read alpha: 403
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        ensure_kb_for_read("alpha", "bob")
    assert exc.value.status_code == 403

    # default 是公共 kb
    assert is_public_kb("default")
    ensure_kb_for_read("default", "anyone")

    # 写公共 kb: 403
    with pytest.raises(HTTPException) as exc:
        ensure_kb_for_write("default", "alice")
    assert exc.value.status_code == 403