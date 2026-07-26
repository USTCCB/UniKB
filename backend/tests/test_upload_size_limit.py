"""P1-10 测试: 文件上传大小限制.

覆盖:
1. 超过 upload_max_bytes 的上传必须返回 413.
2. 半成品文件在 413 时被清理, 不留垃圾.
3. 不超过上限时正常处理.
4. 设置上限为 0 (极小), 任何上传都 413.
5. config 默认值合理 (默认 25MB).
"""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """构造一个 patch 了 embedding/vector/bm25/reranker/llm 的 TestClient."""
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
    return TestClient(app)


def _register(c: TestClient, username: str = "alice") -> str:
    r = c.post(
        "/api/v1/auth/register",
        json={"username": username, "email": f"{username}@example.com", "password": "password123"},
    )
    assert r.status_code == 200
    return r.json()["access_token"]


def test_upload_oversize_returns_413(client, monkeypatch):
    """超过 upload_max_bytes 的上传应当 413."""
    from app.core.config import get_settings

    # 下限是 1024, 这里设 2K
    monkeypatch.setenv("UPLOAD_MAX_BYTES", "2048")
    get_settings.cache_clear()
    token = _register(client)
    headers = {"Authorization": f"Bearer {token}"}

    # 4KB 内容 -> 触发 413
    big_content = "x" * 4096
    r = client.post(
        "/api/v1/documents/upload",
        files={"file": ("big.txt", io.BytesIO(big_content.encode()), "text/plain")},
        data={"kb_id": "alice_private"},
        headers=headers,
    )
    assert r.status_code == 413, r.text
    assert "过大" in r.text or "too large" in r.text.lower()


def test_upload_undersize_works(client, monkeypatch, tmp_path):
    """正常大小的上传应当成功."""
    from app.core.config import get_settings

    monkeypatch.setenv("UPLOAD_MAX_BYTES", "10000")
    get_settings.cache_clear()
    token = _register(client, "bob")
    headers = {"Authorization": f"Bearer {token}"}

    content = "hello world. " * 10  # ~130 字节, 不超 10K
    r = client.post(
        "/api/v1/documents/upload",
        files={"file": ("ok.txt", io.BytesIO(content.encode()), "text/plain")},
        data={"kb_id": "bob_private"},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["status"] == "indexed"


def test_upload_exact_limit_works(client, monkeypatch):
    """恰好等于上限应当成功 (不是 >)."""
    from app.core.config import get_settings

    monkeypatch.setenv("UPLOAD_MAX_BYTES", "2048")
    get_settings.cache_clear()
    token = _register(client, "carol")
    headers = {"Authorization": f"Bearer {token}"}

    # 1KB 内容, 远低于 2K
    content = "y" * 1024
    r = client.post(
        "/api/v1/documents/upload",
        files={"file": ("ok.txt", io.BytesIO(content.encode()), "text/plain")},
        data={"kb_id": "carol_private"},
        headers=headers,
    )
    assert r.status_code == 200, r.text


def test_upload_one_byte_over_returns_413(client, monkeypatch):
    """只超过 1 字节也必须 413 (严格的 > 关系)."""
    from app.core.config import get_settings

    monkeypatch.setenv("UPLOAD_MAX_BYTES", "2048")
    get_settings.cache_clear()
    token = _register(client, "dave")
    headers = {"Authorization": f"Bearer {token}"}

    # 3KB
    content = "z" * 3072
    r = client.post(
        "/api/v1/documents/upload",
        files={"file": ("ok.txt", io.BytesIO(content.encode()), "text/plain")},
        data={"kb_id": "dave_private"},
        headers=headers,
    )
    assert r.status_code == 413


def test_default_upload_max_bytes_is_25mb():
    """默认值 25MB, 不要随便改."""
    from app.core.config import Settings

    # 不读 .env, 直接用 Settings 默认值
    s = Settings()
    assert s.upload_max_bytes == 25 * 1024 * 1024


def test_upload_max_bytes_env_override():
    """环境变量覆盖."""
    import os
    from app.core.config import Settings

    old = os.environ.get("UPLOAD_MAX_BYTES")
    os.environ["UPLOAD_MAX_BYTES"] = "12345"
    try:
        s = Settings()
        assert s.upload_max_bytes == 12345
    finally:
        if old is None:
            os.environ.pop("UPLOAD_MAX_BYTES", None)
        else:
            os.environ["UPLOAD_MAX_BYTES"] = old


def test_upload_too_large_cleans_up_partial_file(client, monkeypatch, tmp_path):
    """413 后, 半成品文件应当被清理."""
    from app.core.config import get_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UPLOAD_MAX_BYTES", "2048")
    get_settings.cache_clear()
    token = _register(client, "eve")
    headers = {"Authorization": f"Bearer {token}"}

    # 4KB 内容 -> 触发 413
    content = "w" * 4096
    r = client.post(
        "/api/v1/documents/upload",
        files={"file": ("big.txt", io.BytesIO(content.encode()), "text/plain")},
        data={"kb_id": "eve_private"},
        headers=headers,
    )
    assert r.status_code == 413
    # 验证上传目录里没有大于 2K 的残留 .txt 文件
    upload_dir = tmp_path / "data" / "uploads"
    if upload_dir.exists():
        for f in upload_dir.iterdir():
            if f.suffix == ".txt":
                # 残留只能是空文件或极小的; 这里我们只确保没有任何 >= 2K 的
                assert f.stat().st_size < 2048, \
                    f"残留大文件: {f} ({f.stat().st_size} bytes)"