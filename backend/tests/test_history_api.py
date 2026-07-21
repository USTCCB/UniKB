"""测试 9: Chat History API (纯内存 / 临时目录).

跳过真实 FastAPI 客户端 (httpx), 直接构造请求对象调用 handler.
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path


def _override_history_dir(monkeypatch, tmpdir: Path):
    from app.api import history

    monkeypatch.setattr(history, "_HISTORY_DIR", tmpdir)


def test_create_session_returns_session_id(monkeypatch):
    from app.api import history

    with tempfile.TemporaryDirectory() as td:
        _override_history_dir(monkeypatch, Path(td))

        async def _call():
            return await history.create_session(payload={"title": "测试"}, user="alice")

        out = asyncio.run(_call())
        assert out["session_id"].startswith("sess_")
        # 文件落地
        assert (Path(td) / "alice.json").exists()


def test_append_and_list_session(monkeypatch):
    from app.api import history

    with tempfile.TemporaryDirectory() as td:
        _override_history_dir(monkeypatch, Path(td))

        async def _call():
            created = await history.create_session(payload={"title": "测试"}, user="bob")
            sid = created["session_id"]
            await history.append_message(
                sid,
                history.HistoryItem(role="user", content="hi", ts=1.0),
                user="bob",
            )
            await history.append_message(
                sid,
                history.HistoryItem(role="assistant", content="hello", ts=2.0),
                user="bob",
            )
            return sid

        sid = asyncio.run(_call())
        # 同步读
        data = history._load("bob")
        sess = data[sid]
        assert len(sess["messages"]) == 2
        assert sess["messages"][0]["role"] == "user"
        assert sess["messages"][1]["role"] == "assistant"


def test_append_message_auto_creates_session(monkeypatch):
    from app.api import history

    with tempfile.TemporaryDirectory() as td:
        _override_history_dir(monkeypatch, Path(td))

        async def _call():
            return await history.append_message(
                "sess_unknown",
                history.HistoryItem(role="user", content="first", ts=1.0),
                user="carol",
            )

        out = asyncio.run(_call())
        assert out["session_id"] == "sess_unknown"
        assert out["message_count"] == 1


def test_delete_session_removes_record(monkeypatch):
    from app.api import history

    with tempfile.TemporaryDirectory() as td:
        _override_history_dir(monkeypatch, Path(td))

        async def _call():
            created = await history.create_session(payload={"title": "x"}, user="dave")
            sid = created["session_id"]
            await history.delete_session(sid, user="dave")
            return sid

        sid = asyncio.run(_call())
        data = history._load("dave")
        assert sid not in data


def test_list_sessions_returns_sorted_by_updated(monkeypatch):
    from app.api import history

    with tempfile.TemporaryDirectory() as td:
        _override_history_dir(monkeypatch, Path(td))

        async def _call():
            s1 = await history.create_session(payload={"title": "旧会话"}, user="eve")
            await history.append_message(
                s1["session_id"],
                history.HistoryItem(role="user", content="x", ts=1.0),
                user="eve",
            )
            s2 = await history.create_session(payload={"title": "新会话"}, user="eve")
            return s1["session_id"], s2["session_id"]

        s1, s2 = asyncio.run(_call())
        out = asyncio.run(history.list_sessions(user="eve"))
        # 按 updated_at 倒序
        assert out[0].session_id == s2
        assert out[1].session_id == s1
        assert out[0].title == "新会话"


def test_get_session_raises_404(monkeypatch):
    from app.api import history
    from fastapi import HTTPException

    with tempfile.TemporaryDirectory() as td:
        _override_history_dir(monkeypatch, Path(td))

        async def _call():
            return await history.get_session("sess_nope", user="frank")

        try:
            asyncio.run(_call())
            assert False, "should have raised"
        except HTTPException as e:
            assert e.status_code == 404


def test_history_item_timestamp_defaults_to_now():
    import time
    from app.api.history import HistoryItem

    before = time.time()
    item = HistoryItem(role="user", content="hi")
    after = time.time()
    assert before <= item.ts <= after
