"""Chat history API: 列出 / 获取 / 删除历史会话。"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from app.api.deps import get_current_user


router = APIRouter(prefix="/api/v1/history", tags=["history"])


_HISTORY_DIR = Path("./data/history")


class HistoryItem(BaseModel):
    role: str  # user / assistant
    content: str
    sources: list[dict] = Field(default_factory=list)
    ts: float = Field(default_factory=lambda: time.time())


class SessionMeta(BaseModel):
    session_id: str
    title: str
    kb_id: str = "default"
    mode: str = "rag"
    created_at: float
    updated_at: float
    message_count: int


class SessionDetail(SessionMeta):
    messages: list[HistoryItem]


def _user_path(user: str) -> Path:
    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c for c in user if c.isalnum() or c in "-_")
    return _HISTORY_DIR / f"{safe}.json"


def _load(user: str) -> dict:
    p = _user_path(user)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(user: str, data: dict) -> None:
    p = _user_path(user)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


@router.get("", response_model=list[SessionMeta], summary="列出当前用户全部会话")
async def list_sessions(user: str = Depends(get_current_user)):
    data = _load(user)
    out: list[SessionMeta] = []
    for sid, sess in data.items():
        try:
            out.append(SessionMeta(
                session_id=sid,
                title=sess.get("title", "新会话"),
                kb_id=sess.get("kb_id", "default"),
                mode=sess.get("mode", "rag"),
                created_at=sess.get("created_at", 0),
                updated_at=sess.get("updated_at", 0),
                message_count=len(sess.get("messages", [])),
            ))
        except Exception:
            continue
    out.sort(key=lambda x: x.updated_at, reverse=True)
    return out


@router.get("/{session_id}", response_model=SessionDetail, summary="获取会话详情")
async def get_session(session_id: str, user: str = Depends(get_current_user)):
    data = _load(user)
    sess = data.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="会话不存在")
    return SessionDetail(
        session_id=session_id,
        title=sess.get("title", "新会话"),
        kb_id=sess.get("kb_id", "default"),
        mode=sess.get("mode", "rag"),
        created_at=sess.get("created_at", 0),
        updated_at=sess.get("updated_at", 0),
        message_count=len(sess.get("messages", [])),
        messages=[HistoryItem(**m) for m in sess.get("messages", [])],
    )


@router.post("", summary="创建新会话")
async def create_session(payload: Optional[dict] = None, user: str = Depends(get_current_user)):
    payload = payload or {}
    sid = "sess_" + uuid.uuid4().hex[:12]
    now = time.time()
    sess = {
        "title": payload.get("title", "新会话"),
        "kb_id": payload.get("kb_id", "default"),
        "mode": payload.get("mode", "rag"),
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    data = _load(user)
    data[sid] = sess
    _save(user, data)
    return {"session_id": sid}


@router.post("/{session_id}/append", summary="追加一条消息")
async def append_message(session_id: str, item: HistoryItem, user: str = Depends(get_current_user)):
    data = _load(user)
    sess = data.get(session_id)
    if not sess:
        sess = {
            "title": item.content[:30] or "新会话",
            "kb_id": "default",
            "mode": "rag",
            "created_at": time.time(),
            "updated_at": time.time(),
            "messages": [],
        }
        data[session_id] = sess
    sess["messages"].append(item.model_dump())
    sess["updated_at"] = time.time()
    if sess.get("title") == "新会话" and item.role == "user":
        sess["title"] = item.content[:30] or "新会话"
    data[session_id] = sess
    _save(user, data)
    return {"session_id": session_id, "message_count": len(sess["messages"])}


@router.delete("/{session_id}", summary="删除会话")
async def delete_session(session_id: str, user: str = Depends(get_current_user)):
    data = _load(user)
    if session_id in data:
        del data[session_id]
        _save(user, data)
    return {"ok": True}
