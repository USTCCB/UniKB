"""测试 10: Pydantic schemas 输入校验."""
from __future__ import annotations

import pytest

from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse
from app.schemas.chat import ChatRequest, ChatResponse, SourceItem
from app.schemas.document import DocumentUploadResponse


def test_register_request_valid():
    r = RegisterRequest(username="alice", email="a@b.com", password="abcdef")
    assert r.username == "alice"
    assert r.email == "a@b.com"


def test_register_short_username_fails():
    with pytest.raises(Exception):
        RegisterRequest(username="ab", email="a@b.com", password="abcdef")


def test_register_invalid_email_fails():
    with pytest.raises(Exception):
        RegisterRequest(username="abc", email="not-email", password="abcdef")


def test_login_request():
    r = LoginRequest(username="u", password="p")
    assert r.username == "u"


def test_token_response_defaults():
    t = TokenResponse(access_token="x", expires_in=3600)
    assert t.token_type == "bearer"


def test_chat_request_defaults():
    c = ChatRequest(question="hi")
    assert c.kb_id == "default"
    assert c.mode == "rag"
    assert c.top_k == 5
    assert c.stream is False


def test_chat_request_top_k_bounds():
    with pytest.raises(Exception):
        ChatRequest(question="hi", top_k=0)
    with pytest.raises(Exception):
        ChatRequest(question="hi", top_k=100)


def test_chat_request_mode_must_be_known():
    with pytest.raises(Exception):
        ChatRequest(question="hi", mode="mystery")  # type: ignore[arg-type]


def test_chat_request_empty_question_fails():
    with pytest.raises(Exception):
        ChatRequest(question="")


def test_source_item_score_required():
    s = SourceItem(doc_id="d", chunk_id="c", content="x", score=0.5)
    assert s.metadata == {}


def test_chat_response_minimal():
    c = ChatResponse(answer="a", session_id="s1")
    assert c.sources == []
    assert c.usage == {}


def test_document_upload_response():
    d = DocumentUploadResponse(doc_id="d1", filename="a.md", chunks=10)
    assert d.status == "indexed"
