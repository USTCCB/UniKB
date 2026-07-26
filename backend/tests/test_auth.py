"""测试 5: JWT 鉴权."""
from __future__ import annotations

import time
from datetime import timedelta

from jose import jwt

from app.core.security import create_access_token, verify_token


def test_create_and_verify_token_roundtrip():
    token = create_access_token(subject="u-1")
    assert isinstance(token, str) and len(token) > 0
    assert verify_token(token) == "u-1"


def test_verify_returns_none_for_invalid_token():
    assert verify_token("not.a.valid.jwt") is None


def test_token_expiry_in_future():
    token = create_access_token(subject="u-2", expires_delta=timedelta(minutes=60))
    payload = jwt.get_unverified_claims(token)
    assert payload["sub"] == "u-2"
    assert payload["exp"] > int(time.time())