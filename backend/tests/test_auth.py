"""测试 5: JWT 鉴权."""
from app.core.auth import create_token, decode_token


def test_create_and_decode_token_roundtrip():
    token = create_token(user_id="u-1", expires_min=10)
    payload = decode_token(token)
    assert payload["sub"] == "u-1"
    assert "exp" in payload


def test_decode_invalid_token_raises():
    import pytest

    with pytest.raises(Exception):
        decode_token("not.a.valid.jwt")


def test_token_expiry_in_future():
    token = create_token(user_id="u-2", expires_min=60)
    payload = decode_token(token)
    import time

    assert payload["exp"] > int(time.time())