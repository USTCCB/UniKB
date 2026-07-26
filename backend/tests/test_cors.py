"""测试 CORS 配置 (P0-4).

覆盖:
1. allow_origins 不再是 "*".
2. 解析器支持逗号分隔 / JSON 数组 / 空字符串.
3. 默认值是 http://localhost:3000.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import _parse_cors_origins, app


import pytest


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("http://localhost:3000", ["http://localhost:3000"]),
        ("http://localhost:3000,https://app.example.com", ["http://localhost:3000", "https://app.example.com"]),
        ("", []),
        ("   ", []),
        ('["http://a.com", "https://b.com"]', ["http://a.com", "https://b.com"]),
        ("http://a.com, https://b.com", ["http://a.com", "https://b.com"]),  # 带空格
    ],
)
def test_parse_cors_origins(raw, expected):
    assert _parse_cors_origins(raw) == expected


def test_parse_cors_origins_rejects_invalid_json_falls_back_empty():
    """无法解析的 JSON 数组 (含语法错误) 不能崩, 退化为空列表."""
    assert _parse_cors_origins("[" ) == []  # 不平衡
    assert _parse_cors_origins("[not-json]") == []


def test_cors_does_not_use_wildcard_with_credentials():
    """核心安全断言: 生产 CORS 中间件不能是 allow_origins=['*'] + credentials=True."""
    from starlette.middleware.cors import CORSMiddleware

    found = False
    for m in app.user_middleware:
        cls = getattr(m, "cls", None) or type(m)
        if cls is CORSMiddleware:
            found = True
            # Starlette 在不同版本上挂参的方式不同: 直接放在对象上 or 放在 kwargs.
            origins = getattr(m, "allow_origins", None)
            if origins is None:
                origins = m.kwargs.get("allow_origins") if hasattr(m, "kwargs") else None
            creds = getattr(m, "allow_credentials", None)
            if creds is None:
                creds = m.kwargs.get("allow_credentials") if hasattr(m, "kwargs") else None
            assert origins is not None and len(origins) > 0, (
                f"CORSMiddleware allow_origins 必须非空, 实际 {origins!r}"
            )
            assert origins != ["*"], f"CORS 仍使用通配符: {origins}"
            assert "*" not in (origins or []), f"allow_origins 含 '*': {origins}"
            assert creds is True
    assert found, "CORSMiddleware 没注册到 app 上"


def test_cors_allowed_origins_default_value():
    """默认配置应该是 http://localhost:3000, 而不是 '*'."""
    from app.core.config import Settings

    s = Settings()
    assert "localhost:3000" in s.cors_allowed_origins
    assert "*" not in s.cors_allowed_origins


def test_cors_headers_actually_set_on_request():
    """集成: 来自白名单 origin 的请求应当拿到 Access-Control-Allow-Origin 头."""
    with TestClient(app) as c:
        # OPTIONS 预检
        r = c.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # 来自白名单 origin, 应有 allow-origin 头
        assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_rejects_non_whitelisted_origin():
    """非白名单 origin 不应拿到 Access-Control-Allow-Origin 回显 (安全要求)."""
    with TestClient(app) as c:
        r = c.options(
            "/health",
            headers={
                "Origin": "http://evil.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        # 不应该回显恶意 origin
        assert r.headers.get("access-control-allow-origin") != "http://evil.example.com"