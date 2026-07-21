"""测试 7: FastAPI 应用启动 + 路由注册."""
from __future__ import annotations

from fastapi import FastAPI

from app.main import app


def _all_paths(application: FastAPI) -> set[str]:
    """递归收集所有路由 path, 兼容 fastapi 0.116+ 的 _IncludedRouter."""
    out: set[str] = set()
    for r in application.routes:
        p = getattr(r, "path", None)
        if isinstance(p, str):
            out.add(p)
        # _IncludedRouter 把原始 router 存在 original_router
        orig = getattr(r, "original_router", None)
        if orig is not None:
            for sr in getattr(orig, "routes", []):
                sp = getattr(sr, "path", None)
                if isinstance(sp, str):
                    out.add(sp)
    return out


def test_app_is_fastapi_instance():
    assert isinstance(app, FastAPI)


def test_app_has_title():
    assert app.title  # type: ignore[attr-defined]


def test_app_exposes_routes():
    paths = _all_paths(app)
    assert "/health" in paths, f"missing /health, got {paths}"
    assert any(p.startswith("/api/v1/") for p in paths), f"no /api/v1 routes, got {paths}"


def test_app_has_history_router():
    paths = _all_paths(app)
    assert "/api/v1/history" in paths, f"history router not registered, got {paths}"


def test_openapi_lists_all_endpoints():
    schema = app.openapi()
    paths = set(schema.get("paths", {}).keys())
    for required in ("/health", "/api/v1/chat/stream", "/api/v1/history", "/api/v1/documents/upload"):
        assert required in paths, f"OpenAPI missing {required}, got {paths}"
