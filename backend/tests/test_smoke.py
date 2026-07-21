"""测试 7: FastAPI 应用启动 + 路由注册."""
from __future__ import annotations

from fastapi import FastAPI

from app.main import app


def test_app_is_fastapi_instance():
    assert isinstance(app, FastAPI)


def test_app_has_title():
    assert app.title  # type: ignore[attr-defined]


def test_app_exposes_routes():
    paths = {r.path for r in app.routes}
    # 必须至少包含健康检查 (其它 api 路由也应在)
    assert any("/api" in p for p in paths) or "/" in paths