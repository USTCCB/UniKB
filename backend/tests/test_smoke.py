"""测试 7: 应用启动 smoke test."""
from app.main import app


def test_app_starts_and_has_title():
    assert app.title


def test_app_exposes_health_endpoint():
    routes = {r.path for r in app.routes}
    assert "/api/v1/health" in routes


def test_app_uses_fastapi():
    from fastapi import FastAPI

    assert isinstance(app, FastAPI)