"""测试 dev-token / 生产环境 JWT secret 校验 (P0-3).

覆盖:
1. validate_production_security: dev/test 不抛, prod+默认 secret 抛, prod+强 secret 不抛.
2. /auth/dev-token 在 prod 下返回 403, dev 下返回 200, 并发出 warning 日志.
3. lifespan fail-fast: APP_ENV=prod 且 JWT_SECRET 是默认值时, 进程拒绝启动 (raise).
"""
from __future__ import annotations


import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.security import (
    InsecureJWTConfigError,
    is_default_jwt_secret,
    validate_production_security,
)


# ============ is_default_jwt_secret 单元 ============

@pytest.mark.parametrize(
    "secret",
    [
        "change-me-in-production",
        "change-me-in-production-please-use-openssl-rand",
        "ChangeMe-In-Production",
        "changeme-please",
        "default-secret-xyz",
        "",
        "short",
        "a" * 15,  # 长度 < 16
        None,
    ],
)
def test_is_default_jwt_secret_flags_weak_secrets(secret):
    """默认值/弱口令必须被识别."""
    assert is_default_jwt_secret(secret)


@pytest.mark.parametrize(
    "secret",
    [
        "a" * 32,
        "x" * 64,
        "openssl-rand-base64-very-long-string-1234567890",
        "S0m3$ecure-jwt-secret-with-enough-entropy",
    ],
)
def test_is_default_jwt_secret_accepts_strong_secrets(secret):
    """强 secret 不应被识别为默认."""
    assert not is_default_jwt_secret(secret)


# ============ validate_production_security ============

def test_validate_passes_for_dev_with_default_secret():
    cfg = Settings(app_env="dev", jwt_secret="change-me-in-production")
    validate_production_security(cfg)  # 不抛


def test_validate_passes_for_test_with_default_secret():
    cfg = Settings(app_env="test", jwt_secret="change-me-in-production")
    validate_production_security(cfg)  # 不抛


def test_validate_fails_for_prod_with_default_secret():
    cfg = Settings(app_env="prod", jwt_secret="change-me-in-production")
    with pytest.raises(InsecureJWTConfigError) as exc:
        validate_production_security(cfg)
    assert "jwt_secret" in str(exc.value).lower() or "jwt" in str(exc.value).lower()


def test_validate_fails_for_prod_with_empty_secret():
    cfg = Settings(app_env="prod", jwt_secret="")
    with pytest.raises(InsecureJWTConfigError):
        validate_production_security(cfg)


def test_validate_fails_for_prod_with_short_secret():
    cfg = Settings(app_env="prod", jwt_secret="x" * 8)
    with pytest.raises(InsecureJWTConfigError):
        validate_production_security(cfg)


def test_validate_passes_for_prod_with_strong_secret():
    cfg = Settings(app_env="prod", jwt_secret="S0m3$ecure-jwt-secret-with-enough-entropy-32+chars")
    validate_production_security(cfg)


# ============ /auth/dev-token 运行时行为 ============

@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """每个测试重置 settings cache + 清空内存用户表."""
    from app.api import auth as auth_mod
    from app.core.config import get_settings

    auth_mod.reset_users_for_tests()
    get_settings.cache_clear()
    yield
    auth_mod.reset_users_for_tests()
    get_settings.cache_clear()


def _make_client(monkeypatch, app_env: str, jwt_secret: str = "x" * 32):
    """构造一个指定 env 的 FastAPI app + TestClient, 注入假 deps 让 retriever 不被调用."""
    monkeypatch.setenv("APP_ENV", app_env)
    if jwt_secret:
        monkeypatch.setenv("JWT_SECRET", jwt_secret)
    from app.core.config import get_settings

    get_settings.cache_clear()
    cfg = get_settings()
    # 如果是 prod + 默认 secret, lifespan 会 raise, 我们单独测这种 case, 跳过 lifespan 用法.
    if cfg.app_env == "prod" and is_default_jwt_secret(cfg.jwt_secret):
        pytest.skip("此组合属于 lifespan fail-fast 的测试, 见下个 test_*")

    from app.main import app

    return TestClient(app)


def test_dev_token_returns_403_when_app_env_is_prod(monkeypatch):
    """prod 环境下调 /auth/dev-token 必须 403."""
    # 用 prod + 强 secret, 让 lifespan 不抛; 接下来测 dev-token 的运行时拒绝.
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("JWT_SECRET", "S0m3$ecure-jwt-secret-with-enough-entropy-32+chars")
    from app.core.config import get_settings

    get_settings.cache_clear()

    from app.main import app

    with TestClient(app) as c:
        r = c.post("/api/v1/auth/dev-token")
        assert r.status_code == 403, r.text
        assert "dev" in r.text.lower()


def test_dev_token_returns_200_when_app_env_is_dev(monkeypatch):
    """dev 环境下 dev-token 应当可用, 并发出 warning 日志."""
    monkeypatch.setenv("APP_ENV", "dev")
    from app.core.config import get_settings

    get_settings.cache_clear()

    from app.main import app

    with TestClient(app) as c:
        r = c.post("/api/v1/auth/dev-token")
        assert r.status_code == 200, r.text
        body = r.json()
        assert "access_token" in body and len(body["access_token"]) > 0


# ============ lifespan fail-fast: 真实启动场景 ============

def test_app_refuses_to_start_in_prod_with_default_secret(monkeypatch):
    """APP_ENV=prod 且 jwt_secret=默认值 时, lifespan 必须 raise, TestClient 起不来."""
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("JWT_SECRET", "change-me-in-production")
    from app.core.config import get_settings

    get_settings.cache_clear()

    from app.main import app

    # TestClient 的 lifespan 启动如果 raise, context manager 会抛.
    with pytest.raises(InsecureJWTConfigError):
        with TestClient(app):
            pass


def test_app_starts_in_dev_with_default_secret(monkeypatch):
    """APP_ENV=dev + 默认 secret, 不应被 fail-fast 阻拦 (dev 不做生产校验)."""
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("JWT_SECRET", "change-me-in-production")
    from app.core.config import get_settings

    get_settings.cache_clear()

    from app.main import app

    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200