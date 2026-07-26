"""Auth API: 极简注册 / 登录（演示用，生产请接企业 SSO / OAuth）。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, status
from loguru import logger

from app.core.config import get_settings, settings
from app.core.rate_limit import (
    check_login_failure,
    record_login_failure,
    record_login_success,
)
from app.core.security import create_access_token, hash_password, verify_password
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

# 内存用户表（演示）；生产请用 PostgreSQL + SQLAlchemy
_USERS: dict[str, dict] = {}


def reset_users_for_tests() -> None:
    """仅供测试使用: 清空内存用户表, 让每个测试隔离."""
    global _USERS
    _USERS = {}


def _client_ip(request: Request) -> str:
    """提取 client IP, 兼容 X-Forwarded-For (反代场景). 多值取第一个."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    if request.client:
        return request.client.host or "unknown"
    return "unknown"


# cookie 名
_AUTH_COOKIE_NAME = "unikb_token"
# cookie 在生产 (非 dev) 时, secure=True + samesite=strict.
# dev 时 secure=False (因为 http://localhost:3000), samesite=lax (避免跨站干扰).
def _set_auth_cookie(response: Response, token: str, expire_seconds: int) -> None:
    cfg = get_settings()
    is_prod = cfg.app_env == "prod"
    response.set_cookie(
        key=_AUTH_COOKIE_NAME,
        value=token,
        max_age=expire_seconds,
        httponly=True,           # JS 读不到, 防 XSS 窃 token
        secure=is_prod,          # dev 用 http, 不强制 https
        samesite="strict" if is_prod else "lax",
        path="/",
    )


def _clear_auth_cookie(response: Response) -> None:
    cfg = get_settings()
    is_prod = cfg.app_env == "prod"
    response.delete_cookie(
        key=_AUTH_COOKIE_NAME,
        path="/",
        httponly=True,
        secure=is_prod,
        samesite="strict" if is_prod else "lax",
    )


@router.post("/register", response_model=TokenResponse)
async def register(req: RegisterRequest, response: Response):
    if req.username in _USERS:
        raise HTTPException(status_code=400, detail="用户已存在")
    _USERS[req.username] = {
        "email": req.email,
        "password_hash": hash_password(req.password),
    }
    token = create_access_token(subject=req.username)
    # 同时写 cookie (P2-11: HttpOnly, 避免 XSS 窃 token)
    _set_auth_cookie(response, token, settings.jwt_expire_minutes * 60)
    return TokenResponse(access_token=token, expires_in=settings.jwt_expire_minutes * 60)


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, request: Request, response: Response):
    ip = _client_ip(request)
    decision = check_login_failure(req.username, ip)
    if not decision.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=decision.reason,
            headers={"Retry-After": str(int(decision.retry_after) + 1)},
        )
    user = _USERS.get(req.username)
    if not user or not verify_password(req.password, user["password_hash"]):
        record_login_failure(req.username, ip)
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    record_login_success(req.username, ip)
    token = create_access_token(subject=req.username)
    _set_auth_cookie(response, token, settings.jwt_expire_minutes * 60)
    return TokenResponse(access_token=token, expires_in=settings.jwt_expire_minutes * 60)


@router.post("/logout", summary="清空 auth cookie (登出)")
async def logout(response: Response):
    _clear_auth_cookie(response)
    return {"status": "ok"}


@router.post("/dev-token", response_model=TokenResponse, summary="仅 dev 模式：直接拿一个 token 调试用")
async def dev_token(response: Response):
    """仅在 APP_ENV=dev 时可用，方便本地调试。

    生产环境被 fail-fast 启动校验保护 (见 app.main.lifespan),
    这里再加运行时二次校验 + 警告日志, 便于运维监控误用.
    """
    # 用 get_settings() 实时读取, 避免模块级缓存的 settings 与进程 env 不一致
    # (例如测试时动态改 env 的场景).
    cfg = get_settings()
    if cfg.app_env != "dev":
        logger.warning(
            "/auth/dev-token 被拒绝调用: app_env={!r} (期望 'dev'). "
            "如果这是生产环境被攻击, 请检查 APP_ENV / JWT_SECRET 配置.",
            cfg.app_env,
        )
        raise HTTPException(status_code=403, detail="仅 dev 模式可用")
    logger.warning("/auth/dev-token 被调用, 颁发 dev-user token; 仅 dev 环境应启用此接口.")
    token = create_access_token(subject="dev-user")
    # dev 也设 cookie, 方便前端用同一套流程测试
    _set_auth_cookie(response, token, cfg.jwt_expire_minutes * 60)
    return TokenResponse(access_token=token, expires_in=cfg.jwt_expire_minutes * 60)
