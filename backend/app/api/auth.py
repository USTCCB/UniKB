"""Auth API: 极简注册 / 登录（演示用，生产请接企业 SSO / OAuth）。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from loguru import logger

from app.core.config import get_settings, settings
from app.core.security import create_access_token, hash_password, verify_password
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

# 内存用户表（演示）；生产请用 PostgreSQL + SQLAlchemy
_USERS: dict[str, dict] = {}


def reset_users_for_tests() -> None:
    """仅供测试使用: 清空内存用户表, 让每个测试隔离."""
    global _USERS
    _USERS = {}


@router.post("/register", response_model=TokenResponse)
async def register(req: RegisterRequest):
    if req.username in _USERS:
        raise HTTPException(status_code=400, detail="用户已存在")
    _USERS[req.username] = {
        "email": req.email,
        "password_hash": hash_password(req.password),
    }
    token = create_access_token(subject=req.username)
    return TokenResponse(access_token=token, expires_in=settings.jwt_expire_minutes * 60)


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    user = _USERS.get(req.username)
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_access_token(subject=req.username)
    return TokenResponse(access_token=token, expires_in=settings.jwt_expire_minutes * 60)


@router.post("/dev-token", response_model=TokenResponse, summary="仅 dev 模式：直接拿一个 token 调试用")
async def dev_token():
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
    return TokenResponse(access_token=token, expires_in=cfg.jwt_expire_minutes * 60)
