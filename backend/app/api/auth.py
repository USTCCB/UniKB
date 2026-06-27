"""Auth API: 极简注册 / 登录（演示用，生产请接企业 SSO / OAuth）。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from loguru import logger

from app.core.config import settings
from app.core.security import create_access_token, hash_password, verify_password
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

# 内存用户表（演示）；生产请用 PostgreSQL + SQLAlchemy
_USERS: dict[str, dict] = {}


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
    """仅在 APP_ENV=dev 时可用，方便本地调试。"""
    if settings.app_env != "dev":
        raise HTTPException(status_code=403, detail="仅 dev 模式可用")
    token = create_access_token(subject="dev-user")
    return TokenResponse(access_token=token, expires_in=settings.jwt_expire_minutes * 60)
