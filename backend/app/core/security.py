"""JWT auth + password hashing."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from loguru import logger
from passlib.context import CryptContext

from app.core.config import settings

_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


# 标记"未改的默认值". 当 prod 环境检测到 jwt_secret 仍是这一类值时, 必须 fail-fast
# 启动, 不允许运行时静默允许 /auth/dev-token 这类口子.
_DEFAULT_JWT_SECRET_PREFIXES = (
    "change-me",
    "changeme",
    "please-change",
    "default-secret",
)


class InsecureJWTConfigError(RuntimeError):
    """生产环境检测到 JWT 配置不安全 (例如 jwt_secret 是默认值)."""


def hash_password(password: str) -> str:
    return _pwd_ctx.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_ctx.verify(plain, hashed)


def create_access_token(subject: str, expires_delta: Optional[timedelta] = None) -> str:
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.jwt_expire_minutes)
    )
    payload = {"sub": subject, "exp": expire, "iat": datetime.now(timezone.utc)}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def verify_token(token: str) -> Optional[str]:
    """Return subject (user id) or None if invalid."""
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return payload.get("sub")
    except JWTError:
        return None


def is_default_jwt_secret(secret: str) -> bool:
    """判断 jwt_secret 是否还是 config.py 里的默认值/弱口令. 用于 prod fail-fast."""
    if not secret:
        return True
    s = secret.strip().lower()
    if not s:
        return True
    if any(s.startswith(p) for p in _DEFAULT_JWT_SECRET_PREFIXES):
        return True
    # 长度 < 16 也视为弱 (生产建议 ≥ 32 字符随机串)
    if len(secret) < 16:
        return True
    return False


def validate_production_security(cfg=None) -> None:
    """生产环境安全校验. 不通过抛 InsecureJWTConfigError, 调用方应当 fail-fast.

    检查项:
    - app_env == "prod" 且 jwt_secret 是默认/弱口令 → 拒绝启动
    - app_env == "prod" 且 app_env 真被设置 (避免遗忘)
    """
    if cfg is None:
        from app.core.config import settings as _default_cfg

        cfg = _default_cfg
    if cfg.app_env != "prod":
        return
    if is_default_jwt_secret(cfg.jwt_secret):
        raise InsecureJWTConfigError(
            "生产环境 (APP_ENV=prod) 检测到 jwt_secret 仍是默认值或弱口令, "
            "请通过环境变量 JWT_SECRET 设置一个高熵随机串 (建议 ≥ 32 字符). "
            "这通常意味着 .env 没有被正确加载, 启动必须 fail-fast, 而不是运行时留后门."
        )
