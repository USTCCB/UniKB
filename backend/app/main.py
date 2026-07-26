"""FastAPI application entry."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from app import __version__
from app.api import auth, chat, documents, health, history
from app.core.config import get_settings
from app.core.logging import logger
from app.core.security import InsecureJWTConfigError, validate_production_security
from app.db import create_tables


def _parse_cors_origins(raw: str) -> list[str]:
    """解析 CORS 白名单: 支持逗号分隔字符串或 JSON 数组."""
    s = (raw or "").strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            import json as _json
            parsed = _json.loads(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            return []
    return [x.strip() for x in s.split(",") if x.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 通过 get_settings() 实时读取 (而不是模块级缓存), 这样部署时改 env
    # 也能正确触发 fail-fast.
    cfg = get_settings()
    try:
        validate_production_security(cfg)
    except InsecureJWTConfigError as e:
        logger.critical(f"Refusing to start: {e}")
        raise

    # P2-12: 启动时创建用户表 (SQLite 持久化 _USERS 替代内存 dict)
    create_tables()

    logger.info(f"UniKB v{__version__} starting... env={cfg.app_env}")
    yield
    logger.info("UniKB shutting down.")


class SecurityHeadersMiddleware:
    """统一安全响应头.

    P2-11: 当使用 HttpOnly cookie 存 JWT 时, XSS 仍能读出页面内容, 因此需要 CSP
    限制脚本来源; 同时打开 HSTS / X-Content-Type-Options 等基础 header.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def _send(message):
            if message["type"] == "http.response.start":
                headers = message.get("headers") or []
                # 防止 MIME 嗅探
                headers.append((b"x-content-type-options", b"nosniff"))
                # XSS 过滤 (legacy 但无害)
                headers.append((b"x-frame-options", b"DENY"))
                headers.append((b"x-xss-protection", b"1; mode=block"))
                # HSTS (仅 prod + https). dev 环境不强制.
                cfg = get_settings()
                if cfg.app_env == "prod":
                    headers.append((b"strict-transport-security", b"max-age=31536000; includeSubDomains"))
                # CSP: 默认 self only; 允许 eval (一些前端库可能用) 但不允许 inline script.
                # 生产请根据实际前端静态资源域名收紧 script-src / style-src / connect-src.
                headers.append((
                    b"content-security-policy",
                    b"default-src 'self'; script-src 'self' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; connect-src 'self';",
                ))
                # 禁止在请求 / 响应中携带来源 URL, 降低 referrer 泄漏风险.
                headers.append((b"referrer-policy", b"strict-origin-when-cross-origin"))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, _send)


app = FastAPI(
    title="UniKB - 通用企业级 RAG 知识库平台",
    description="Multi-Agent + MCP + Hybrid Search + Full-Stack",
    version=__version__,
    lifespan=lifespan,
)

# 安全: 严禁 allow_origins=["*"] 与 allow_credentials=True 同时使用 (违反 CORS 规范).
# origin 严格走 settings.cors_allowed_origins 白名单.
_cors_origins = _parse_cors_origins(get_settings().cors_allowed_origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局安全响应头
app.add_middleware(SecurityHeadersMiddleware)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(history.router)


if __name__ == "__main__":
    import uvicorn

    from app.core.config import settings

    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=(settings.app_env == "dev"),
    )