"""FastAPI application entry."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import auth, chat, documents, health, history
from app.core.config import get_settings
from app.core.logging import logger
from app.core.security import InsecureJWTConfigError, validate_production_security


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

    logger.info(f"UniKB v{__version__} starting... env={cfg.app_env}")
    yield
    logger.info("UniKB shutting down.")


app = FastAPI(
    title="UniKB - 通用企业级 RAG 知识库平台",
    description="Multi-Agent + MCP + Hybrid Search + Full-Stack",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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