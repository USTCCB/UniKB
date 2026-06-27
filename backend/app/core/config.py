"""Application configuration loaded from environment variables."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===== LLM =====
    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = "https://api.deepseek.com"

    dashscope_api_key: Optional[str] = None
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"

    default_llm_provider: Literal["deepseek", "qwen", "openai"] = "deepseek"
    default_llm_model: str = "deepseek-chat"

    # ===== Embedding =====
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: str = "cpu"

    # ===== Rerank =====
    rerank_model: str = "BAAI/bge-reranker-base"

    # ===== Database / Cache / Vector =====
    database_url: str = "sqlite+aiosqlite:///./data/unikb.db"
    redis_url: str = "redis://localhost:6379/0"
    chroma_persist_dir: str = "./data/chroma"

    # ===== Security =====
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24

    # ===== Server =====
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_env: Literal["dev", "prod", "test"] = "dev"
    log_level: str = "INFO"

    # ===== MCP =====
    mcp_enabled: bool = True
    mcp_servers_config: str = "./mcp_servers.json"

    # ===== Observability =====
    langfuse_enabled: bool = False
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # ===== RAG 切分参数 =====
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=80, ge=0, le=400)
    top_k_vector: int = 20
    top_k_bm25: int = 20
    top_k_final: int = 5

    def get_llm_api_key(self, provider: str) -> Optional[str]:
        return {
            "deepseek": self.deepseek_api_key,
            "qwen": self.dashscope_api_key,
            "openai": self.openai_api_key,
        }.get(provider)

    def get_llm_base_url(self, provider: str) -> str:
        return {
            "deepseek": self.deepseek_base_url,
            "qwen": self.qwen_base_url,
            "openai": self.openai_base_url,
        }.get(provider, self.openai_base_url)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
