"""Application configuration loaded from environment variables.

.env.example 中列出的所有 key 都在这里集中映射, 业务代码通过 settings.xxx 访问。
"""
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

    # ===== App =====
    app_env: Literal["dev", "prod", "test"] = "dev"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_log_level: str = "INFO"

    # ===== LLM (三个 provider 可并存) =====
    llm_provider: Literal["deepseek", "qwen", "openai"] = "deepseek"

    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    qwen_api_key: Optional[str] = None
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen-plus"

    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    # 兼容旧字段
    @property
    def default_llm_provider(self) -> str:
        return self.llm_provider

    @property
    def default_llm_model(self) -> str:
        return {
            "deepseek": self.deepseek_model,
            "qwen": self.qwen_model,
            "openai": self.openai_model,
        }[self.llm_provider]

    # ===== Embedding / Rerank =====
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: str = "cpu"
    rerank_model: str = "BAAI/bge-reranker-base"

    # ===== Storage =====
    database_url: str = "sqlite:///./data/unikb.db"
    redis_url: str = "redis://localhost:6379/0"
    chroma_persist_dir: str = "./data/chroma"
    file_storage_dir: str = "./data/files"

    # ===== Auth =====
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24

    # ===== Knowledge Base ACL =====
    # 公共只读知识库白名单. pydantic-settings 对 list 默认要求 JSON 格式,
    # 这里用 str, 由 kb_registry 自行解析 (支持逗号分隔或 JSON 数组).
    # 这些 kb_id 任何登录用户可读, 不可写入; 上传文档时若 kb 不存在会自动创建并归属当前用户.
    # 默认含 "default" 以兼容 demo.
    public_kb_ids: str = "default"

    # ===== CORS =====
    # 允许跨域的前端 origin 白名单 (逗号分隔 / JSON 数组).
    # 严禁 "*" + credentials=True (违反 CORS 规范, 也是明显的安全反模式).
    # 多个 origin 用逗号分隔, 例如 "http://localhost:3000,https://app.example.com"
    cors_allowed_origins: str = "http://localhost:3000"

    # ===== Rate Limit =====
    # 同一账号 login 失败 5 次 / 5 分钟, 锁定 15 分钟 (返回 429).
    rate_limit_login_window_sec: int = 300
    rate_limit_login_fail_max: int = 5
    rate_limit_login_lock_sec: int = 900
    # chat 接口每用户每分钟请求上限 (默认 20).
    rate_limit_chat_per_min: int = 20
    # 是否启用限流 (测试时可关掉以免 flaky)
    rate_limit_enabled: bool = True

    # ===== MCP =====
    mcp_enabled: bool = True
    mcp_servers_config: str = "./mcp_servers.json"
    mcp_transport: str = "stdio"
    mcp_server_name: str = "unikb"

    # ===== Observability =====
    langfuse_enabled: bool = False
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # ===== RAG 切分 / 检索参数 =====
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=80, ge=0, le=400)
    top_k_vector: int = 20
    top_k_bm25: int = 20
    top_k_final: int = 5

    # ===== RAG 质量调优 (来自 RAG 知识库工程实践) =====
    # 1) 文档清洗: 占 RAG 质量调优 ~60% 精力, 42 条正则规则去除噪声.
    cleaner_enabled: bool = True
    # 2) 自适应切分: 短文档整篇保留为 1 个 chunk; 长文档按 adaptive_chunk_size 切.
    adaptive_chunking_enabled: bool = True
    adaptive_chunk_size: int = Field(default=5500, ge=500, le=8000)
    adaptive_short_doc_chars: int = Field(default=500, ge=100)
    # 3) 元数据增强 / HyDE: 入库存假设性问题 (hypothetical question) 提升召回.
    hyde_enabled: bool = True
    # 入库侧 HyDE 是"每 chunk 一次 LLM 调用", 大文档会把上传拖成分钟级并烧 token.
    # 因此只对前 N 个 chunk 走 LLM, 其余回退抽取式首句 (metadata 仍完整, 只是弱一些);
    # 设为 0 表示入库侧完全不调 LLM (纯离线入库).
    hyde_index_max_chunks: int = Field(default=20, ge=0, le=500)
    # 这 N 个 chunk 的 LLM 调用并发度, 避免串行等待也避免打爆 provider 限流.
    hyde_index_concurrency: int = Field(default=4, ge=1, le=32)
    # 4) 父文档召回 (连坐召回): 召回某 chunk 时一并召回同 doc 的兄弟 chunk.
    parent_recall_enabled: bool = True
    parent_recall_extra: int = Field(default=6, ge=0, le=32)
    # 5) 重排优化: 候选池 Top-60 -> 重排后 Top-25 (控制 Cross-Encoder 算力).
    rerank_candidate_pool: int = Field(default=60, ge=10, le=200)
    rerank_top_n: int = Field(default=25, ge=5, le=100)
    # 6) 查询改写 (Query Rewriting): LLM 改写/拆解问题提升召回.
    query_rewrite_enabled: bool = True

    # ===== Upload =====
    # 单文件最大字节数, 防止恶意上传把磁盘填爆或拖死 embedding.
    # 默认 25MB, 可通过环境变量调整. 超出时返回 413.
    upload_max_bytes: int = Field(default=25 * 1024 * 1024, ge=1024)

    def get_llm_api_key(self, provider: str) -> Optional[str]:
        return {
            "deepseek": self.deepseek_api_key,
            "qwen": self.qwen_api_key,
            "openai": self.openai_api_key,
        }.get(provider)

    def get_llm_base_url(self, provider: str) -> str:
        return {
            "deepseek": self.deepseek_base_url,
            "qwen": self.qwen_base_url,
            "openai": self.openai_base_url,
        }.get(provider, self.openai_base_url)

    def get_llm_model(self, provider: str) -> str:
        return {
            "deepseek": self.deepseek_model,
            "qwen": self.qwen_model,
            "openai": self.openai_model,
        }.get(provider, self.openai_model)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
