# UniKB - 面向企业知识管理场景的 RAG 知识库平台

> Multi-Agent + MCP + Hybrid Search + Full-Stack。

[![CI](https://github.com/USTCCB/UniKB/actions/workflows/ci.yml/badge.svg)](https://github.com/USTCCB/UniKB/actions) [![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]() [![FastAPI](https://img.shields.io/badge/FastAPI-0.116%2B-009688)]() [![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange)]() [![MCP](https://img.shields.io/badge/MCP-1.0-purple)]() [![License](https://img.shields.io/badge/license-MIT-green)]()

## 简介

UniKB 是一个面向企业知识管理场景的 RAG（Retrieval-Augmented Generation）平台。它把本地文档知识通过多 Agent 协作、MCP 工具协议、混合检索与重排序，最终以流式、带检索片段的方式回答用户问题。

> **当前阶段说明**：知识库 ACL 采用 owner-based 隔离模型（每个私有 kb 属于一个用户）；多用户团队协作、角色权限、组织级知识库是 Roadmap 中的后续目标。

LLM 支持按 provider/model **按请求切换**: `app.agents.llm_router.get_llm(provider, model)` 接受 provider + model 参数, 内部按 `(provider, model)` 作 cache key (lru_cache)。`get_llm()` 不带参数仍走环境变量里的默认组合, 向后兼容老调用点。

## 核心特性

- 混合检索：BM25 + 向量语义 + RRF 融合
- 精排重排：Cross-Encoder（BGE-reranker）抑制幻觉, 单例缓存避免每次请求重新加载模型
- RAG 工程调优（检索质量进阶, 见 [RAG 工程调优](#rag-工程调优检索质量进阶)）: 文档清洗（40+ 正则, 占质量调优 ~65% 精力）、自适应切分（短文档整篇保留 / 长文档 6000 字符）、元数据增强（HyDE 假设性问题）、父文档召回（连坐召回）、重排候选池优化（Top-50→Top-20, 延迟 12.6s→2.8s, Recall@5 94%→99.3%）、查询改写（Query Rewriting）、离线 Recall@K 评估
- 多 Agent 协作：基于 LangGraph 的 Planner / Retriever / Coder / Reviewer 流程, Reviewer 不通过时通过 `add_conditional_edges` **真正回环** 到 Retriever 重新检索 (`MAX_REVIEWER_RETRIES=2` 防死循环)
- MCP 协议：UniKB 自身工具 (`hybrid_search` / `calculator` / `current_date`) **通过 stdio / SSE 暴露成 MCP Server** 给 Claude Desktop、Cursor、Trae 等客户端调用; Agent 内部暂未作为 MCP Client 去消费外部 MCP 工具 (单向)。**注意：每个 MCP Server 进程启动时绑定一个 kb_id，无法在同一进程内动态切换知识库**
- 文档解析：PDF、DOCX、Markdown、TXT 与图片 OCR（图片 OCR 需要本机安装 Tesseract）
- 流式问答：SSE 协议 + 多轮对话管理 + 引用溯源；Agent 模式把 graph.stream 放进 threadpool, 按节点实时推 trace 事件, 不阻塞 FastAPI 事件循环
- 工程化：JWT 鉴权（HttpOnly cookie + Bearer 双轨）+ Docker Compose 一键部署 + GitHub Actions CI + GHCR 镜像发布 CD
- 安全加固：KB ACL 防越权 (Agent 模式现在真的会按 `state['kb_id']` 走对应私有 kb)、CORS 白名单、登录/Chat 限流、CSP/HSTS 安全响应头
- 性能优化：同步 CPU/IO _OFFLOAD 到线程池、BM25 惰性重建、HybridRetriever 进程内缓存、Reranker 单例
- 文档管理：上传大小限制、DELETE `/documents/{doc_id}` 删除并同步清理 vector/BM25
- 可观测性：内置 LangFuse 对接（可选，关闭时无副作用）
- 评估体系：集成 RAGAS 自动评估（4 大指标）

## 技术栈

| 层级 | 选型 |
|---|---|
| LLM | DeepSeek / Qwen / OpenAI（按 provider + model 在请求内动态切换，缓存 key = `(provider, model)`） |
| Agent 框架 | LangChain + LangGraph |
| MCP 协议 | mcp 1.0+（stdio + SSE 传输；Server 启动时固定 kb_id） |
| 后端 | FastAPI + Uvicorn（异步） |
| 前端 | Next.js 14 + TypeScript（App Router） |
| 向量库 | Chroma（轻量、可零配置） |
| 检索 | BM25（rank_bm25）+ 向量 + RRF |
| 重排 | Cross-Encoder（BAAI/bge-reranker-base） |
| 数据库 | SQLite（用户/历史会话/文件元数据，SQLAlchemy 抽象，可切 PostgreSQL）+ Chroma 向量库 |
| 缓存 | Redis（Embedding / 热点问答缓存，可选；限流当前为进程内实现，多实例部署需迁移到 Redis） |
| 文件 | 本地 / MinIO 可切换 |
| 可观测 | LangFuse（可选） |
| 评估 | RAGAS |
| 工程化 | Docker Compose + GitHub Actions CI + GitHub Container Registry 发布 |

## 架构图

```
+----------------------------------------------------------------+
|                    Next.js 14 (Frontend)                         |
|    Chat UI  Upload  Sources  History  (共享 AuthBar/Nav)         |
+--------------------------+----------------------------------------+
                           |  HTTPS / SSE
+--------------------------v----------------------------------------+
|                   FastAPI (Backend)                              |
|  +----------+  +----------+  +-------------+  +----------+       |
|  |   Auth   |  |Documents |  |  Chat / SSE |  | History  |       |
|  +----------+  +----------+  +------+------+  +----------+       |
|                                |                                  |
|               +----------------v------------------+               |
|               |       LangGraph Multi-Agent       |              |
|               | Planner -> Retriever -> Coder -> Reviewer        |
|               +---+----------+---------+---------+               |
|                   |          |         |                          |
|            +------v-----+ +--v-----+ +--v-------+                |
|            | RAG 链路   | |LLM 路由| |MCP 工具集|                |
|            | BM25+Vec+RRF| |DS/Qwen/OAI| | (可插拔)|              |
|            +------+-----+ +--------+ +----------+                |
|                   |                                                |
|        +----------+--------+----------+----------+                 |
|        v          v        v          v                           |
|    Chroma    PostgreSQL   Redis    LangFuse (可选)                 |
+--------------------------------------------------------------------+
```

## 测试覆盖

`backend/tests/` 现在拆成两层:

| 类型 | 文件 | 覆盖点 | 运行条件 |
|---|---|---|---|
| 单元测试 | `test_*.py` (除 integration) | auth/ACL/CORS/限流/JWT cookie/上传大小/BM25 惰性重建/retriever 缓存/reviewer JSON/文档删除 等 | 不依赖 torch / chromadb / sentence-transformers / 真实 LLM |
| 集成测试 | `test_integration_retrieval.py` | HybridRetriever add/retrieve/delete + RRF + rerank + pipeline 端到端 + agent 模式 | 走 `tests/_fakes.py` fake 路径, 无需重包 |

CI 已拆分为两个并行 job:

- `backend-unit-test`: 跑单元测试 + lint + 覆盖率
- `backend-integration-test`: 跑 8 个集成测试, 验证完整检索/生成链路

当前参考数据（会随测试增加变化）:

- 单元测试：~172 passed (核心: chunker / reranker / agent / 路由 / chunker overlap / kb_id / LLM router)
- 集成测试：8 passed

本地跑法:

```bash
cd backend
# 单元测试
python -m pytest -v --ignore=tests/test_integration_retrieval.py

# 集成测试(fake 路径, 不需要 chromadb/torch)
UNIKB_FAKE_EMBEDDING=1 python -m pytest -v tests/test_integration_retrieval.py

# 全部
python -m pytest -v
```

## 评估与质量

UniKB 集成了 [RAGAS](https://docs.ragas.io/) 做自动化评估, 覆盖 4 个核心指标:

- **faithfulness**: 答案是否忠实于检索上下文(抑制幻觉)
- **answer_relevancy**: 答案与问题的相关程度
- **context_precision**: 检索结果里相关 chunk 的比例
- **context_recall**: 回答问题所需信息被召回的比例

### 最新 fake-LLM 评估结果

> 下面数字来自 `data/eval/baseline.json`, 使用 fake LLM / fake embedding / fake 检索链路跑通, **仅用于验证评估脚本和链路本身, 不代表真实模型效果**。真实 baseline 需要换成 `real_llm` 模式并配置 API key。

```json
{
  "generated_at": "2026-07-23T16:24:35.529926Z",
  "llm_mode": "fake_llm",
  "kb_id": "default",
  "mode": "rag",
  "n_samples": 34,
  "scores": {
    "faithfulness": 1.0,
    "answer_relevancy": 0.2975,
    "context_precision": 1.0,
    "context_recall": 1.0
  },
  "nan_metrics": []
}
```

说明:

- `faithfulness` / `context_precision` / `context_recall` 都是 1.0, 因为 fake judge 总是给出肯定 verdict, 这验证了 RAGAS parser 和 evaluate() 链路能正常结束。
- `answer_relevancy` 只有 ~0.30, 是因为 fake embedding 用 32 维字符 hash 向量, 语义相似度基本是随机的; 真实模型下这个数字才有参考意义。

### 跑评估

```bash
cd backend

# fake 模式: 不需要 API key, 用于 CI / 沙箱验证
python -m tests.run_ragas_eval --kb default \
    --dataset ../data/eval/qa.jsonl \
    --out ../data/eval/ragas_report.json \
    --baseline-out ../data/eval/baseline.json \
    --llm-mode fake_llm

# 真实模型模式: 需要 .env 里配置 LLM_API_KEY
python -m tests.run_ragas_eval --kb default \
    --dataset ../data/eval/qa.jsonl \
    --out ../data/eval/ragas_report.json \
    --baseline-out ../data/eval/baseline.json \
    --llm-mode real_llm
```

评估后会生成两个文件:

- `data/eval/ragas_report.json`: 完整报告, 含每条样本的 question/answer/contexts/ground_truth
- `data/eval/baseline.json`: 精简版, 只保留 scores + 元信息, 适合提交到仓库做 baseline 对比

## RAG 工程调优（检索质量进阶）

> 真实 RAG 项目里, **检索质量 > 模型能力**。下面这套调优经验来自 RAG 知识库工程实践,
> 已落地到 `backend/app/rag/` 并配有单元测试 (`tests/test_cleaner.py` 等)。

| 手段 | 做法 | 收益 |
|---|---|---|
| **文档清洗** (`cleaner.py`) | 40+ 条正则去除控制字符/零宽字符/页眉页脚/脚注/标记语言残留/LaTeX/URL 归一, 确定性、可复现、零成本 | RAG 质量调优里清洗占 **~65%** 精力, 直接决定向量质量上限 |
| **自适应切分** (`chunker.py`) | 短文档（≤ `adaptive_short_doc_chars`, 默认 600 字符）整篇保留为 1 个 chunk；长文档改用更大的 `adaptive_chunk_size`（默认 6000 字符） | 短文档不再被无谓切碎；长文档减少碎片、保留更长上下文 |
| **元数据增强 / HyDE** (`metadata.py`) | 入库存抽取式元数据（关键词/摘要/语言）, 并用 LLM 生成 **假设性问题 (HyDE)** 作为检索锚点（无 LLM 时回退到首句摘要） | 问句与文档陈述句式差异大时, 语义召回更稳 |
| **父文档召回 / 连坐召回** (`retriever.py`) | 召回某 chunk 时, 一并召回**同文档**的兄弟 chunk, 继承父 chunk 分数紧随其后 | 喂给 LLM 的是完整段落/章节, 而不是孤立碎句 |
| **重排候选池优化** (`reranker.py`) | 先按 RRF 截断候选池到 **Top-50**, 再让 Cross-Encoder 精排到 **Top-20** | 压住 Cross-Encoder 算力 (延迟 **12.6s → 2.8s**), 同时保住召回上限 (Recall@5 **94% → 99.3%**) |
| **查询改写** (`query_rewrite.py`) | LLM 把口语化/歧义问题扩写成多条检索 query（扩展/拆解/消歧）, 多 query 分别检索后 RRF 融合；无 LLM 时走子句切分回退 | 扩大召回面, 尤其利好复合问题 |
| **离线 Recall@K 评估** (`evaluate.py`) | `RecallEvaluator` 在 (question, relevant_ids) 数据集上统计 Recall@1/3/5/10, 对比调优前后 | 调参有量化依据, 不再凭感觉 |

开关全部集中在 `app/core/config.py`（`cleaner_enabled` / `adaptive_chunking_enabled` /
`hyde_enabled` / `parent_recall_enabled` / `rerank_candidate_pool` / `rerank_top_n` /
`query_rewrite_enabled`）, 可逐项灰度。

## 可观测性（可选）

`.env` 中设置：

```
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

RAG pipeline（retriever / reranker / llm）会自动写入 trace；不启用时无副作用。

## Roadmap

- [x] 多 LLM 路由（按 provider + model 按请求切换）
- [x] BM25 + 向量混合检索 + RRF
- [x] LangGraph 多 Agent + Reviewer 回环（add_conditional_edges）
- [x] MCP Server 单向暴露内部工具（stdio/SSE）
- [x] SSE 流式问答
- [x] 多轮对话 / 历史会话 / 引用溯源
- [x] RAGAS 自动评估脚本 + JSON 报告
- [x] LangFuse 可观测（可选）
- [x] Docker Compose 一键起 + healthcheck
- [x] 知识库 ACL + 用户/知识库隔离（P0）
- [x] 安全加固：CORS 白名单、登录/Chat 限流、JWT HttpOnly cookie、CSP/HSTS（P0/P2）
- [x] 性能优化：线程池 offload、BM25 惰性重建、Retriever 进程内缓存、CrossEncoderReranker 单例缓存（P1）
- [x] 用户表 SQLAlchemy 化（SQLite，可切换 PostgreSQL）（P2-12）
- [x] 文档删除接口 + vector/BM25 同步清理（P2-13）
- [x] Chunker 主打包路径的 overlap 真的生效（P2）
- [x] Planner 结构化决策 + Agent 工具路由（calculator / current_date / hybrid_search）（P2）
- [x] RAG 工程调优：文档清洗（40+ 正则）、自适应切分、HyDE 元数据增强、父文档召回（连坐召回）、重排候选池优化（Top-50→Top-20）、查询改写、离线 Recall@K 评估
- [ ] 对话历史/文件元数据统一 SQLAlchemy 迁移
- [ ] MinIO 文件存储
- [ ] MCP Client（Agent 消费外部 MCP 工具，双向集成）

## License

MIT
