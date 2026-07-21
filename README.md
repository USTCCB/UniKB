# UniKB - 通用企业级 RAG 知识库平台

> Multi-Agent + MCP + Hybrid Search + Full-Stack。

[![CI](https://github.com/USTCCB/UniKB/actions/workflows/ci.yml/badge.svg)](https://github.com/USTCCB/UniKB/actions) [![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]() [![FastAPI](https://img.shields.io/badge/FastAPI-0.116%2B-009688)]() [![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange)]() [![MCP](https://img.shields.io/badge/MCP-1.0-purple)]() [![License](https://img.shields.io/badge/license-MIT-green)]()

## 简介

UniKB 是一个面向企业知识管理场景的 RAG（Retrieval-Augmented Generation）平台，核心目标：把 文档/网页/图片 等多源知识，通过 多 Agent 协作 + MCP 工具协议 + 混合检索 + 重排序 + 引用溯源，最终以 流式、可溯源 的方式回答用户问题。

支持 DeepSeek / Qwen / OpenAI 等多种 LLM 灵活切换。

## 核心特性

- 混合检索：BM25 + 向量语义 + RRF 融合
- 精排重排：Cross-Encoder（BGE-reranker）抑制幻觉
- 多 Agent 协作：基于 LangGraph 的 Planner / Retriever / Coder / Reviewer 流程
- MCP 协议：原生支持 Model Context Protocol，可扩展工具集
- 多模态文档解析：PDF（含 OCR）+ Markdown + 图片（VLM 摘要）
- 流式问答：SSE 协议 + 多轮对话管理 + 引用溯源
- 工程化：JWT 鉴权 + API Key + Docker Compose 一键部署 + GitHub Actions CI/CD
- 可观测性：内置 LangFuse 对接（可选，关闭时无副作用）
- 评估体系：集成 RAGAS 自动评估（4 大指标）

## 技术栈

| 层级 | 选型 |
|---|---|
| LLM | DeepSeek / Qwen / OpenAI（可配置切换） |
| Agent 框架 | LangChain + LangGraph |
| MCP 协议 | mcp 1.0+（stdio + SSE 传输） |
| 后端 | FastAPI + Uvicorn（异步） |
| 前端 | Next.js 14 + TypeScript（App Router） |
| 向量库 | Chroma（轻量、可零配置） |
| 检索 | BM25（rank_bm25）+ 向量 + RRF |
| 重排 | Cross-Encoder（BAAI/bge-reranker-base） |
| 数据库 | PostgreSQL（对话历史）/ SQLite（默认） |
| 缓存 | Redis（Embedding / 热点问答） |
| 文件 | 本地 / MinIO 可切换 |
| 可观测 | LangFuse（可选） |
| 评估 | RAGAS |
| 工程化 | Docker Compose + GitHub Actions |

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

`backend/tests/` 下覆盖了以下模块的纯算法/纯逻辑路径，**不依赖重包**（不需 torch / sentence-transformers / chromadb / mcp / ragas 实跑）：

| 模块 | 测试文件 | 覆盖点 |
|---|---|---|
| RAG 切片 | `test_chunker.py`, `test_chunking.py` | 中文/超长/空输入/边界 |
| RRF 融合 | `test_rrf.py`, `test_retriever.py` | 多路/单路/无交集 |
| Cross-Encoder 重排 | `test_reranker.py` | 排序/空输入/score 注入 |
| LangGraph Agent 节点 | `test_agent.py` | Planner / Reviewer + 重写闭环 |
| JWT 鉴权 | `test_auth.py` | 签发/校验/过期 |
| Schema 输入校验 | `test_schemas.py` | 注册/聊天/文档 12 个 case |
| 配置管理 | `test_config_and_security.py` | env 加载/路由/越界 |
| History API | `test_history_api.py` | CRUD/排序/404 |
| Observability | `test_observability.py` | noop 上下文 + pipeline 集成 |
| RAGAS 脚本 | `test_ragas_eval_script.py` | JSONL 加载/格式化 |
| FastAPI 注册 | `test_smoke.py` | app/router/OpenAPI 完整性 |
| MCP server | `test_mcp.py` | 缺包时自动 skip |

CI 跑通 **63 passed, 1 skipped**。要覆盖更深（如真实 Chroma 检索、Cross-Encoder 推理），需在本地 `docker compose up backend` 起服务并下载模型后再跑。

## 快速开始

### 0. 准备

- Python 3.10+
- Node.js 18+（前端）
- 可选：Docker / Docker Compose

### 1. 克隆与安装

```bash
git clone https://github.com/USTCCB/UniKB.git
cd UniKB
```

### 2. 配置环境变量

```bash
cd backend
cp .env.example .env
# 编辑 .env，至少填一个 LLM_API_KEY（DeepSeek / Qwen / OpenAI）
```

### 3. 一键启动（推荐 Docker）

```bash
docker compose up -d
```

- 后端 API：http://localhost:8000
- 前端 UI：http://localhost:3000
- API 文档：http://localhost:8000/docs

可选 profile：

```bash
docker compose --profile pg up -d             # 加 Postgres
docker compose --profile observability up -d  # 加 LangFuse 自托管
```

### 4. 本地开发模式

```bash
# 后端
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# 前端（新终端）
cd frontend
npm install
npm run dev
```

### 5. 跑测试

```bash
cd backend
python -m pytest -v --cov=app
```

## 关键功能演示

### 1) 上传文档 / 自动入库

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload ^
  -H "Authorization: Bearer YOUR_TOKEN" ^
  -F "file=@./data/samples/handbook.pdf"
```

返回 `{ "doc_id": "...", "chunks": 87, "status": "indexed" }`

### 2) 智能问答（流式）

```bash
curl -N http://localhost:8000/api/v1/chat/stream ^
  -H "Authorization: Bearer YOUR_TOKEN" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"产品的保修期是多久？\", \"kb_id\": \"default\"}"
```

### 3) 多 Agent 协作模式

请求体 `{"mode": "agent", ...}`，后端执行 Planner / Retriever / Coder / Reviewer 全链路，返回带中间步骤的轨迹。

### 4) 历史会话

```bash
# 列出
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8000/api/v1/history
# 追加
curl -X POST http://localhost:8000/api/v1/history/<sid>/append \
  -H "Authorization: Bearer YOUR_TOKEN" -H "Content-Type: application/json" \
  -d '{"role": "user", "content": "hi"}'
```

### 5) 检索片段预览（前端 Sources 页用同一接口）

```bash
# 通过 SSE 取 source 事件, 或直接调 /api/v1/chat/stream 设置 question=关键词
```

## 项目结构

```
UniKB/
+- backend/                # FastAPI 后端
|  +- app/
|  |  +- api/            # auth / chat / documents / history / health
|  |  +- core/           # config / logging / security / observability
|  |  +- rag/            # parser / chunker / embedding / retriever / reranker / pipeline
|  |  +- agents/         # LangGraph 多 Agent + tools + llm_router
|  |  +- mcp/            # MCP 工具集
|  |  +- schemas/        # Pydantic 模型
|  +- tests/              # 单元测试 (63 passed) + RAGAS 评估脚本
|  +- Dockerfile
|  +- requirements.txt
+- frontend/               # Next.js 14 (app/ + components/ + lib/)
|  +- app/{chat,upload,sources,history}/page.tsx
|  +- components/{AuthBar,Nav}.tsx
|  +- lib/api.ts
+- docs/                   # 架构、面试讲解
+- data/samples/           # 示例文档
+- docker-compose.yml
+- .github/workflows/      # CI/CD (lint + pytest + coverage + frontend build + docker build)
+- LICENSE
+- README.md
```

## 评估与质量

```bash
cd backend
python -m tests.run_ragas_eval --kb default \
    --dataset ../data/eval/qa.jsonl \
    --out ../data/eval/ragas_report.json
```

输出 faithfulness / answer_relevancy / context_precision / context_recall 四个核心指标 + JSON 报告（含每条样本）。

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

- [x] 多 LLM 路由
- [x] BM25 + 向量混合检索 + RRF
- [x] LangGraph 多 Agent
- [x] MCP 协议适配
- [x] SSE 流式问答
- [x] 多轮对话 / 历史会话 / 引用溯源
- [x] RAGAS 自动评估脚本 + JSON 报告
- [x] LangFuse 可观测（可选）
- [x] Docker Compose 一键起 + healthcheck
- [ ] 用户 / 知识库多租户
- [ ] Postgres 迁移 SQLAlchemy
- [ ] MinIO 文件存储

## License

MIT
