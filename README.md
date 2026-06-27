# UniKB - 通用企业级 RAG 知识库平台

> Multi-Agent + MCP + Hybrid Search + Full-Stack，可作为 2026 届 AI 应用开发岗的项目作品。

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]() [![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688)]() [![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange)]() [![MCP](https://img.shields.io/badge/MCP-1.0-purple)]() [![License](https://img.shields.io/badge/license-MIT-green)]()

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
- 可观测性：内置 LangFuse 对接（可选）
- 评估体系：集成 RAGAS 自动评估

## 技术栈

| 层级 | 选型 |
|---|---|
| LLM | DeepSeek / Qwen / OpenAI（可配置切换） |
| Agent 框架 | LangChain + LangGraph |
| MCP 协议 | mcp 1.0+（stdio + SSE 传输） |
| 后端 | FastAPI + Uvicorn（异步） |
| 前端 | Next.js 14 + TypeScript + shadcn/ui |
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
|         Chat UI  Upload  Sources  History                        |
+--------------------------+----------------------------------------+
                           |  HTTPS / SSE
+--------------------------v----------------------------------------+
|                   FastAPI (Backend)                              |
|  +----------+  +----------+  +-------------+  +----------+       |
|  |   Auth   |  |Documents |  |  Chat / SSE |  |  Admin   |       |
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
|        +----------+--------+----------+                            |
|        v          v        v                                       |
|    Chroma    PostgreSQL   Redis                                     |
+--------------------------------------------------------------------+
```

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

## 项目结构

```
UniKB/
+- backend/                # FastAPI 后端
|  +- app/
|  |  +- api/            # 路由层
|  |  +- core/           # 配置、鉴权、日志
|  |  +- rag/            # 文档解析、切片、检索、重排
|  |  +- agents/         # LangGraph 多 Agent
|  |  +- mcp/            # MCP 工具集
|  |  +- schemas/        # Pydantic 模型
|  +- tests/              # 单元测试 + RAGAS 评估
|  +- Dockerfile
|  +- requirements.txt
+- frontend/               # Next.js 前端
+- docs/                   # 架构、面试讲解
+- data/samples/           # 示例文档
+- docker-compose.yml
+- .github/workflows/      # CI/CD
+- LICENSE
+- README.md
```

## 评估与质量

```bash
cd backend
python -m tests.run_ragas_eval --kb default --dataset data/eval/qa.jsonl
```

输出 faithfulness / answer_relevancy / context_precision / context_recall 四个核心指标。

## Roadmap

- [x] 多 LLM 路由
- [x] BM25 + 向量混合检索 + RRF
- [x] LangGraph 多 Agent
- [x] MCP 协议适配
- [x] SSE 流式问答
- [ ] 用户 / 知识库多租户
- [ ] RAGAS 自动回归
- [ ] LangFuse 线上可观测面板

## License

MIT