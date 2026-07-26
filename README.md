# UniKB - 通用企业级 RAG 知识库平台

> Multi-Agent + MCP + Hybrid Search + Full-Stack。

[![CI](https://github.com/USTCCB/UniKB/actions/workflows/ci.yml/badge.svg)](https://github.com/USTCCB/UniKB/actions) [![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]() [![FastAPI](https://img.shields.io/badge/FastAPI-0.116%2B-009688)]() [![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange)]() [![MCP](https://img.shields.io/badge/MCP-1.0-purple)]() [![License](https://img.shields.io/badge/license-MIT-green)]()

## 简介

UniKB 是一个面向企业知识管理场景的 RAG（Retrieval-Augmented Generation）平台。它把本地文档知识通过多 Agent 协作、MCP 工具协议、混合检索与重排序，最终以流式、带检索片段的方式回答用户问题。

支持 DeepSeek / Qwen / OpenAI 等多种 LLM 灵活切换。

## 核心特性

- 混合检索：BM25 + 向量语义 + RRF 融合
- 精排重排：Cross-Encoder（BGE-reranker）抑制幻觉
- 多 Agent 协作：基于 LangGraph 的 Planner / Retriever / Coder / Reviewer 流程
- MCP 协议：原生支持 Model Context Protocol，可扩展工具集
- 文档解析：PDF、DOCX、Markdown、TXT 与图片 OCR（图片 OCR 需要本机安装 Tesseract）
- 流式问答：SSE 协议 + 多轮对话管理 + 引用溯源
- 工程化：JWT 鉴权 + API Key + Docker Compose 一键部署 + GitHub Actions CI + GHCR 镜像发布 CD
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
| 单元测试 | `test_*.py` (除 integration) | 算法/schema/auth/history/smoke/MCP noop 等 | 不依赖 torch / chromadb / sentence-transformers / 真实 LLM |
| 集成测试 | `test_integration_retrieval.py` | HybridRetriever add/retrieve/delete + RRF + rerank + pipeline 端到端 + agent 模式 | 走 `tests/_fakes.py` fake 路径, 无需重包 |

CI 已拆分为两个并行 job:

- `backend-unit-test`: 跑 63 个单元测试 + lint + 覆盖率
- `backend-integration-test`: 跑 8 个集成测试, 验证完整检索/生成链路

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
