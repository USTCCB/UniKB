# UniKB 架构说明

本文只描述当前仓库已实现、可在代码中核对的能力。

## 处理链路

```
上传文件 → 解析 → 切片 → Embedding → Chroma + BM25
                                          ↓
用户问题 → 向量召回 + BM25 召回 → RRF 融合 → Cross-Encoder 重排
                                          ↓
                              LLM 生成 / SSE 逐 token 返回
```

- 解析器支持 PDF、DOCX、Markdown、TXT 和 PNG/JPG/JPEG。图片使用可选的 Tesseract OCR；PDF 目前提取文本层，不提供扫描件 OCR。
- `HybridRetriever` 同时执行 Chroma 向量召回与 BM25 词项召回，以 RRF 融合，再交给 `CrossEncoderReranker` 精排。
- 普通问答会在 SSE 中先发送检索片段，再逐 token 返回生成内容；非流式接口同时返回结构化 `sources`。

## Agent 模式

`mode=agent` 时，LangGraph 固定执行：

```
Planner → Retriever → Coder → Reviewer
```

Reviewer 合格时直接返回草稿；否则将审查意见交给 Coder 进行一次改写。该模式会在返回体或 SSE trace 事件中附带各节点轨迹。

## MCP

UniKB 以 stdio 方式把内部工具暴露为 MCP server。目前注册的工具是：

| 工具 | 作用 |
|---|---|
| `hybrid_search` | 执行 BM25 + 向量 + RRF + 重排检索 |
| `calculator` | 计算受限的数学表达式 |
| `current_date` | 返回服务端当前时间 |

启动命令与 Claude Desktop / Cursor / Trae 的配置示例请见 [MCP_SETUP.md](MCP_SETUP.md)。

## 工程化

- `docker-compose.yml` 提供后端、前端和 Redis；PostgreSQL 与 LangFuse 为可选 profile。
- GitHub Actions 的 `ci.yml` 执行后端 lint/test、前端类型检查和构建、后端镜像构建。
- `publish-image.yml` 在 `main` 或 `v*` 标签更新时，将后端镜像推送到 GitHub Container Registry。

## 可验证边界

该仓库没有内置线上环境的自动部署、网页抓取器、VLM 图片摘要或公开的检索效果基准。若使用这些能力，应在新增实现与评测后再写入项目描述。
