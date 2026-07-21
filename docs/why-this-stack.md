# 技术选型权衡

> 面试经常被问"为什么用 X 不用 Y"，这里统一整理答案。

## 1. LangGraph vs AutoGen vs CrewAI

| 框架 | 选 / 不选 | 理由 |
|---|---|---|
| **LangGraph** ✅ | 选 | 状态机显式建模、可中断恢复、Human-in-the-loop 天然支持；与 LangChain 生态无缝集成 |
| AutoGen | 不选 | 对话模式偏"群聊"，状态转移不可控，不适合生产级可观测 |
| CrewAI | 不选 | 上手最简单但抽象太黑盒，排查问题困难 |

**LangGraph 的关键优势**：`Reviewer → Retriever` 的回环是图结构，不是 if-else，天然适合"自检 → 重试"语义。

## 2. Chroma vs Milvus vs Qdrant vs Weaviate

| 向量库 | 选 / 不选 | 理由 |
|---|---|---|
| **Chroma** ✅ | 选 | 嵌入式运行、零配置、Python 原生 API；单机 10w chunk 内性能足够 |
| Milvus | 不选 | 强但运维重，需要单独集群；个人项目过度设计 |
| Qdrant | 备选 | Rust 性能强，HTTP API 干净，但 Docker 镜像大 |
| Weaviate | 不选 | 模块化丰富但学习曲线陡 |

**取舍逻辑**：UniKB 目标场景是企业知识库（单库 1-10w 文档），Chroma 完全够用；想升级到 Milvus 只需替换 `app/rag/vectorstore.py` 一个文件。

## 3. BGE-small-zh-v1.5 vs OpenAI text-embedding-3

| Embedding | 选 / 不选 | 理由 |
|---|---|---|
| **BGE-small-zh** ✅ | 选 | 中文 SOTA、模型小（93MB）、本地推理零成本、可换 GPU |
| text-embedding-3-large | 不选 | API 调用有成本、有网络依赖、数据出境问题 |

**默认 CPU 跑 BGE-small**，3 万文档首次建索引约 8 分钟；GPU 加速 20x。

## 4. Cross-Encoder 必上吗？

不上也能跑，但**长尾问题准确率会掉 15-25%**。

代价是延迟 +200ms（在 GPU 上）/ +2s（CPU）。UniKB 默认 CPU 上跑 `bge-reranker-base`，可在 `.env` 里切到 `bge-reranker-large` 提升效果。

## 5. 为什么是 SSE 而不是 WebSocket

| 协议 | 选 / 不选 | 理由 |
|---|---|---|
| **SSE** ✅ | 选 | 单向（服务端→客户端）正好对应"流式回答"；HTTP/1.1 兼容，Nginx/CDN 友好 |
| WebSocket | 不选 | 双向全双工——我们不需要客户端推送；握手开销大，断线重连复杂 |

## 6. MCP 真的有必要吗？

有。三条理由：

1. **一次实现多端用**：Claude Desktop / Cursor / Trae / Cline / Continue 全兼容
2. **官方协议稳定**：Anthropic 主导，工具描述走 JSON Schema，自带文档
3. **解耦客户端逻辑**：未来加企业 IM (飞书/钉钉) Bot 也只是换 transport

## 7. Next.js vs Vue vs Streamlit

| 前端 | 选 / 不选 | 理由 |
|---|---|---|
| **Next.js 14** ✅ | 选 | App Router + Server Component，SSE 流式渲染原生支持；TypeScript 生态 |
| Vue | 不选 | 个人偏好问题，团队 React 系多 |
| Streamlit | 不选 | 演示够用但不能产品化，无自定义交互 |

## 8. 为什么不用 Serverless / Lambda

- RAG 链路冷启动 5-10s（要加载 embedding 模型），用户体验崩
- 长连接 SSE 不适合 FaaS
- Chroma 嵌入式存储需要持久磁盘

UniKB 坚持传统 Docker Compose，部署到任意一台 2c4g 机器即可跑。

## 9. 一句话总结

> **每一步都选了"够用 + 可替换"的中间路线**——不追求最 fancy，但要每层都可独立替换/升级。
> 这也是 README 里"工程化"的真正含义：不是堆技术名词，是给未来的自己留出口。