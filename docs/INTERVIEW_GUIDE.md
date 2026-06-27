# UniKB 面试讲解稿（陈彪专用）

> 在面试时被问"你这个项目做了什么、为什么这么做"时，照着这个文档讲即可。

## 1. 一句话定位（15 秒内讲完）

> UniKB 是一个面向企业知识库场景的 RAG 问答平台。我从 0 到 1 设计并实现了
> 多 Agent 协作 + MCP 协议 + 混合检索 + 重排序 + 引用溯源 的端到端系统，
> 覆盖后端 FastAPI、前端 Next.js、Docker 部署、CI/CD 与 RAGAS 评估全链路。

## 2. 架构（30 秒）

四层：
1. 接入层：Next.js 前端 + FastAPI 后端 + JWT 鉴权
2. Agent 层：LangGraph 编排 Planner -> Retriever -> Coder -> Reviewer
3. RAG 层：BM25 + 向量 + RRF 融合 + Cross-Encoder 重排
4. 存储层：Chroma / BM25（默认）/ PostgreSQL / Redis

## 3. 技术亮点（按权重讲，5–10 分钟）

### 3.1 混合检索 + RRF 融合
- 单一向量检索在中文长尾词上召回低
- 引入 BM25 做关键词兜底，用 RRF (Reciprocal Rank Fusion, k=60) 融合两路
- 公式：score(d) = sum( 1 / (k + rank_i(d)) )，无需调权重，鲁棒性强

### 3.2 Cross-Encoder 重排
- 双塔向量召回快但粗排
- 用 BGE-reranker-base 做精排，按 query-doc 联合打分
- 抑制幻觉效果显著（经验 20%–40%）

### 3.3 多 Agent 协作
- Planner：把问题拆解成可执行计划 + 检索关键词
- Retriever：执行 hybrid_search 工具
- Coder：基于检索结果生成带引用的回答
- Reviewer：质量审查 + 必要时让 Coder 改写
- 用 LangGraph 状态机实现，支持断点恢复

### 3.4 MCP 协议
- Model Context Protocol，2024 年底由 Anthropic 推出，已成为业界标准
- 把 hybrid_search / calculator / current_date 三个工具通过 stdio 暴露
- Claude Desktop / Cursor / Trae 等客户端可以直接接入
- 体现"懂前沿协议 + 能落地"

### 3.5 工程化
- FastAPI 异步 + SSE 流式输出
- 多 LLM 路由（DeepSeek / Qwen / OpenAI 配置切换）
- Docker Compose 一键起 backend / frontend / redis
- GitHub Actions CI：lint + smoke import + docker build
- 评估：RAGAS 4 大指标（faithfulness / answer_relevancy / context_precision / context_recall）

## 4. 高频追问 & 参考答案

### Q1：为什么选 Chroma 而不是 Milvus / Faiss？
A：Chroma 零配置、嵌入式、易上手，适合个人项目和中小规模（< 100w 向量）。
   如果切到生产 / 多租户 / 亿级向量，我会换 Milvus 或 Qdrant，并通过
   `app.rag.vector_store` 这一层抽象无缝切换。

### Q2：RRF 和加权融合有什么区别？
A：RRF 是 rank-based fusion，不依赖各路分数的绝对值分布，对调参不敏感；
   加权融合（score = alpha * bm25 + (1-alpha) * vec）需要归一化，且对 alpha 敏感。
   经验上 RRF 更鲁棒，所以是工业界主流。

### Q3：为什么需要 Coder 和 Reviewer 两个 Agent？
A：Coder 负责"生成"，Reviewer 负责"质检"。
   单一 Coder 模型容易自洽自信、产生幻觉；引入 Reviewer 后能形成 self-refine 闭环。
   代价是 latency + 1 次 LLM 调用，收益是事实性显著提升。

### Q4：MCP 和 Function Calling 有什么区别？
A：Function Calling 是 LLM 调用工具的协议（各家实现不同）；
   MCP 是 Anthropic 提出的统一标准，让 LLM Client（Claude/Cursor）和 Server（任何工具集）解耦。
   我的项目里同时支持：内部 LangChain tool 调用 + 外部 MCP server 暴露。

### Q5：怎么评估 RAG 效果？
A：用 RAGAS 在一个标注集（> 50 条 QA）上跑：
- faithfulness：答案与检索结果的一致性（抑制幻觉）
- answer_relevancy：答案与问题的相关性
- context_precision：检索 top_k 中相关 chunk 的占比
- context_recall：ground_truth 涉及的信息是否被召回

## 5. 这个项目能给你简历加的关键词

- LLM 应用 / RAG / Agent / LangGraph
- Hybrid Search (BM25 + 向量 + RRF)
- Cross-Encoder / Rerank
- MCP / Model Context Protocol
- Function Calling / Tool Use
- FastAPI / SSE 流式
- DeepSeek / Qwen / OpenAI 多 LLM 路由
- Embedding / 向量检索
- Docker / CI/CD
- RAGAS 评估
- JWT 鉴权 / API Key

## 6. 自我介绍（90 秒版）

> 我是陈彪，安徽农业大学智能科学与技术专业，大三。
> 主攻大模型应用开发，独立完成了 UniKB 这个企业级 RAG 知识库项目。
> 核心做了四件事：
> 一是把 BM25 和向量检索用 RRF 融合，加上 Cross-Encoder 重排，把事实准确率拉高；
> 二是用 LangGraph 编排了 Planner、Retriever、Coder、Reviewer 四个 Agent 做自检；
> 三是把内部工具通过 MCP 协议暴露，能让 Claude/Cursor 直接接入；
> 四是用 FastAPI + Next.js + Docker + CI/CD 跑通了工程化全链路，并用 RAGAS 做了量化评估。
> 我对 LLM 应用落地很感兴趣，希望能加入贵团队做更多有意思的 AI 应用。
