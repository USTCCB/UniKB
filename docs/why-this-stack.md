# 我们为什么选这套技术栈

这篇文档不是"某框架优缺点对比表"——那种表格谁都能拿 AI 生成。下面记录的是我们在做 UniKB 的过程中, 真实踩到的坑、推翻过的方案、以及最后留下的妥协。

## 1. LangGraph: 不是因为它火, 是因为Reviewer必须能回头找Retriever

最早我们用普通函数链实现 RAG: `retrieve → rerank → generate`。答案写出来就结束。但很快发现两类顽固错误:

- **引用不够**: LLM 生成了"根据文档...", 但检索结果里根本没有对应 chunk。
- **问题拆分错误**: 用户问"保修期和退换货政策", 检索只回了保修, 没回退换货, 模型就开始胡编。

我们尝试过在 prompt 里加"如果你不确定就拒绝回答"、"必须引用原文"之类的约束, 效果有限。LLM 还是会把检索片段里的词拼成一个看似合理的答案。

LangGraph 的价值在于把流程拆成显式节点, 并在 `Reviewer` 节点给出一个**可回环的判断**:

```
Planner → Retriever → Coder → Reviewer
              ↑_______________|
```

当 Reviewer 发现引用缺失或答案不完整时, 直接回到 Retriever 重新检索, 而不是让 LLM 硬编。这个回环用 LangGraph 的 `add_conditional_edges` 表达很自然; 用普通 if-else 写也能跑, 但状态字段会散得到处都是, 调试时很痛苦。

另一个真实需求是 **Human-in-the-loop**。Reviewer 判断"需要人工确认"时, 我们用 `interrupt_before={"coder"}` 把控制权交还用户。这个能力在生产审核场景里不是锦上添花, 是刚需。

**不选 AutoGen / CrewAI 的原因**: AutoGen 的群聊模型里, 谁能发言、发言顺序很难控制; CrewAI 的 Agent 抽象太黑盒, 出问题只能打日志猜。我们需要的是"状态机 + 可观测", 不是"多 Agent 对话"。

## 2. Chroma: 为了测试能跑, 我们甚至给它写了假实现

选 Chroma 一开始原因很简单: 单机零配置, `pip install chromadb` 就能跑, 10 万 chunk 以内性能足够。

但很快暴露出两个问题:

1. **CI 环境装 chromadb 很慢**, 还会拉一堆重依赖。我们早期 CI 经常卡在依赖安装上。
2. **本地跑单元测试时如果忘了起 Chroma, 测试直接报错**, 开发者体验很差。

我们本可以改 CI 用 Docker Compose 起 Chroma 服务, 但那会让单元测试变慢、变重。最后我们选择了**在测试层 fake 掉 Chroma**:

- `backend/tests/_fakes.py` 里实现了一个 `FakeVectorStore`, 用余弦相似度的纯 Python 版本替换 Chroma。
- `FakeBM25Store`  likewise 替换了 `rank_bm25`。
- 集成测试 `test_integration_retrieval.py` 用这些 fake 跑完整链路, 不依赖 torch / chromadb / sentence-transformers。

这套 fake 路径让 CI 能在 30 秒内跑完 71 个测试(63 单元 + 8 集成), 同时保证检索逻辑(add → retrieve → RRF → rerank → delete)被真实覆盖。

**Milvus / Qdrant 呢?** 性能确实更好, 但我们当前场景单库 1-10 万文档, Chroma 完全够用。如果将来要上十亿级, 需要换的也是 `app/rag/vector_store.py` 一个文件, 对外接口不变。

## 3. BGE-small-zh-v1.5: 93MB 模型是我们能接受的最大妥协

Embedding 模型我们试过两条路线:

- **OpenAI text-embedding-3-small**: 效果好, 但每个问题都要发请求, 有网络延迟、成本、数据出境三重顾虑。
- **BGE-small-zh-v1.5**: 93MB, CPU 可跑, 本地推理零成本。

对中文企业文档,BGE-small 的检索质量足够。真正让我们选择的是**冷启动时间**: 在 2c4g 的云主机上, BGE-small 首次加载约 3-5 秒; BGE-base 要 10 秒以上; text-embedding-3 则完全取决于网络。

但我们也付出了代价:

- **语义区分度不如大模型**: 对"保修"和"退换货"这种近义词, 偶尔会把售后政策 chunk 混到一起。
- **短查询召回弱**: "几天能退"这种短句, 向量检索容易跑偏, 必须靠 BM25 补回来。

所以最后方案是 **BM25 + 向量 + RRF 融合**, 而不是纯向量。BGE-small 不是最优解, 是"效果够用 + 成本可控"的折中。

## 4. Cross-Encoder 重排: 不上它, 长尾问题会掉 15-25%

 Hybrid 检索召回的 top-10 里, 常有"看起来相关、实际不相关"的 chunk。典型例子: 用户问"支持哪些支付方式", BM25 把"我们支持微信、支付宝"排在第一; 向量却把"企业月结协议模板"也召回了(因为都含"支付"相关语义)。

Cross-Encoder 用 query 和 doc 拼接做二分类/相关性打分, 能显著抑制这种 false positive。我们在小样本上测过, 加了 `bge-reranker-base` 后, 长尾问题的准确率从约 65% 提升到 80% 左右。这个数字因数据集而异, 但方向稳定。

代价是延迟:

- GPU 上: +200ms 左右(一次打分 10 对)。
- CPU 上: +1-3s(模型虽小, 但 10 次前向传播 still 累加)。

UniKB 默认开重排, 但允许在 `.env` 里关闭或换更小模型。对内部知识库问答来说, 用户更在乎答案准, 不太在意多等 1 秒。

## 5. SSE 而不是 WebSocket: 我们只需要服务器往客户端推流

流式回答只有一条路: 服务器 → 客户端。WebSocket 的双向全双工对我们来说是过度设计, 还带来:

- 握手和心跳开销
- 断线重连状态恢复复杂
- Nginx/CDN 配置更麻烦

SSE 是 HTTP 的一部分, 前端用 `EventSource` 即可, 后端 FastAPI 用 `StreamingResponse` 直接返回。 Claude / Cursor 等客户端对接也方便。

唯一的小坑是 SSE 只能按行推文本, 如果我们要传结构化事件(如 `{"type": "source", "payload": [...]}`), 需要自己封装 JSON 行。我们在 `backend/app/api/chat.py` 里做了统一封装, 前端按 event 类型解析。

## 6. MCP: 一条协议对接所有客户端, 但 transport  still 让人头疼

MCP 让我们可以只实现一次工具集(检索、上传、历史等), 然后 Claude Desktop、Cursor、Trae、Cline 都能调用。这是我们选它的核心理由。

但实际做下来发现, **transport 比协议本身更麻烦**:

- **stdio**: Claude Desktop 本地插件用, 最简单, 但只能本地跑。
- **SSE**: 远程服务用, 但我们得自己处理认证、超时、错误重试。
- 不同客户端对 MCP 工具描述的解析有细微差异, 同样的 JSON Schema 在 A 客户端能过, 在 B 客户端可能报错。

我们的结论是: MCP 协议本身是对的, 但距离"一次开发, 处处运行"还有距离。目前 UniKB 的 MCP 层被设计成可插拔 transport, 未来企业 IM Bot(飞书/钉钉) 也能复用同一套工具描述。

## 7. FastAPI + SQLAlchemy: 为了能从 SQLite 平滑切到 Postgres

我们起步用 SQLite, 因为本地开发和测试零配置。但企业部署不可能长期用 SQLite, 所以 ORM 必须从一开始就按 Postgres 写。

SQLAlchemy 2.x 的类型提示和异步支持足够现代, 配合 Alembic 可以做迁移。JWT 鉴权用 `python-jose` + `bcrypt`, 密码用 `passlib` 处理, 这些都是成熟到不需要再争论的选型。

FastAPI 的好处不是"快", 是**自动 OpenAPI 文档和类型校验**。前端团队看 `http://localhost:8000/docs` 就能对接, 省去了大量沟通成本。

## 8. RAGAS: 没有数字, 面试官问你"怎么保证检索质量"你会露怯

我们很早就集成了 RAGAS, 但一直没真正跑出报告。直到整理简历才发现,"集成了 RAGAS"和"跑过 RAGAS 并拿到 context_recall 95%"是完全不同的表述。

真实跑评估时遇到两个问题:

1. **CI 环境没 chromadb / torch, RAGAS 跑不起来**。
2. **RAGAS 默认调 OpenAI, 没有 API key 直接报错**。

我们的解决方案是走 **fake LLM + fake embedding + fake vector/bm25** 路径:

- `backend/tests/_fakes.py` 实现 FakeRagasJudge、FakeRagasEmbeddings。
- `backend/tests/run_ragas_eval.py` 用 fake 路径跑完整评估, 输出 `data/eval/ragas_report.json` 和 `data/eval/baseline.json`。
- fake 路径的分数**不代表真实模型效果**, 只保证评估链路本身通顺; 真实 baseline 需要换 `real_llm` 模式并配 API key。

这条 fake 路径让我们在沙箱里也能持续验证: 当数据集扩到 34 条、pipeline 改结构时, RAGAS 脚本不会突然挂掉。

## 9. 为什么不选 Serverless / Lambda

三个硬伤:

1. **Embedding 模型冷启动 5-10s**, 用户体验不可接受。
2. **SSE 长连接不适合 FaaS 计费模型**。
3. **Chroma 嵌入式存储需要持久磁盘**, Lambda 的临时存储会丢。

UniKB 用传统 Docker Compose 部署, 2c4g 机器即可跑。这不是因为我们保守, 是因为 RAG 链路本身就有状态(模型、向量库、文件), 强行无状态化会引入更多复杂度。

## 10. 一句话总结

> 我们选的每层技术, 都不是"最好的", 而是"当前阶段能承受、未来能替换的"。
>
> 真正难得的判断力, 不是看对比表做决定, 而是知道什么时候该 fake、什么时候该上真模型、什么时候该把测试和真实链路拆开。
