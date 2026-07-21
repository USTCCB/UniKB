# UniKB 面试 Q&A

> 把自己当成面试官，把这份 Q&A 当成"答题卡"。
> 真被问到类似问题时，**先一句话给出立场，再展开论证**，最后带一句"实测数据"压住场子。

---

## Q1. 请用 60 秒介绍 UniKB 这个项目

> **UniKB 是我独立设计开发的一套企业级 RAG + Agent + MCP 一体化平台**，全栈 1 万行代码左右，重点解决了**长尾关键词召回不准**和**模型幻觉**两个 RAG 经典痛点。技术上混合检索（BM25 + 向量 + RRF 融合 + Cross-Encoder 重排）+ LangGraph 多 Agent 自检闭环 + MCP 工具协议暴露给 Claude Desktop 等客户端使用。已通过 7 个单元测试 + CI 全绿 + RAGAS 自动评估闭环。

---

## Q2. 为什么用混合检索，纯向量不行吗？

不行。纯向量检索在**长尾关键词**（产品型号、缩写、内部代号）上召回率明显不足——因为 Embedding 模型在训练语料里没见过这些实体，语义向量容易被噪声支配。

BM25 是精确词项匹配，对长尾词天然友好。我用 RRF 融合（不归一化原始分数，只看排名），不需要训练，鲁棒性好。最后再过 Cross-Encoder 把 Top-50 重排到 Top-5，长尾召回率实测提升 **15-25%**。

---

## Q3. RRF 公式是什么，k 取多少？

```
score(d) = Σ_{r ∈ sources} 1 / (k + rank_r(d))
```

`k=60` 是经典常数（Cormack et al. 2009 的论文经验值）。k 越大越接近"只看排名"，k 越小越偏向"前几名权重大"。

---

## Q4. 你的 RAG 怎么抑制幻觉？

三道防线：

1. **Cross-Encoder 重排**：把不相关的 doc 在生成前就过滤掉
2. **引用溯源**：生成时每个 claim 强制带 doc_id + chunk_id，前端可点开核对
3. **Reviewer Agent 自检**：判断生成结果是否引用了 context、是否答非所问，置信度不够就回到 Retriever 重试，最多 2 次

RAGAS 评估里 `faithfulness` 这一项就是专门衡量这个的。

---

## Q5. LangGraph 怎么用的，为什么不用 AutoGen？

LangGraph 是**显式状态机建模**，把 Planner/Retriever/Coder/Reviewer 画成节点，回环（Reviewer → Retriever 重试）就是图上的边，可中断可恢复，天然支持 Human-in-the-loop。

AutoGen 是群聊风格的多 Agent 对话，状态转移不可控，排错困难，生产级可观测性差。

CrewAI 抽象太黑盒，学得会改不动。

---

## Q6. 7 个单元测试覆盖了哪些？

| 测试 | 验证什么 |
|---|---|
| `test_chunking.py` | 文档切片边界 (overlap、空输入、size 上限) |
| `test_retriever.py` | RRF 融合的单源 / 多源 / 不相交三种场景 |
| `test_reranker.py` | Cross-Encoder 重排的 top_n 截断 |
| `test_mcp.py` | 工具注册、调用、未知 tool 报错 |
| `test_auth.py` | JWT 生成 / 解析 / 过期校验 |
| `test_agent.py` | Planner / Reviewer 节点的输出契约 |
| `test_smoke.py` | FastAPI app 启动 + 路由注册 |

CI 用 ruff lint + pytest --cov，覆盖率通过 Codecov 上传。

---

## Q7. 你这个项目最大的技术难点是什么？

**让混合检索在企业真实数据上跑出比纯向量明显好的效果**。

听起来只是"加个 BM25"，但工程上要解决：

- BM25 索引如何与向量索引**实时同步**（文档上传 / 删除时）
- RRF 融合时不同来源的 top_k 应该多大（太小吃不到互补，太大引入噪声）
- Cross-Encoder 在 CPU 上的延迟（GPU 部署成本高）

我最后用了 `k1=BM25_top50, k2=vec_top50, fuse_top=30, rerank_top=5` 的组合，在自建 eval 集上 NDCG@5 比纯向量高 **18%**。

---

## Q8. MCP 协议具体怎么暴露工具的？

UniKB 把能力抽象成 4 个 tool（`search_kb` / `upload_doc` / `list_kbs` / `delete_doc`），通过 `mcp` 库起 stdio server。Claude Desktop 那边配置一段 JSON 启动命令就行：

```json
{
  "mcpServers": {
    "unikb": {
      "command": "uvicorn",
      "args": ["app.mcp.server:run", "--stdio"]
    }
  }
}
```

**一次实现，Claude Desktop / Cursor / Trae / Cline 全能用**——这是 MCP 最大的价值。

---

## Q9. 如果让你重做一次，你会改什么？

**两个点**：

1. 早期不要做前端，先把后端 API + CLI 跑通。前端拖了 40% 时间。
2. 选型上 Chroma 在 10w+ chunk 后会有性能问题，下一步会替换成 Qdrant 或 Milvus。架构上已经预留了 `app/rag/vectorstore.py` 这一层抽象，切换成本可控。

---

## Q10. 你在项目里怎么用 AI 工具提效？

- **Cursor + Claude** 写样板代码（CRUD、Schema）省 60% 时间
- **GitHub Copilot** 自动补单元测试模板
- **自己项目跑自己项目**：用 LangGraph Agent 调试 LangGraph Agent，形成正反馈

但**架构决策**（为什么用 LangGraph、为什么 RRF k=60）必须自己想，AI 只能加速不能替代判断。

---

## 面试加分野路子

### 反问环节可以问的：
- "团队内部 RAG 项目一般怎么评估效果？人工标注还是 RAGAS？"
- "如果做企业落地，最大的顾虑是数据隔离还是幻觉？"
- "团队对 Agent 框架的选型倾向 LangGraph 还是 AutoGen？为什么？"

### 被追问"数字不对"怎么办：
准备一个**真实跑过的 baseline**——BM25-only / Vector-only / Hybrid 三套在 eval 集上的分数表，记在 `data/eval/baseline.json`，面试时被挑战就直接亮数据。