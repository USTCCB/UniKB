"""RAG pipeline: 检索 + 重排 + 生成,作为统一的对外入口。

被 API 层、Agent 层和评估脚本共同使用。可选接入 LangFuse 做可观测。
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from app.agents.llm_router import get_llm
from app.core.config import settings
from app.core.observability import get_tracer
from app.rag.reranker import CrossEncoderReranker
from app.rag.retriever import HybridRetriever


_PROMPT_TEMPLATE = (
    "你是一个严谨的企业知识库助手。请严格基于【检索结果】回答用户问题，"
    "并在引用处使用 [1] [2] 等标注。如果检索结果中没有答案，请直接说「未找到相关信息」。\n\n"
    "【检索结果】\n{contexts}\n\n"
    "【用户问题】{question}\n\n"
    "【回答】"
)


def _format_contexts(docs: list[dict], top_k: int) -> tuple[list[str], list[str], list[float]]:
    contexts, ids, scores = [], [], []
    for i, c in enumerate(docs[:top_k]):
        content = (c.get("document") or "").strip()
        if not content:
            continue
        contexts.append(content)
        ids.append(c.get("id", ""))
        scores.append(float(c.get("rerank_score", c.get("rrf_score", 0.0))))
    return contexts, ids, scores


async def run_rag(
    question: str,
    kb_id: str = "default",
    top_k: int | None = None,
    use_agent: bool = False,
    user_id: str | None = None,
    session_id: str | None = None,
) -> dict:
    """主入口: 同步可调用,内部兼容 agent 模式。

    Returns:
        dict 形如 {answer, contexts, context_ids, context_scores, metadata}
        与 RAGAS 期待的字段对齐(question/answer/contexts/ground_truth)。
    """
    top_k = top_k or settings.top_k_final
    tracer = get_tracer()
    retriever = HybridRetriever(kb_id=kb_id)

    trace = tracer.trace(
        name="rag.run",
        metadata={"kb_id": kb_id, "mode": "agent" if use_agent else "rag", "user_id": user_id or "anon"},
        user_id=user_id or "anon",
        session_id=session_id,
    )
    with trace:
        if use_agent:
            from langchain_core.messages import HumanMessage

            from app.agents.graph import build_agent_graph

            with tracer.span(trace, "agent.invoke"):
                graph = build_agent_graph()
                result = graph.invoke(
                    {
                        "messages": [HumanMessage(content=question)],
                        "plan": "",
                        "retrieved": "",
                        "draft": "",
                        "final": "",
                        "trace": [],
                    }
                )
            answer = result.get("final", "") or ""
            retrieved = result.get("retrieved", "") or ""
            contexts = [retrieved] if retrieved else []
            return {
                "answer": answer,
                "contexts": contexts,
                "context_ids": [],
                "context_scores": [],
                "metadata": {"mode": "agent", "trace": result.get("trace", [])},
            }

        with tracer.span(trace, "retriever.hybrid"):
            candidates = retriever.retrieve(question, top_k=top_k * 2)
        with tracer.span(trace, "reranker.cross_encoder"):
            reranked = CrossEncoderReranker().rerank(question, candidates, top_k=top_k)
        contexts, ids, scores = _format_contexts(reranked, top_k)

        ctx_text = "\n\n".join(f"[{i + 1}] {c[:600]}" for i, c in enumerate(contexts))
        prompt = _PROMPT_TEMPLATE.format(contexts=ctx_text or "(无)", question=question)

        try:
            llm = get_llm()
            with tracer.span(trace, "llm.generate", model=settings.default_llm_model):
                resp = llm.invoke(prompt)
            answer = getattr(resp, "content", "") or str(resp)
            # 在 trace 上挂一次 generation(若启用)
            try:
                tracer.generation(
                    trace,
                    name="answer.generate",
                    model=settings.default_llm_model,
                    prompt=prompt,
                    completion=answer,
                )
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"LLM invocation failed during RAG: {e}; falling back to concatenated contexts.")
            answer = "\n\n".join(contexts) if contexts else "未找到相关信息"

    return {
        "answer": answer,
        "contexts": contexts,
        "context_ids": ids,
        "context_scores": scores,
        "metadata": {"mode": "rag", "kb_id": kb_id},
    }
