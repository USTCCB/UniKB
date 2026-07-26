# -*- coding: utf-8 -*-
from __future__ import annotations
import asyncio
import json
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from app.agents.graph import build_agent_graph
from app.agents.llm_router import get_llm
from app.api.deps import get_current_user
from app.core.kb_registry import ensure_kb_for_read
from app.core.rate_limit import check_chat_quota
from app.rag.retriever import get_retriever
from app.rag.reranker import CrossEncoderReranker
from app.schemas.chat import ChatRequest, ChatResponse, SourceItem


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    if request.client:
        return request.client.host or "unknown"
    return "unknown"


def _check_chat_quota_or_429(user: str, request: Request):
    ip = _client_ip(request)
    decision = check_chat_quota(user, ip)
    if not decision.allowed:
        from fastapi import HTTPException, status

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=decision.reason,
            headers={"Retry-After": str(int(decision.retry_after) + 1)},
        )

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


def _build_prompt(question, contexts):
    parts = []
    for i, c in enumerate(contexts):
        doc = c.get("document", "") or ""
        parts.append("[" + str(i + 1) + "] " + doc[:600])
    ctx_text = "\n\n".join(parts)
    return (
        "你是一个严谨的企业知识库助手。请严格基于检索结果回答用户问题，"
        "并在回答末尾用 [1] [2] 等标注引用来源。如果检索结果中没有答案，请直接说「未找到相关信息」。\n\n"
        + "【检索结果】\n" + ctx_text + "\n\n"
        + "【用户问题】" + question + "\n\n"
        + "【回答】"
    )


@router.post("", response_model=ChatResponse, summary="一次性问答")
async def chat(req: ChatRequest, request: Request, user: str = Depends(get_current_user)):
    _check_chat_quota_or_429(user, request)
    # ACL: 公共 kb 直接放行; 私有 kb 必须 owner == 当前用户 (403)
    ensure_kb_for_read(req.kb_id, user)
    retriever = get_retriever(req.kb_id)
    if req.mode == "agent":
        return await _chat_agent(req)

    # CPU 密集: 把 retriever.retrieve() 和 reranker.rerank() 卸到 threadpool,
    # 让 FastAPI 事件循环在其他请求处理时不会被这两个同步调用阻塞.
    candidates = await retriever.retrieve_async(req.question, top_k=req.top_k * 2)
    reranked = await asyncio.to_thread(
        CrossEncoderReranker().rerank, req.question, candidates, req.top_k,
    )
    prompt = _build_prompt(req.question, reranked)
    # LLM 调用也是阻塞型同步 SDK, 也卸到 threadpool.
    resp = await asyncio.to_thread(get_llm().invoke, prompt)
    sources = []
    for c in reranked:
        meta = c.get("metadata") or {}
        doc_id = meta.get("doc_id", "")
        chunk_id = c.get("id", "")
        content = (c.get("document") or "")[:300]
        score = float(c.get("rerank_score", c.get("rrf_score", 0)))
        sources.append(SourceItem(
            doc_id=doc_id, chunk_id=chunk_id, content=content, score=score, metadata=meta,
        ))
    sid = req.session_id or ("sess_" + uuid.uuid4().hex[:8])
    return ChatResponse(answer=resp.content, session_id=sid, sources=sources, usage={})


@router.post("/stream", summary="SSE 流式问答")
async def chat_stream(req: ChatRequest, request: Request, user: str = Depends(get_current_user)):
    _check_chat_quota_or_429(user, request)
    # ACL: 进入流式生成前先校验, 避免生成到一半才抛 403
    ensure_kb_for_read(req.kb_id, user)
    async def event_gen():
        try:
            retriever = get_retriever(req.kb_id)
            if req.mode == "agent":
                graph = build_agent_graph()
                from langchain_core.messages import HumanMessage
                state_in = {
                    "messages": [HumanMessage(content=req.question)],
                    "plan": "", "retrieved": "", "draft": "", "final": "", "trace": [],
                }
                result = graph.invoke(state_in)
                for step in result.get("trace", []):
                    yield "data: " + json.dumps({"type": "trace", "data": step}, ensure_ascii=False) + "\n\n"
                final_text = result.get("final", "")
                yield "data: " + json.dumps({"type": "token", "data": {"text": final_text}}, ensure_ascii=False) + "\n\n"
                yield "data: {\"type\":\"done\"}\n\n"
                return

            candidates = await retriever.retrieve_async(req.question, top_k=req.top_k * 2)
            reranked = await asyncio.to_thread(
                CrossEncoderReranker().rerank, req.question, candidates, req.top_k,
            )
            for c in reranked:
                payload = {
                    "type": "source",
                    "data": {
                        "chunk_id": c.get("id", ""),
                        "content": (c.get("document") or "")[:200],
                        "score": c.get("rerank_score", c.get("rrf_score", 0)),
                    },
                }
                yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
            prompt = _build_prompt(req.question, reranked)
            # LLM 流式调用也是阻塞型同步 SDK, 用线程池迭代.
            # run_in_executor 返回迭代器本身是阻塞的, 我们用一个 async 包装,
            # 让每轮 token 在 threadpool 里生成, 通过 queue 喂给 asyncio.
            from app.core.streaming import sync_iter_to_async
            async for token in sync_iter_to_async(get_llm().stream(prompt)):
                if token:
                    yield "data: " + json.dumps({"type": "token", "data": {"text": token}}, ensure_ascii=False) + "\n\n"
            yield "data: {\"type\":\"done\"}\n\n"
        except Exception as e:
            logger.exception("stream error: " + str(e))
            yield "data: " + json.dumps({"type": "error", "data": {"msg": str(e)}}, ensure_ascii=False) + "\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


async def _chat_agent(req: ChatRequest) -> ChatResponse:
    from langchain_core.messages import HumanMessage
    graph = build_agent_graph()
    state_in = {
        "messages": [HumanMessage(content=req.question)],
        "plan": "", "retrieved": "", "draft": "", "final": "", "trace": [],
    }
    result = graph.invoke(state_in)
    sid = req.session_id or ("sess_" + uuid.uuid4().hex[:8])
    return ChatResponse(
        answer=result.get("final", ""),
        session_id=sid,
        sources=[],
        agent_trace=result.get("trace", []),
        usage={},
    )
