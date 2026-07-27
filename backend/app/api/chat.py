# -*- coding: utf-8 -*-
from __future__ import annotations
import asyncio
import json
import queue
import threading
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
from app.rag.reranker import get_reranker
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
        get_reranker().rerank, req.question, candidates, req.top_k,
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
                # ----- agent 分支: 严格 offload + 按节点推 trace (真实"流") -----
                # graph.invoke 是同步阻塞 (langgraph 同步实现), 必须卸到 threadpool,
                # 否则会卡住 FastAPI 事件循环, 这是 P1 offload fix 的最低要求.
                # 进一步: 用 graph.stream() 在 threadpool 内部每跑完一个节点, 就通过
                # queue 把这个节点的 trace 喂给事件循环, 推到前端, 而不是等整图跑完
                # 才吐. 这样 SSE 是真正的"分步反馈", 而不是延迟一致性.
                from langchain_core.messages import HumanMessage

                graph = build_agent_graph()
                state_in = {
                    "messages": [HumanMessage(content=req.question)],
                    # 关键: 把 req.kb_id 显式塞进 state, 让 retriever_node 走正确 kb.
                    "kb_id": req.kb_id,
                    "plan": "",
                    "retrieved": "",
                    "draft": "",
                    "final": "",
                    "trace": [],
                    "retry_count": 0,
                    "last_reviewer_reason": "",
                }

                q: "queue.Queue[tuple[str, object]]" = queue.Queue()

                def _runner():
                    try:
                        # graph.stream() 每跑完一个节点就 yield {node_name: state_after_step}.
                        last_trace_len = 0
                        last_final = ""
                        for step in graph.stream(state_in):
                            for _node_name, ns in step.items():
                                if not isinstance(ns, dict):
                                    continue
                                curr_trace = ns.get("trace", []) or []
                                new_entries = curr_trace[last_trace_len:]
                                last_trace_len = len(curr_trace)
                                last_final = ns.get("final", last_final) or last_final
                                if new_entries:
                                    q.put(("trace", new_entries))
                        # 整图跑完, 把 final 一起 push 到 queue, 客户端从 token 事件里拿.
                        q.put(("done", last_final))
                    except Exception as e:  # pragma: no cover
                        q.put(("error", e))

                t = threading.Thread(target=_runner, daemon=True)
                t.start()

                # 在事件循环里 round-robin 拉取 trace 事件, 真正流式.
                final_text = ""
                while True:
                    kind, payload = await asyncio.to_thread(q.get)
                    if kind == "trace":
                        for step in payload:
                            yield "data: " + json.dumps(
                                {"type": "trace", "data": step}, ensure_ascii=False
                            ) + "\n\n"
                    elif kind == "error":
                        err = payload
                        logger.exception("agent stream error: " + str(err))
                        yield "data: " + json.dumps(
                            {"type": "error", "data": {"msg": str(err)}}, ensure_ascii=False
                        ) + "\n\n"
                        return
                    elif kind == "done":
                        final_text = payload or ""
                        break

                # 用一个 token 事件收尾 (跟 RAG 分支 SSE 协议一致).
                if final_text:
                    yield "data: " + json.dumps(
                        {"type": "token", "data": {"text": final_text}}, ensure_ascii=False
                    ) + "\n\n"
                yield "data: {\"type\":\"done\"}\n\n"
                return

            candidates = await retriever.retrieve_async(req.question, top_k=req.top_k * 2)
            reranked = await asyncio.to_thread(
                get_reranker().rerank, req.question, candidates, req.top_k,
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
        # 关键: 把 req.kb_id 显式塞进 state, 让 retriever_node 走正确 kb.
        "kb_id": req.kb_id,
        "plan": "",
        "retrieved": "",
        "draft": "",
        "final": "",
        "trace": [],
        "retry_count": 0,
        "last_reviewer_reason": "",
    }
    # graph.invoke 是同步阻塞, 卸到 threadpool 不阻塞 FastAPI 事件循环.
    result = await asyncio.to_thread(graph.invoke, state_in)
    sid = req.session_id or ("sess_" + uuid.uuid4().hex[:8])
    return ChatResponse(
        answer=result.get("final", ""),
        session_id=sid,
        sources=[],
        agent_trace=result.get("trace", []),
        usage={},
    )
