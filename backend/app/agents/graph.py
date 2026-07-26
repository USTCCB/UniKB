# -*- coding: utf-8 -*-
"""LangGraph 多 Agent 协作：Planner -> Retriever -> Coder -> Reviewer"""
from __future__ import annotations
import json
import re
from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.agents.llm_router import get_llm
from app.agents.tools import build_tools


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "对话消息"]
    plan: str
    retrieved: str
    draft: str
    final: str
    trace: List[dict]


PLANNER_SYS = "你是一个任务规划 Agent。给定用户问题，输出一份精简执行计划，格式：\n1. 目标\n2. 检索关键词\n3. 是否需要计算/查时间\n不要超过 80 字。\n"

CODER_SYS = "你是回答生成 Agent。基于【检索结果】直接回答用户问题。\n要求：\n- 用中文\n- 简洁、结构化\n- 必须引用检索结果中标注的 [1] [2] 等\n- 不确定就说「未找到相关信息」\n"

# Reviewer 现在要求严格 JSON 输出, 避免以前用 `if "通过" in out.content`
# 字符串匹配被误判 (例如 "审核通过率" 这种 token 也会触发).
REVIEWER_SYS = (
    "你是质量审查 Agent。检查【草稿】是否：\n"
    "1. 直接回答了用户问题\n"
    "2. 是否引用了检索结果\n"
    "3. 是否有明显事实错误\n"
    "请只输出一个 JSON 对象 (不要输出任何额外文字 / markdown 代码块):\n"
    '{"pass": true|false, "reason": "如果 pass=false 简要说明原因, 不超过 60 字"}\n'
    "如果合格, pass 必须是 true; 否则 false."
)


def _append_trace(state, role, content):
    trace = state.get("trace", [])
    trace.append({"role": role, "content": content})
    return trace


_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\"pass\"[^{}]*\"reason\"[^{}]*\}", re.DOTALL)


def _parse_reviewer_decision(raw: str) -> dict:
    """从 LLM 输出里挑出 JSON 决策.

    容错策略:
    1. 尝试整体 json.loads.
    2. 失败 -> 去掉 ```json ... ``` 代码块再 parse.
    3. 再失败 -> 正则抓第一个 {"pass": ...} 块.
    4. 全失败 -> fallback {"pass": True, "reason": "(parse failed, 默认通过)"},
       避免 LLM 输出格式问题导致整条链路死循环 (reviewer 不通过会触发重写).
    """
    s = (raw or "").strip()
    if not s:
        return {"pass": True, "reason": "(empty reviewer output)"}
    # 1) 直接
    try:
        d = json.loads(s)
        if isinstance(d, dict) and "pass" in d:
            return _normalize_decision(d)
    except Exception:
        pass
    # 2) 去掉代码块
    if "```" in s:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group(1))
                if isinstance(d, dict) and "pass" in d:
                    return _normalize_decision(d)
            except Exception:
                pass
    # 3) 正则抓 {"pass":..., "reason":...}
    m = _JSON_BLOCK_RE.search(s)
    if m:
        try:
            d = json.loads(m.group(0))
            return _normalize_decision(d)
        except Exception:
            pass
    # 4) fallback
    return {"pass": True, "reason": "(unparseable reviewer output, 默认通过)"}


def _normalize_decision(d: dict) -> dict:
    """把 LLM 输出 normalize 成统一 schema."""
    p = d.get("pass")
    # 兼容 true/false/"true"/"false"/1/0/"yes" 等
    if isinstance(p, str):
        pl = p.strip().lower()
        p = pl in ("true", "yes", "1", "pass", "通过")
    elif isinstance(p, int):
        p = bool(p)
    else:
        p = bool(p)
    reason = str(d.get("reason") or "").strip()
    if len(reason) > 200:
        reason = reason[:200]
    return {"pass": p, "reason": reason}


def planner_node(state):
    llm = get_llm()
    msgs = [SystemMessage(content=PLANNER_SYS), state["messages"][-1]]
    out = llm.invoke(msgs)
    state["plan"] = out.content
    state["trace"] = _append_trace(state, "planner", out.content)
    return state


def retriever_node(state):
    tools = build_tools()
    hybrid = next(t for t in tools if t.name == "hybrid_search")
    query_text = (state.get("plan") or "") + "\n" + state["messages"][-1].content
    res = hybrid.invoke({"query": query_text[:512], "top_k": 5})
    state["retrieved"] = res
    state["trace"] = _append_trace(state, "retriever", res[:500])
    return state


def coder_node(state):
    llm = get_llm()
    user_q = state["messages"][-1].content
    retrieved = state.get("retrieved", "")
    prompt = "用户问题：" + user_q + "\n\n检索结果：\n" + retrieved + "\n\n请基于检索结果给出最终回答。"
    msgs = [SystemMessage(content=CODER_SYS), HumanMessage(content=prompt)]
    out = llm.invoke(msgs)
    state["draft"] = out.content
    state["trace"] = _append_trace(state, "coder", out.content)
    return state


def reviewer_node(state):
    llm = get_llm()
    user_q = state["messages"][-1].content
    draft = state.get("draft", "")
    prompt = "用户问题：" + user_q + "\n\n草稿：\n" + draft + "\n\n请审查并按 JSON 格式输出决策."
    msgs = [SystemMessage(content=REVIEWER_SYS), HumanMessage(content=prompt)]
    out = llm.invoke(msgs)
    raw = out.content
    decision = _parse_reviewer_decision(raw)
    if decision["pass"]:
        state["final"] = draft
    else:
        fix = llm.invoke([
            SystemMessage(content=CODER_SYS),
            HumanMessage(content="请按审查意见修改：" + decision["reason"] + "\n\n原稿：" + draft),
        ])
        state["final"] = fix.content
    # trace 同时记录 raw + parsed, 方便后续审计 LLM 决策是否被 parse.
    state["trace"] = _append_trace(
        state, "reviewer",
        {"raw": raw, "decision": decision},
    )
    return state


def build_agent_graph():
    g = StateGraph(AgentState)
    g.add_node("planner", planner_node)
    g.add_node("retriever", retriever_node)
    g.add_node("coder", coder_node)
    g.add_node("reviewer", reviewer_node)
    g.set_entry_point("planner")
    g.add_edge("planner", "retriever")
    g.add_edge("retriever", "coder")
    g.add_edge("coder", "reviewer")
    g.add_edge("reviewer", END)
    return g.compile()
