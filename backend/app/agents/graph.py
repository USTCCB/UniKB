# -*- coding: utf-8 -*-
"""LangGraph 多 Agent 协作：Planner -> Retriever -> Coder -> Reviewer"""
from __future__ import annotations
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

REVIEWER_SYS = "你是质量审查 Agent。检查【草稿】是否：\n1. 直接回答了用户问题\n2. 是否引用了检索结果\n3. 是否有明显事实错误\n如果合格，输出「通过」；否则指出问题并要求重写。"


def _append_trace(state, role, content):
    trace = state.get("trace", [])
    trace.append({"role": role, "content": content})
    return trace


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
    prompt = "用户问题：" + user_q + "\n\n草稿：\n" + draft + "\n\n请审查。"
    msgs = [SystemMessage(content=REVIEWER_SYS), HumanMessage(content=prompt)]
    out = llm.invoke(msgs)
    if "通过" in out.content:
        state["final"] = draft
    else:
        fix = llm.invoke([
            SystemMessage(content=CODER_SYS),
            HumanMessage(content="请按审查意见修改：" + out.content + "\n\n原稿：" + draft),
        ])
        state["final"] = fix.content
    state["trace"] = _append_trace(state, "reviewer", out.content)
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
