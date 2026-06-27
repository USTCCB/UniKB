"""LangGraph 多 Agent 协作：
Planner -> Retriever -> Coder -> Reviewer
"""
from __future__ import annotations

from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.agents.llm_router import get_llm
from app.agents.tools import build_tools


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "对话消息"]
    plan: str
    retrieved: str
    draft: str
    final: str
    trace: List[dict]


PLANNER_SYS = """你是一个任务规划 Agent。给定用户问题，输出一份精简执行计划，格式：
1. 目标
2. 检索关键词
3. 是否需要计算/查时间
不要超过 80 字。
"""

CODER_SYS = """你是回答生成 Agent。基于【检索结果】直接回答用户问题。
要求：
- 用中文
- 简洁、结构化
- 必须引用检索结果中标注的 [1] [2] 等
- 不确定就说"未找到相关信息"
"""

REVIEWER_SYS = """你是质量审查 Agent。检查【草稿】是否：
1. 直接回答了用户问题
2. 是否引用了检索结果
3. 是否有明显事实错误
如果合格，输出"通过"；否则指出问题并要求重写。"""


def _append_trace(state: AgentState, role: str, content: str):
    trace = state.get("trace", [])
    trace.append({"role": role, "content": content})
    return trace


def planner_node(state: AgentState) -> AgentState:
    llm = get_llm()
    msgs = [SystemMessage(content=PLANNER_SYS), state["messages"][-1]]
    out = llm.invoke(msgs)
    state["plan"] = out.content
    state["trace"] = _append_trace(state, "planner", out.content)
    return state


def retriever_node(state: AgentState) -> AgentState:
    # 使用 hybrid_search tool 强制调一次
    from app.agents.tools import build_tools
    tools = build_tools()
    hybrid = next(t for t in tools if t.name == "hybrid_search")
    # 用 plan 中关键词 + 原问题一起查
    query_text = (state.get("plan") or "") + "\n" + state["messages"][-1].content
    res = hybrid.invoke({"query": query_text[:512], "top_k": 5})
    state["retrieved"] = res
    state["trace"] = _append_trace(state, "retriever", res[:500])
    return state


def coder_node(state: AgentState) -> AgentState:
    llm = get_llm()
    user_q = state["messages"][-1].content
    prompt = f"用户问题：{user_q}\n\n检索结果：\n{state.get(