"""测试 Agent 模式真的会用 state['kb_id'] 而不是默认 "default".

P0-2 bug 的根因是: chat.py 调 `build_agent_graph()` 时, `AgentState` 里
没有传 `kb_id`; `retriever_node` 内部调 `build_tools()` (无参数), 默认值
是 "default". 结果是无论用户选哪个私有 kb, agent 模式永远只检索公共库
"default", 架空了 kb_registry 的 ACL.

本测试覆盖:
1. retriever_node 把 `state["kb_id"]` 透传给 `build_tools(kb_id=...)`.
2. 端到端 build_agent_graph().invoke(...): retriever_node 拿到的 kb_id
   必须等于 state["kb_id"], 而不是 "default".
"""
from __future__ import annotations

import json
from typing import List

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from app.agents import graph as graph_mod
from app.agents.graph import AgentState, build_agent_graph


def _planner_returns_needs_retrieval(keywords: str = ""):
    """构造一个 mock LLM, 跑 planner 时返回需要 retrieval 的决策."""

    class _LLM:
        def __init__(self, content):
            self._content = content

        def invoke(self, _msgs):
            return AIMessage(content=self._content)

    payload = json.dumps({
        "needs_calculator": False,
        "needs_retrieval": True,
        "needs_date": False,
        "keywords": keywords,
    })
    return _LLM(payload)


def _make_stub_toolbox(kb_id_captured: dict):
    """构造一个 build_tools 桩, 记下被传入的 kb_id, 并返回带 hybrid_search 的 list."""

    def _fake_build_tools(kb_id: str = "default"):
        kb_id_captured["kb_id"] = kb_id

        @tool
        def hybrid_search(query: str, top_k: int = 5) -> str:
            "stub hybrid"
            return f"hybrid({query}) from kb={kb_id}"

        return [hybrid_search]

    return _fake_build_tools


def test_retriever_node_propagates_kb_id_via_graph():
    """端到端: build_agent_graph().invoke({"kb_id": "alice_kb", ...}) 后, retriever_node
    拿到的 kb_id 应该是 "alice_kb" 而不是 "default"."""
    captured: dict = {}
    fake_tools = _make_stub_toolbox(captured)
    graph_mod.get_llm = lambda: _planner_returns_needs_retrieval("hello")
    graph_mod.build_tools = fake_tools

    g = build_agent_graph()
    out = g.invoke({
        "messages": [HumanMessage(content="hello")],
        "kb_id": "alice_kb",
        "plan": "",
        "retrieved": "",
        "draft": "",
        "final": "",
        "trace": [],
        "retry_count": 0,
        "last_reviewer_reason": "",
    })

    assert captured.get("kb_id") == "alice_kb", (
        f"agent 模式没把 state['kb_id'] 透传给 retriever 工具, 实际 {captured.get('kb_id')!r}"
    )
    # 顺便验证: trace 里 retriever 节点的输出包含了 "kb=alice_kb" 而不是 "kb=default".
    retriever_traces = [t for t in out["trace"] if t["role"] == "retriever"]
    assert retriever_traces, "retriever 节点没写 trace"
    snippet = retriever_traces[0]["content"]
    assert "kb=alice_kb" in snippet, (
        f"retriever 工具拿到的 kb_id 错了, 实际 snippet={snippet!r}"
    )


def test_retriever_node_distinct_kb_id_per_request():
    """同一个 graph 跑两次, 第二次用不同的 kb_id, retriever 拿到的 kb_id 必须跟
    第二次请求的 state 走, 而不是缓存了第一次的."""
    captured_runs: List[str] = []
    call_count = {"n": 0}

    def _fake_build_tools(kb_id: str = "default"):
        captured_runs.append(kb_id)

        @tool
        def hybrid_search(query: str, top_k: int = 5) -> str:
            "stub"
            return "ctx"

        return [hybrid_search]

    graph_mod.get_llm = lambda: _planner_returns_needs_retrieval("x")
    graph_mod.build_tools = _fake_build_tools

    g = build_agent_graph()
    g.invoke({
        "messages": [HumanMessage(content="q1")],
        "kb_id": "kb_one",
        "plan": "", "retrieved": "", "draft": "", "final": "", "trace": [],
        "retry_count": 0, "last_reviewer_reason": "",
    })
    g.invoke({
        "messages": [HumanMessage(content="q2")],
        "kb_id": "kb_two",
        "plan": "", "retrieved": "", "draft": "", "final": "", "trace": [],
        "retry_count": 0, "last_reviewer_reason": "",
    })

    # 注意 build_tools 在 retriever_node 里被调用, 也可能在 graph 编译时;
    # 我们只关心有 "kb_one" 和 "kb_two" 出现过, 且没出现 "default".
    assert "kb_one" in captured_runs
    assert "kb_two" in captured_runs
    assert "default" not in captured_runs, (
        f"agent 模式不应该默认走 default, 实际 captured={captured_runs!r}"
    )


def test_retriever_node_with_empty_kb_id_falls_back_to_default():
    """state 里没传 kb_id 时 (旧调用点, 比如测试代码), 兜底走 'default',
    而不是 crash. 这是向后兼容保证."""
    captured: dict = {}

    def _fake_build_tools(kb_id: str = "default"):
        captured["kb_id"] = kb_id

        @tool
        def hybrid_search(query: str, top_k: int = 5) -> str:
            "stub"
            return "ctx"

        return [hybrid_search]

    graph_mod.get_llm = lambda: _planner_returns_needs_retrieval("x")
    graph_mod.build_tools = _fake_build_tools

    state: AgentState = {
        "messages": [HumanMessage(content="q")],
        "kb_id": "",  # 故意空
        "plan": "", "retrieved": "", "draft": "", "final": "", "trace": [],
        "retry_count": 0, "last_reviewer_reason": "",
    }
    # 不通过 graph (graph 编译时不做 retriever), 直接调 retriever_node.
    graph_mod.retriever_node(state)
    assert captured["kb_id"] == "default", (
        f"kb_id 空时应当兜底为 'default', 实际 {captured['kb_id']!r}"
    )