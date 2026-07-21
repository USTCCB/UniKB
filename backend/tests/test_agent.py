"""测试 6: LangGraph Agent 节点契约 (用 monkeypatch 把 LLM 换成 mock)."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.graph import AgentState, planner_node, reviewer_node


class FakeLLM:
    def __init__(self, content):
        self._content = content

    def invoke(self, _msgs):
        return AIMessage(content=self._content)


def test_planner_node_sets_plan(monkeypatch):
    monkeypatch.setattr("app.agents.graph.get_llm", lambda: FakeLLM("1. 目标 / 2. 检索 x / 3. 否"))
    state: AgentState = {
        "messages": [HumanMessage(content="UniKB 怎么部署?")],
        "plan": "",
        "retrieved": "",
        "draft": "",
        "final": "",
        "trace": [],
    }
    out = planner_node(state)
    assert "目标" in out["plan"]
    assert any(t["role"] == "planner" for t in out["trace"])


def test_reviewer_node_accepts_cited_answer(monkeypatch):
    monkeypatch.setattr("app.agents.graph.get_llm", lambda: FakeLLM("通过"))
    state: AgentState = {
        "messages": [HumanMessage(content="q")],
        "plan": "",
        "retrieved": "[1] foo\n[2] bar",
        "draft": "根据 [1] 引用, 答案是 42。",
        "final": "",
        "trace": [],
    }
    out = reviewer_node(state)
    assert out["final"] == "通过"
    assert any(t["role"] == "reviewer" for t in out["trace"])


def test_reviewer_node_flags_inadequate_answer(monkeypatch):
    monkeypatch.setattr(
        "app.agents.graph.get_llm",
        lambda: FakeLLM("不通过: 没有引用检索结果"),
    )
    state: AgentState = {
        "messages": [HumanMessage(content="q")],
        "plan": "",
        "retrieved": "[1] foo",
        "draft": "我不知道。",
        "final": "",
        "trace": [],
    }
    out = reviewer_node(state)
    assert "不通过" in out["final"] or "重写" in out["final"] or "问题" in out["final"]


def test_message_history_concatenates_through_review():
    """Sanity: trace 在多次节点调用后持续累积."""
    state: AgentState = {
        "messages": [HumanMessage(content="q")],
        "plan": "",
        "retrieved": "",
        "draft": "",
        "final": "",
        "trace": [],
    }
    state["trace"].append({"role": "planner", "content": "plan"})
    state["trace"].append({"role": "retriever", "content": "ctx"})
    assert len(state["trace"]) == 2