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
    """reviewer_node 在 LLM 返回 '通过' 时, state.final 等于 draft."""
    calls = {"n": 0}

    class FakeLLM:
        def invoke(self, _msgs):
            calls["n"] += 1
            return AIMessage(content="通过" if calls["n"] == 1 else "")

    monkeypatch.setattr("app.agents.graph.get_llm", FakeLLM)
    state: AgentState = {
        "messages": [HumanMessage(content="q")],
        "plan": "",
        "retrieved": "[1] foo\n[2] bar",
        "draft": "根据 [1] 引用, 答案是 42。",
        "final": "",
        "trace": [],
    }
    out = reviewer_node(state)
    assert out["final"] == "根据 [1] 引用, 答案是 42。"  # draft 被保留
    assert any(t["role"] == "reviewer" for t in out["trace"])


def test_reviewer_node_triggers_rewrite_on_reject(monkeypatch):
    """LLM 返回非通过时, 第二次 invoke 会被调用来重写, state.final 是重写结果."""
    state: AgentState = {
        "messages": [HumanMessage(content="q")],
        "plan": "",
        "retrieved": "[1] foo",
        "draft": "我不知道。",
        "final": "",
        "trace": [],
    }

    # 把调用计数放在一个共享可变 dict 里, 让两个独立的 FakeLLM 实例共享状态
    calls = {"n": 0}

    class FakeLLM:
        def invoke(self, _msgs):
            calls["n"] += 1
            if calls["n"] == 1:
                return AIMessage(content="不通过: 没有引用检索结果")
            return AIMessage(content="改写后答案: 根据 [1], 答案是 X。")

    monkeypatch.setattr("app.agents.graph.get_llm", FakeLLM)
    out = reviewer_node(state)
    assert "改写" in out["final"]
    assert out["final"] != "我不知道。"


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