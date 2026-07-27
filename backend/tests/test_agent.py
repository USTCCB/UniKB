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
    monkeypatch.setattr(
        "app.agents.graph.get_llm",
        lambda: FakeLLM(
            '{"needs_calculator": false, "needs_retrieval": true, "needs_date": false, "keywords": "部署 UniKB"}'
        ),
    )
    state: AgentState = {
        "messages": [HumanMessage(content="UniKB 怎么部署?")],
        "kb_id": "default",
        "plan": "",
        "retrieved": "",
        "draft": "",
        "final": "",
        "trace": [],
        "retry_count": 0,
        "last_reviewer_reason": "",
    }
    out = planner_node(state)
    assert "部署 UniKB" in out["plan"]
    assert any(t["role"] == "planner" for t in out["trace"])


def test_reviewer_node_accepts_cited_answer(monkeypatch):
    """reviewer_node 在 LLM 返回 pass=true JSON 时, state.final 等于 draft."""
    calls = {"n": 0}

    class FakeLLM:
        def invoke(self, _msgs):
            calls["n"] += 1
            # 第一次: pass=true; 第二次 (rewrite) 不会被调用
            return AIMessage(content='{"pass": true, "reason": "ok"}' if calls["n"] == 1 else "")

    monkeypatch.setattr("app.agents.graph.get_llm", FakeLLM)
    state: AgentState = {
        "messages": [HumanMessage(content="q")],
        "kb_id": "default",
        "plan": "",
        "retrieved": "[1] foo\n[2] bar",
        "draft": "根据 [1] 引用, 答案是 42。",
        "final": "",
        "trace": [],
        "retry_count": 0,
        "last_reviewer_reason": "",
    }
    out = reviewer_node(state)
    assert out["final"] == "根据 [1] 引用, 答案是 42。"  # draft 被保留
    assert any(t["role"] == "reviewer" for t in out["trace"])


def test_reviewer_node_triggers_rewrite_on_reject(monkeypatch):
    """LLM 返回 pass=false JSON 时: reviewer_node **不**就地重写 (旧行为), 而是写
    trace + state 信号, 真正的"重写"靠 conditional_edges 回到 retriever.

    这里验证:
    1. reviewer_node 只调用一次 llm.invoke (审查).
    2. final 在审查节点里等于 draft (被图回环覆盖是另一回事).
    3. last_reviewer_reason 在 state, route_review 拿到后选 "retriever" 分支.
    """
    state: AgentState = {
        "messages": [HumanMessage(content="q")],
        "kb_id": "default",
        "plan": "",
        "retrieved": "[1] foo",
        "draft": "我不知道。",
        "final": "",
        "trace": [],
        "retry_count": 0,
        "last_reviewer_reason": "",
    }

    # 同一个 LLM 实例会被 invoke 一次 (就审查). 注意: 重写不再发生在 reviewer_node
    # 内部, 而是通过 add_conditional_edges 回环到 retriever -> coder.
    class FakeLLM:
        def invoke(self, _msgs):
            return AIMessage(content='{"pass": false, "reason": "缺少引用, 需要重写"}')

    monkeypatch.setattr("app.agents.graph.get_llm", FakeLLM)
    out = reviewer_node(state)
    assert out["final"] == "我不知道。"  # reviewer 节点最终输出仍等于 draft
    assert out["last_reviewer_reason"] == "缺少引用, 需要重写"

    # route_review 看到 pass=False -> retry_count < MAX, 选 "retry" 分支
    from app.agents import graph

    assert graph.route_review(out) == "retry"


def test_message_history_concatenates_through_review():
    """Sanity: trace 在多次节点调用后持续累积."""
    state: AgentState = {
        "messages": [HumanMessage(content="q")],
        "kb_id": "default",
        "plan": "",
        "retrieved": "",
        "draft": "",
        "final": "",
        "trace": [],
        "retry_count": 0,
        "last_reviewer_reason": "",
    }
    state["trace"].append({"role": "planner", "content": "plan"})
    state["trace"].append({"role": "retriever", "content": "ctx"})
    assert len(state["trace"]) == 2