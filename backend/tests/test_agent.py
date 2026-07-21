"""测试 6: LangGraph 多 Agent 节点."""
from app.agents.graph import AgentState, planner_node, reviewer_node


def test_planner_node_returns_plan_with_steps():
    state = AgentState(question="UniKB 怎么部署?", context=[])
    out = planner_node(state)
    assert "plan" in out
    assert isinstance(out["plan"], list)
    assert len(out["plan"]) >= 1


def test_reviewer_node_flags_low_confidence_answer():
    state = AgentState(
        question="q",
        context=["doc1"],
        draft="我不知道",  # 低置信
    )
    out = reviewer_node(state)
    assert out["needs_retry"] is True


def test_reviewer_node_accepts_confident_answer():
    state = AgentState(
        question="q",
        context=["doc1", "doc2"],
        draft="根据 doc1 和 doc2, 答案是 42。",
    )
    out = reviewer_node(state)
    assert out["needs_retry"] is False