"""P1-9 测试: Reviewer 结构化 JSON 输出 + 健壮解析.

覆盖:
1. 纯 JSON 字符串 -> pass=True 时 final=draft, 不再调 fix.
2. 纯 JSON 字符串 -> pass=False 时 final!=draft (因为会调 fix 重写).
3. markdown 代码块包裹的 JSON -> 仍能解析.
4. 文本混杂: "{\"pass\": false, \"reason\": \"...\"}" -> 正确解析.
5. 完全不能 parse -> fallback 默认 pass=True (避免死循环).
6. pass 字段类型兼容 (true/false/"true"/1/0).
7. 老 bug 修复: "审核通过率" 这种 token 不再误判 (老逻辑会被 substring 触发).
8. trace 同时记录 raw 和 parsed decision.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.agents.graph import (
    REVIEWER_SYS,
    _normalize_decision,
    _parse_reviewer_decision,
    reviewer_node,
)


# ============ _parse_reviewer_decision ============

class TestParseReviewerDecision:
    def test_pure_json_pass_true(self):
        out = '{"pass": true, "reason": "ok"}'
        d = _parse_reviewer_decision(out)
        assert d == {"pass": True, "reason": "ok"}

    def test_pure_json_pass_false(self):
        out = '{"pass": false, "reason": "no citation"}'
        d = _parse_reviewer_decision(out)
        assert d == {"pass": False, "reason": "no citation"}

    def test_markdown_code_block(self):
        out = '```json\n{"pass": true, "reason": "ok"}\n```'
        d = _parse_reviewer_decision(out)
        assert d["pass"] is True
        assert d["reason"] == "ok"

    def test_markdown_code_block_no_lang(self):
        out = '```\n{"pass": false, "reason": "too short"}\n```'
        d = _parse_reviewer_decision(out)
        assert d["pass"] is False
        assert d["reason"] == "too short"

    def test_text_with_json_inline(self):
        out = '审查结果: {"pass": false, "reason": "无引用"} 以上.'
        d = _parse_reviewer_decision(out)
        assert d["pass"] is False
        assert d["reason"] == "无引用"

    def test_unparseable_falls_back_to_pass_true(self):
        """完全不可解析时, 默认 pass=True 防止 reviewer 死循环重写."""
        out = "LLM 抽风了 输出乱码"
        d = _parse_reviewer_decision(out)
        assert d["pass"] is True
        assert "parse" in d["reason"].lower() or "unparseable" in d["reason"].lower() or "默认" in d["reason"]

    def test_empty_string_falls_back(self):
        d = _parse_reviewer_decision("")
        assert d["pass"] is True


class TestNormalizeDecision:
    def test_bool_pass(self):
        assert _normalize_decision({"pass": True})["pass"] is True
        assert _normalize_decision({"pass": False})["pass"] is False

    def test_string_pass_values(self):
        for s in ("true", "True", "TRUE", "yes", "1", "pass", "通过"):
            assert _normalize_decision({"pass": s})["pass"] is True, f"failed for {s!r}"
        for s in ("false", "False", "no", "0", "fail"):
            assert _normalize_decision({"pass": s})["pass"] is False, f"failed for {s!r}"

    def test_int_pass_values(self):
        assert _normalize_decision({"pass": 1})["pass"] is True
        assert _normalize_decision({"pass": 0})["pass"] is False

    def test_reason_truncated_to_200(self):
        long = "x" * 500
        d = _normalize_decision({"pass": False, "reason": long})
        assert len(d["reason"]) == 200

    def test_reason_missing(self):
        d = _normalize_decision({"pass": True})
        assert d["reason"] == ""


# ============ 老 bug: 子串误判 ============

class TestLegacyBugFixed:
    """老逻辑是 `if "通过" in out.content`. 一些 token 会被误判.

    现在改成解析 JSON, 这些都不再误判. 但我们验证 parse 不会崩.
    """

    def test_pass_token_no_longer_misclassifies(self):
        """旧逻辑: '审核通过率很高' 会被误判为 pass. 新逻辑不依赖 substring."""
        # 实际不会被 parse 成 JSON, 会 fallback 到 pass=True (默认).
        # 但关键是, 这不再是基于 substring 判断.
        out = "审核通过率很高"  # 不是合法 JSON
        d = _parse_reviewer_decision(out)
        # 因为不能 parse, fallback 默认 pass=True. (旧逻辑也判通过, 但理由是错的)
        assert d["pass"] is True
        assert "默认" in d["reason"] or "parse" in d["reason"].lower() or "unparseable" in d["reason"]


# ============ reviewer_node 集成 ============

def _make_state(draft: str, decision_raw: str, fix_content: str = "fixed draft"):
    """构造 reviewer_node 所需的 state."""
    from langchain_core.messages import HumanMessage
    return {
        "messages": [HumanMessage(content="用户问题")],
        "plan": "",
        "retrieved": "",
        "draft": draft,
        "final": "",
        "trace": [],
        # 测试用
        "_decision_raw": decision_raw,
        "_fix_content": fix_content,
    }


def _patch_get_llm(monkeypatch, raw: str, fix: str):
    """把 get_llm 替换成给 raw 输出 + fix 内容."""
    from app.agents import graph

    fake_llm = MagicMock()
    responses = [raw, fix]
    counter = {"i": 0}

    def _invoke(msgs):
        i = counter["i"]
        counter["i"] += 1
        r = MagicMock()
        r.content = responses[i] if i < len(responses) else fix
        return r

    fake_llm.invoke = MagicMock(side_effect=_invoke)
    monkeypatch.setattr(graph, "get_llm", lambda: fake_llm)
    return fake_llm


def test_reviewer_pass_keeps_draft(monkeypatch):
    """pass=True 时 final=draft, 不再调 fix."""
    from app.agents import graph

    state = _make_state("draft content", '{"pass": true, "reason": "ok"}')
    _patch_get_llm(monkeypatch, '{"pass": true, "reason": "ok"}', "should not be used")

    out = graph.reviewer_node(state)
    assert out["final"] == "draft content"
    # trace 应包含 raw + decision
    rev = [t for t in out["trace"] if t["role"] == "reviewer"]
    assert len(rev) == 1
    assert rev[0]["content"]["decision"]["pass"] is True
    assert rev[0]["content"]["decision"]["reason"] == "ok"


def test_reviewer_fail_triggers_fix(monkeypatch):
    """pass=False 时 final != draft (fix 重写)."""
    from app.agents import graph

    state = _make_state("draft content", '{"pass": false, "reason": "需要引用"}', "fixed content")
    fake_llm = _patch_get_llm(
        monkeypatch,
        '{"pass": false, "reason": "需要引用"}',
        "fixed content",
    )

    out = graph.reviewer_node(state)
    # 调了第二次 llm.invoke (fix)
    assert fake_llm.invoke.call_count == 2
    assert out["final"] == "fixed content"
    # trace 里有 reason
    rev = [t for t in out["trace"] if t["role"] == "reviewer"]
    assert rev[0]["content"]["decision"]["pass"] is False
    assert rev[0]["content"]["decision"]["reason"] == "需要引用"


def test_reviewer_malformed_falls_back_to_pass(monkeypatch):
    """不可 parse 的输出走 fallback (pass=True), 不调 fix."""
    from app.agents import graph

    state = _make_state("draft content", "garbage output")
    fake_llm = _patch_get_llm(monkeypatch, "garbage output", "should not be used")

    out = graph.reviewer_node(state)
    # 只调用了一次 (审查), 没有调 fix
    assert fake_llm.invoke.call_count == 1
    assert out["final"] == "draft content"
    rev = [t for t in out["trace"] if t["role"] == "reviewer"]
    assert rev[0]["content"]["decision"]["pass"] is True


def test_reviewer_sys_prompts_json_format():
    """REVIEWER_SYS 必须显式要求 JSON 格式."""
    assert "{" in REVIEWER_SYS
    assert "pass" in REVIEWER_SYS
    assert "reason" in REVIEWER_SYS