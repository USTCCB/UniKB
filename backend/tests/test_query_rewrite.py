"""测试: 查询改写 QueryRewriter (LLM + 确定性回退)."""
from __future__ import annotations

from app.rag.query_rewrite import QueryRewriter


def test_rewrite_includes_original():
    qw = QueryRewriter(enabled=True)
    out = qw.rewrite("保修期多久？", llm=None)
    assert out[0] == "保修期多久？"


def test_fallback_splits_compound_question():
    # 无 LLM 时, 复合问题按标点拆子句, 原句 + 子句分别检索.
    qw = QueryRewriter(enabled=True, max_variants=3)
    out = qw.rewrite("你们支持哪些支付方式？配送要几天？", llm=None)
    assert len(out) >= 2
    assert "你们支持哪些支付方式？配送要几天？" in out
    assert "配送要几天" in out


def test_single_clause_no_split():
    qw = QueryRewriter(enabled=True)
    out = qw.rewrite("保修多久", llm=None)
    # 单子句不拆, 只返回原句
    assert out == ["保修多久"]


def test_llm_rewrite_returns_variants():
    def fake_llm(prompt: str) -> str:
        return "保修期限是多久\n支付方式有哪些\n如何申请售后"

    qw = QueryRewriter(enabled=True, max_variants=3)
    out = qw.rewrite("你们的产品售后政策", llm=fake_llm)
    # 原句 + 3 条改写
    assert out[0] == "你们的产品售后政策"
    assert "保修期限是多久" in out
    assert "支付方式有哪些" in out
    assert "如何申请售后" in out


def test_disabled_returns_only_original():
    qw = QueryRewriter(enabled=False)
    out = qw.rewrite("任意问题？子句二？", llm=None)
    assert out == ["任意问题？子句二？"]


def test_deduplicates_variants():
    def fake_llm(prompt: str) -> str:
        return "保修期多久？\n保修期多久？"  # 重复

    qw = QueryRewriter(enabled=True)
    out = qw.rewrite("保修期多久？", llm=fake_llm)
    # 去重后不应有重复项
    assert len(out) == len(set(out))
