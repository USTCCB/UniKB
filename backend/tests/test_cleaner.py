"""测试: 文档清洗 DocumentCleaner (40+ 正则规则)."""
from __future__ import annotations

from app.rag.cleaner import DocumentCleaner


def test_rule_count_exceeds_40():
    # 文章强调 40+ 条正则, 这里硬断言, 防止后人把规则删到太少.
    assert DocumentCleaner().rule_count >= 40


def test_removes_html_tags_and_entities():
    raw = "<div>你好 <b>世界</b></div>&nbsp;测试&#39;引用"
    out = DocumentCleaner().clean(raw)
    assert "<div>" not in out
    assert "<b>" not in out
    assert "&nbsp;" not in out
    assert "你好" in out and "世界" in out


def test_removes_markdown_artifacts():
    raw = "## 标题\n[链接文本](http://x.com) ![图](a.png) **加粗** `代码`"
    out = DocumentCleaner().clean(raw)
    assert "##" not in out
    assert "http://x.com" not in out
    assert "![图]" not in out
    assert "链接文本" in out and "加粗" in out and "代码" in out


def test_removes_control_and_zero_width_chars():
    raw = "正\u200b常\u0000文本﻿﻿"
    out = DocumentCleaner().clean(raw)
    assert "\u200b" not in out
    assert "\x00" not in out
    assert out == "正常文本"


def test_strips_page_numbers_and_toc_dots():
    raw = "第一章 引言\n第 1 页\n目录项............12\n正文内容"
    out = DocumentCleaner().clean(raw)
    assert "第 1 页" not in out
    assert "............12" not in out
    assert "正文内容" in out


def test_collapses_redundant_whitespace():
    raw = "段落一\n\n\n\n段落二    多余空格\t\t制表"
    out = DocumentCleaner().clean(raw)
    assert "\n\n\n" not in out
    assert "    多余" not in out
    assert "段落一" in out and "段落二" in out


def test_disabled_cleaner_returns_input():
    raw = "<tag>保留</tag>"
    out = DocumentCleaner(enabled=False).clean(raw)
    assert out == "<tag>保留</tag>"


def test_clean_is_idempotent():
    raw = "<p>  Hello   World \n\n\n 测试 </p>\x00\u200b"
    c = DocumentCleaner()
    once = c.clean(raw)
    twice = c.clean(once)
    assert once == twice
