"""测试 12: run_ragas_eval 脚本自身 (纯 helper 函数)."""
from __future__ import annotations

import json
from pathlib import Path


def test_load_jsonl(tmp_path: Path):
    p = tmp_path / "qa.jsonl"
    p.write_text(
        json.dumps({"question": "q1", "ground_truth": "a1"}, ensure_ascii=False) + "\n"
        + json.dumps({"question": "q2", "ground_truth": "a2"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    from tests.run_ragas_eval import load_jsonl

    records = load_jsonl(p)
    assert len(records) == 2
    assert records[0]["question"] == "q1"


def test_load_jsonl_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "qa.jsonl"
    p.write_text(
        "\n"
        + json.dumps({"question": "q", "ground_truth": "a"}, ensure_ascii=False)
        + "\n\n",
        encoding="utf-8",
    )
    from tests.run_ragas_eval import load_jsonl

    assert len(load_jsonl(p)) == 1


def test_load_jsonl_handles_utf8_chinese(tmp_path: Path):
    p = tmp_path / "qa.jsonl"
    p.write_text(
        json.dumps({"question": "UniKB 支持哪些 LLM?", "ground_truth": "DeepSeek/Qwen/OpenAI"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    from tests.run_ragas_eval import load_jsonl

    records = load_jsonl(p)
    assert "DeepSeek" in records[0]["ground_truth"]


def test_fmt_float():
    from tests.run_ragas_eval import _fmt

    assert _fmt(0.87654) == "0.8765"
    assert _fmt(None) == "None"
    assert _fmt("oops") == "oops"
    assert _fmt(0) == "0.0000"
