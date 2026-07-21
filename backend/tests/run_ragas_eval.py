"""跑一次 RAGAS 评估.

用法:
    cd backend
    python -m tests.run_ragas_eval --kb default --dataset ../data/eval/qa.jsonl

依赖: 已安装 ragas + datasets + 一个能跑的 LLM (DEEPSEEK_API_KEY 等).
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from datasets import Dataset

from app.rag.pipeline import run_rag  # 项目里的主检索+生成入口


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def _evaluate(records: list[dict], kb: str) -> dict:
    questions, answers, contexts, ground_truths = [], [], [], []
    for r in records:
        q = r["question"]
        ref_ctx = r.get("reference_context", [])  # noqa: F841  # reserved for custom metrics later
        gt = r["ground_truth"]

        result = await run_rag(question=q, kb_id=kb, top_k=5)

        questions.append(q)
        answers.append(result["answer"])
        contexts.append(result["contexts"])
        ground_truths.append(gt if isinstance(gt, list) else [gt])

    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    scores = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", default="default")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parents[2] / "data" / "eval" / "qa.jsonl")
    args = parser.parse_args()

    records = load_jsonl(args.dataset)
    print(f"Loaded {len(records)} QA pairs from {args.dataset}")

    scores = asyncio.run(_evaluate(records, args.kb))

    print("\n=== RAGAS Scores ===")
    for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        val = scores.get(name) if hasattr(scores, "get") else scores[name]
        print(f"{name:>20s}: {val:.4f}" if isinstance(val, float) else f"{name:>20s}: {val}")


if __name__ == "__main__":
    main()