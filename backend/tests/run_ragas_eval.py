"""跑一次 RAGAS 评估,产出 4 大指标。

用法:
    cd backend
    python -m tests.run_ragas_eval --kb default \
        --dataset ../data/eval/qa.jsonl \
        --out ../data/eval/ragas_report.json

输出字段:
    faithfulness         答案与检索结果一致性(抑制幻觉)
    answer_relevancy     答案与问题的相关性
    context_precision    top_k 中相关 chunk 的占比
    context_recall       ground_truth 是否被召回
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from app.rag.pipeline import run_rag


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


async def _run_records(records: list[dict], kb: str, mode: str) -> dict:
    questions, answers, contexts, ground_truths = [], [], [], []
    per_sample = []

    for r in records:
        q = r["question"]
        gt = r.get("ground_truth", "")
        ref_ctx = r.get("reference_context") or []

        try:
            result = await run_rag(question=q, kb_id=kb, use_agent=(mode == "agent"))
        except Exception as e:
            logger.warning(f"RAG failed for q='{q[:30]}...': {e}; skipping.")
            continue

        questions.append(q)
        answers.append(result["answer"])
        contexts.append(result["contexts"])
        ground_truths.append(gt if isinstance(gt, list) else [gt])
        per_sample.append(
            {
                "question": q,
                "answer": result["answer"],
                "contexts": result["contexts"],
                "ground_truth": gt,
                "ref_context_for_inspection": ref_ctx,
            }
        )

    # ragas 需要 datasets.Dataset
    from datasets import Dataset

    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    # ragas 0.1.x 的 evaluate 接受 metrics list
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    scores = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    # ragas 0.1.x 返回 EvaluationResult,支持 .to_pandas() / 字典式访问
    try:
        score_dict = dict(scores)
    except Exception:
        # 兼容旧 API
        score_dict = {m.name: float(scores[m.name]) for m in (faithfulness, answer_relevancy, context_precision, context_recall)}

    return {"scores": score_dict, "samples": per_sample, "n": len(per_sample)}


def _fmt(val) -> str:
    try:
        return f"{float(val):.4f}"
    except Exception:
        return str(val)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", default="default")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parents[2] / "data" / "eval" / "qa.jsonl")
    parser.add_argument("--mode", choices=["rag", "agent"], default="rag")
    parser.add_argument("--out", type=Path, default=None, help="可选: 评估结果 JSON 输出路径")
    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(f"数据集不存在: {args.dataset}")

    records = load_jsonl(args.dataset)
    print(f"[UniKB RAGAS] Loaded {len(records)} QA pairs from {args.dataset}")

    report = asyncio.run(_run_records(records, args.kb, args.mode))

    print("\n=== RAGAS Scores ===")
    for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        val = report["scores"].get(name)
        print(f"  {name:>20s}: {_fmt(val)}")
    print(f"  n = {report['n']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "kb_id": args.kb,
            "mode": args.mode,
            "dataset": str(args.dataset),
            "scores": report["scores"],
            "samples": report["samples"],
        }
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已写入: {args.out}")


if __name__ == "__main__":
    main()
