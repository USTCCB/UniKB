# -*- coding: utf-8 -*-
"""RAGAS 评估脚本
python -m tests.run_ragas_eval --kb default --dataset data/eval/qa.jsonl
数据集格式：每行 {\"question\": str, \"ground_truth\": str}
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from loguru import logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="default")
    ap.add_argument("--dataset", default="data/eval/qa.jsonl")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.warning(f"Dataset not found: {dataset_path}; create a demo one.")
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_path.write_text(
            '{"question": "UniKB 支持哪些 LLM？", "ground_truth": "DeepSeek、Qwen、OpenAI"}\n',
            encoding="utf-8",
        )

    items = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    logger.info(f"Loaded {len(items)} eval items")

    from app.rag.retriever import HybridRetriever
    from app.rag.reranker import CrossEncoderReranker
    from app.agents.llm_router import get_llm

    retriever = HybridRetriever(kb_id=args.kb)
    reranker = CrossEncoderReranker()
    llm = get_llm()

    answers, contexts, questions, gts = [], [], [], []
    for it in items:
        q = it["question"]
        cands = retriever.retrieve(q, top_k=args.top_k * 2)
        top = reranker.rerank(q, cands, top_k=args.top_k)
        ctx_texts = [c.get("document", "") for c in top]
        prompt = (
            "请基于【检索结果】回答问题。\n"
            + "\n".join(f"[{i+1}] {t[:300]}" for i, t in enumerate(ctx_texts))
            + f"\n\n问题: {q}\n回答:"
        )
        resp = llm.invoke(prompt)
        answers.append(resp.content)
        contexts.append(ctx_texts)
        questions.append(q)
        gts.append(it.get("ground_truth", ""))

    try:
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
        from datasets import Dataset
        ds = Dataset.from_dict({
            "question": questions, "answer": answers, "contexts": contexts, "ground_truth": gts,
        })
        result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
        logger.info(f"RAGAS Result: {result}")
        print(result)
    except ImportError:
        logger.warning("ragas/datasets 未安装，跳过自动评估。可 pip install ragas datasets 后重跑。")
        print(json.dumps({
            "note": "ragas not installed",
            "items": len(items),
            "sample_answer": answers[0] if answers else "",
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
