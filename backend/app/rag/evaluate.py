"""离线召回评估 (Recall@K).

RAG 链路调优离不开**可量化的召回指标**. 这里提供:
  - `recall_at_k(relevant, retrieved, k)`: 单条问答的 Recall@K.
  - `RecallEvaluator`: 在一份 (question, relevant_chunk_ids) 数据集上,
    跑 retriever 统计 Recall@1 / @3 / @5 / @10, 用来对比
    "加/不加清洗、自适应切分、父文档召回、重排优化" 前后的差异.

`retriever.retrieve` 返回的是 dict (含 id), 这里统一按 `id` 对齐.
数据集格式 (JSONL / 列表):
    {"question": "...", "relevant_ids": ["doc_x_c0", "doc_x_c3"]}
"""
from __future__ import annotations

import json
from typing import Callable, Dict, List, Sequence


def recall_at_k(relevant: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    """Recall@K = |relevant ∩ retrieved[:k]| / |relevant|.

    relevant 为空时定义 Recall=1.0 (无相关项可漏).
    """
    if not relevant:
        return 1.0
    if k <= 0:
        return 0.0
    rel_set = set(relevant)
    # 必须按**集合交集**算命中, 不能逐条累加:
    # 父文档召回 (连坐) 会让同一个 chunk id 在候选里出现多次, 逐条累加会把
    # Recall 算到 > 1.0, 直接把评估指标刷虚高.
    top = list(retrieved)[:k]
    hits = len(rel_set & set(top))
    return hits / len(rel_set)


class RecallEvaluator:
    """在数据集上评估 retriever 的 Recall@K."""

    DEFAULT_KS = (1, 3, 5, 10)

    def __init__(self, ks: Sequence[int] = DEFAULT_KS):
        self.ks = tuple(ks)

    def evaluate(
        self,
        dataset: List[dict],
        retrieve_fn: Callable[[str], Sequence[dict]],
        id_key: str = "id",
    ) -> Dict[str, object]:
        """retrieve_fn(question) -> list of dict (含 id_key).

        返回 {recall@k: float, per_question: [...], n: int}.
        """
        per_question: List[dict] = []
        sum_recall = {k: 0.0 for k in self.ks}
        for item in dataset:
            q = item.get("question", "")
            relevant = item.get("relevant_ids", []) or item.get("relevant", [])
            retrieved = retrieve_fn(q)
            ids = [r.get(id_key) for r in retrieved if r.get(id_key)]
            row = {"question": q, "recall": {}}
            for k in self.ks:
                r = recall_at_k(relevant, ids, k)
                sum_recall[k] += r
                row["recall"][f"@{k}"] = round(r, 4)
            per_question.append(row)
        n = len(dataset) or 1
        return {
            "n": len(dataset),
            "recall": {f"@{k}": round(sum_recall[k] / n, 4) for k in self.ks},
            "per_question": per_question,
        }

    @staticmethod
    def load_dataset(path: str) -> List[dict]:
        """读取 JSONL 或 JSON 数组格式的评估集."""
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            return []
        if text.startswith("["):
            return json.loads(text)
        out = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out


if __name__ == "__main__":
    import sys

    ds_path = sys.argv[1] if len(sys.argv) > 1 else "eval/recall_dataset.jsonl"
    data = RecallEvaluator.load_dataset(ds_path)
    print(f"loaded {len(data)} qa pairs from {ds_path}")
    print(json.dumps({"ks": RecallEvaluator.DEFAULT_KS}, ensure_ascii=False, indent=2))
