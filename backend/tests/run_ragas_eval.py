"""真跑 RAGAS 评估, 落 faithfulness / answer_relevancy / context_precision / context_recall 4 项指标.

两种模式:
  - 默认 (--llm-mode fake_llm): 用内置 FakeLLM, 不需要 API key, 用于 CI / 沙箱环境验证脚本能跑通.
  - 真 LLM (--llm-mode real_llm): 用 settings.llm_provider 配置的真实 provider, 需要 .env 里配 API key.

输出:
  data/eval/ragas_report.json  (包含 scores + 每条样本的 question/answer/contexts/ground_truth)
  data/eval/baseline.json      (更紧凑, 只保留 4 项 scores + 元信息)

用法:
  cd backend
  python -m tests.run_ragas_eval --kb default \
      --dataset ../data/eval/qa.jsonl \
      --out ../data/eval/ragas_report.json \
      --baseline-out ../data/eval/baseline.json \
      --llm-mode fake_llm
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from datetime import datetime
from pathlib import Path

from loguru import logger


class _StubVertexAI:  # noqa: D401 - stub
    """空壳 vertexai, 仅供 ragas 0.1.10 import 链使用."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("VertexAI stub: not available in fake mode")


def _stub_langchain_community_vertexai() -> None:
    """ragas 0.1.10 在 import 时硬拉 langchain_community.chat_models.vertexai, 但它在
    sandbox 里会因为 pydantic/langchain-core 大版本错位触发元类冲突.

    这里在 ragas import 前先往 sys.modules 注入一个空壳 vertexai 模块,
    让 ragas 的 import 链跑通; 真正评估用的是 FakeLLM, 不依赖 vertexai.
    """
    pkg_lc = types.ModuleType("langchain_community")
    pkg_lc.__path__ = []  # 标记为 namespace package
    sys.modules.setdefault("langchain_community", pkg_lc)

    pkg_llms = types.ModuleType("langchain_community.llms")
    # ragas 直接 from langchain_community.llms import VertexAI
    pkg_llms.VertexAI = _StubVertexAI
    pkg_llms._VertexAICommon = _StubVertexAI
    sys.modules.setdefault("langchain_community.llms", pkg_llms)

    pkg_chat = types.ModuleType("langchain_community.chat_models")
    sys.modules.setdefault("langchain_community.chat_models", pkg_chat)

    fake_llms_module = types.ModuleType("langchain_community.llms.vertexai")
    fake_llms_module.VertexAI = _StubVertexAI
    fake_llms_module._VertexAICommon = _StubVertexAI
    sys.modules["langchain_community.llms.vertexai"] = fake_llms_module

    fake_chat_module = types.ModuleType("langchain_community.chat_models.vertexai")
    fake_chat_module.ChatVertexAI = _StubVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = fake_chat_module
    logger.debug("Stubbed langchain_community.vertexai for ragas import compatibility")


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# 先 stub 一遍, 因为后续 import ragas 会立即触发 langchain_community.vertexai.
_stub_langchain_community_vertexai()


# 复用 tests/_fakes.py 里的假实现, 跟 test_integration_retrieval.py 共用一套 fake,
# 避免两边各写一份还容易漂.
from tests._fakes import (  # noqa: E402
    FakeBM25Store,
    FakeLLM,
    FakeRagasEmbeddings,
    FakeRagasJudge,
    FakeReranker,
    FakeVectorStore,
    tokenize as _tokenize,  # noqa: F401  (备用)
)


# 这些 chunks 与 data/samples/产品手册.md 内容一致, 用于 fake 检索链路.
# 让 eval 在不依赖 chromadb / rank_bm25 的情况下跑出接近真实链路的表现.
FAKE_DOCS_BY_KB: dict[str, list[dict]] = {
    "default": [
        {"id": "c1", "document": "本产品自购买日起享受一年免费保修。保修范围包括非人为损坏的硬件故障。", "metadata": {"source": "产品手册-售后"}},
        {"id": "c2", "document": "用户可在 7 天内无理由退货，15 天内因质量问题可换货。", "metadata": {"source": "产品手册-售后"}},
        {"id": "c3", "document": "登录官网 → 我的订单 → 申请售后 → 提交故障描述与图片。", "metadata": {"source": "产品手册-售后流程"}},
        {"id": "c4", "document": "支持微信、支付宝、银联信用卡、企业月结。", "metadata": {"source": "产品手册-支付"}},
        {"id": "c5", "document": "一线城市 24h 达，二线城市 48h 达，偏远地区 3-5 个工作日。", "metadata": {"source": "产品手册-配送"}},
        {"id": "c6", "document": "发票需在订单完成后 30 天内申请开具，逾期不再补开。", "metadata": {"source": "产品手册-发票"}},
    ],
}


def install_fakes() -> None:
    """把 Embedding / Vector Store / BM25 / Reranker / LLM 都换成 fake, 让检索/重排/生成都能跑.

    顺序很重要: 设环境变量 -> 清 lru_cache -> 让 retriever 重新实例化 -> monkey-patch HybridRetriever.
    """
    import os
    from app.rag import embedding as emb_mod
    from app.rag import vector_store as vs_mod
    from app.rag import bm25_store as bm_mod
    from app.rag import reranker as rk_mod
    from app.rag import retriever as rt_mod
    from app.agents import llm_router as lr_mod

    os.environ["UNIKB_FAKE_EMBEDDING"] = "1"
    emb_mod.get_embedding_service.cache_clear()

    # 把 Chroma / BM25 / CrossEncoderReranker 全部短路, 不依赖外部库
    vs_mod.ChromaStore = FakeVectorStore
    bm_mod.BM25Store = FakeBM25Store
    rk_mod.CrossEncoderReranker = FakeReranker

    # HybridRetriever.retrieve 走真的逻辑 (FAKE_DOCS_BY_KB 已经在 FAKE_DOCS 准备好),
    # 我们在 __init__ 时把 FAKE_DOCS 灌进 vector / bm25 里.
    def _fake_init(self, kb_id: str = "default"):
        self.kb_id = kb_id
        # Fake store 构造时把 FAKE_DOCS 灌进去; Vector 需要 embedding, 我们直接
        # 给 c-dim 的 hash vector, 与 FakeEmbeddingService._vec 保持一致.
        from tests._fakes import FakeEmbeddingService as _FES
        _emb = _FES()
        emb = _emb.embed([c["document"] for c in FAKE_DOCS_BY_KB[kb_id]])
        self.vector_store = FakeVectorStore(collection_name=f"kb_{kb_id}")
        self.vector_store.add(
            ids=[c["id"] for c in FAKE_DOCS_BY_KB[kb_id]],
            documents=[c["document"] for c in FAKE_DOCS_BY_KB[kb_id]],
            embeddings=emb,
            metadatas=[c["metadata"] for c in FAKE_DOCS_BY_KB[kb_id]],
        )
        self.bm25_store = FakeBM25Store(persist_path=f"./data/bm25_{kb_id}.pkl")
        self.bm25_store.add(
            ids=[c["id"] for c in FAKE_DOCS_BY_KB[kb_id]],
            documents=[c["document"] for c in FAKE_DOCS_BY_KB[kb_id]],
            metadatas=[c["metadata"] for c in FAKE_DOCS_BY_KB[kb_id]],
        )
        self.embedding = _emb

    rt_mod.HybridRetriever.__init__ = _fake_init

    lr_mod.get_llm.cache_clear()
    lr_mod.get_llm = lambda: FakeLLM()
    logger.info("Fake mode installed: embedding/vector/bm25/rerank/llm all stubbed")





def _run_records(records: list[dict], kb: str, mode: str, llm_mode: str) -> dict:
    """ragas 0.1.10 evaluate() 内部 asyncio.run, 不能在 async 上下文里跑, 所以这里是同步函数.

    对每个样本同步调用 run_rag (里面无 asyncio, 只是顶层 await 但实际上 sync-兼容).
    """
    from app.rag.pipeline import run_rag
    import asyncio as _asyncio

    questions, answers, contexts, ground_truths = [], [], [], []
    per_sample = []

    for i, r in enumerate(records, 1):
        q = r["question"]
        gt = r.get("ground_truth", "")
        ref_ctx = r.get("reference_context") or []

        try:
            result = _asyncio.run(
                run_rag(question=q, kb_id=kb, top_k=3, use_agent=(mode == "agent"))
            )
        except Exception as e:
            logger.warning(f"RAG failed for q='{q[:30]}...': {e}; skipping.")
            continue

        questions.append(q)
        answers.append(result["answer"])
        contexts.append(result["contexts"])
        # ragas 0.1.10 要求 ground_truth 是 str, 不是 list[str]
        gt_str = gt if isinstance(gt, str) else (gt[0] if gt else "")
        ground_truths.append(gt_str)
        per_sample.append(
            {
                "question": q,
                "answer": result["answer"],
                "contexts": result["contexts"],
                "ground_truth": gt_str,
                "ref_context_for_inspection": ref_ctx,
            }
        )
        logger.info(f"[{i}/{len(records)}] {q[:40]} -> {result['answer'][:60]}")

    from datasets import Dataset
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

    eval_kwargs = {
        "metrics": [faithfulness, answer_relevancy, context_precision, context_recall],
    }
    if llm_mode == "fake_llm":
        # 让指标自己跑, 不要 RAGAS 默认去 OpenAI 拉 chat model.
        # 1) 直接传 llm + embeddings 给 evaluate().
        # 2) 同时 monkey-patch ragas.llms.base.llm_factory 和 ragas.embeddings.factory,
        #    防止某个 metric 内部 None 检查时再次触发远端初始化.
        import ragas.llms.base as ragas_llm_base
        import ragas.embeddings as ragas_emb_base
        ragas_llm_base.llm_factory = lambda *a, **kw: FakeRagasJudge()
        if hasattr(ragas_emb_base, "embedding_factory"):
            ragas_emb_base.embedding_factory = lambda *a, **kw: FakeRagasEmbeddings()
        # 再保险: 也覆盖每个 metric 实例
        for m in (faithfulness, answer_relevancy, context_precision, context_recall):
            try:
                m.llm = FakeRagasJudge()
            except Exception:
                pass
            try:
                m.embeddings = FakeRagasEmbeddings()
            except Exception:
                pass
        eval_kwargs["llm"] = FakeRagasJudge()
        eval_kwargs["embeddings"] = FakeRagasEmbeddings()
    scores = evaluate(ds, **eval_kwargs)
    try:
        score_dict = dict(scores)
    except Exception:
        score_dict = {
            m.name: float(scores[m.name])
            for m in (faithfulness, answer_relevancy, context_precision, context_recall)
        }

    return {"scores": score_dict, "samples": per_sample, "n": len(per_sample)}


def _fmt(val) -> str:
    try:
        return f"{float(val):.4f}"
    except Exception:
        return str(val)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", default="default")
    parser.add_argument(
        "--dataset", type=Path,
        default=Path(__file__).parents[2] / "data" / "eval" / "qa.jsonl",
    )
    parser.add_argument("--mode", choices=["rag", "agent"], default="rag")
    parser.add_argument(
        "--llm-mode", choices=["fake_llm", "real_llm"], default="fake_llm",
        help="fake_llm 用内置 fake 跑通评估脚本; real_llm 用真实 provider (需 .env API key)",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--baseline-out", type=Path, default=None)
    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(f"数据集不存在: {args.dataset}")

    if args.llm_mode == "fake_llm":
        install_fakes()
    else:
        logger.info("Real LLM mode: using settings.llm_provider (需要 .env 配置 API key)")

    records = load_jsonl(args.dataset)
    print(f"[UniKB RAGAS] llm_mode={args.llm_mode} | dataset={args.dataset} | n={len(records)}")

    report = _run_records(records, args.kb, args.mode, args.llm_mode)

    print("\n=== RAGAS Scores ===")
    for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        val = report["scores"].get(name)
        print(f"  {name:>20s}: {_fmt(val)}")
    print(f"  n = {report['n']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # 同样的 NaN -> 0 清理
        scores_clean: dict = {}
        for k, v in report["scores"].items():
            try:
                fv = float(v)
                if fv != fv:
                    scores_clean[k] = 0.0
                else:
                    scores_clean[k] = fv
            except Exception:
                scores_clean[k] = 0.0
        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "kb_id": args.kb,
            "mode": args.mode,
            "llm_mode": args.llm_mode,
            "dataset": str(args.dataset),
            "scores": scores_clean,
            "samples": report["samples"],
            "note": (
                "fake_llm 路径: 用于 CI / 沙箱验证脚本能跑通, 数字不代表真实模型效果."
                if args.llm_mode == "fake_llm"
                else "real_llm 路径: 用真实 API key 跑的分数, 可作 baseline."
            ),
        }
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已写入: {args.out}")

    if args.baseline_out:
        args.baseline_out.parent.mkdir(parents=True, exist_ok=True)
        # NaN 转 0: ragas 在某些 metric parser 全失败时会给 nan, JSON 标准不允许 NaN,
        # 这里统一处理成 0 并在 baseline 里记下哪几个 metric 退化了.
        cleaned = {}
        nan_metrics = []
        for k, v in report["scores"].items():
            try:
                fv = float(v)
                if fv != fv:  # NaN check
                    nan_metrics.append(k)
                    cleaned[k] = 0.0
                else:
                    cleaned[k] = fv
            except Exception:
                nan_metrics.append(k)
                cleaned[k] = 0.0
        compact = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "llm_mode": args.llm_mode,
            "kb_id": args.kb,
            "mode": args.mode,
            "n_samples": report["n"],
            "scores": cleaned,
            "nan_metrics": nan_metrics,
            "note": (
                "fake LLM 跑出来的分数, 仅作脚本通断校验, 不作 baseline."
                "nan_metrics 列出退化的指标 (parser 全失败, 已置 0)."
                if args.llm_mode == "fake_llm"
                else "请把这里数字作为 baseline 提交, 后续优化做对比."
            ),
        }
        args.baseline_out.write_text(
            json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"baseline 已写入: {args.baseline_out}")


if __name__ == "__main__":
    main()