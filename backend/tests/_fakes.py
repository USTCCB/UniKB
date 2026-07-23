"""测试用的 fake 实现: 不依赖 torch / chromadb / sentence-transformers.

设计要点:
1. FakeEmbeddingService: 字符哈希到固定维度, deterministic.
2. FakeVectorStore / FakeBM25Store: 内存版, 接口对齐 ChromaStore / BM25Store,
   但不读写磁盘, 不加载任何重依赖.
3. FakeLLM / FakeRagasJudge: 跑通 LLM 接口 + RAGAS judge 协议.

和 backend/tests/run_ragas_eval.py 里那一坨是重复代码, 抽出来共用;
后者改成 import 这里的 fake, 行为一致.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any


# ----- 维度对齐: bge-small 是 512, fake 用 64 即可, 检索方向不受影响 -----
class FakeEmbeddingService:
    dim = 64

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for ch in text:
            idx = int(hashlib.md5(ch.encode()).hexdigest(), 16) % self.dim
            v[idx] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str):
        return self._vec(text)


# ----- Tokenize, 与 bm25_store 保持一致 -----
def tokenize(text: str) -> list[str]:
    text = text.lower()
    return re.findall(r"[一-龥]+|[a-z0-9]+", text)


# ----- Fake BM25: 简单的关键词计数 + IDF -----
class FakeBM25Store:
    """in-memory 简易 BM25: 文档数较小时足够接近, 不依赖 rank_bm25."""

    def __init__(self, persist_path: str | None = None):
        self.persist_path = persist_path
        self.docs: list[dict] = []
        self._df: dict[str, int] = {}
        self._avgdl: float = 0.0

    def add(self, ids, documents, metadatas):
        for i, d in enumerate(documents):
            self.docs.append({"id": ids[i], "text": d, "metadata": metadatas[i]})
            seen = set()
            for tok in tokenize(d):
                if tok not in seen:
                    self._df[tok] = self._df.get(tok, 0) + 1
                    seen.add(tok)
        if self.docs:
            self._avgdl = sum(len(tokenize(d["text"])) for d in self.docs) / len(self.docs)

    def query(self, text: str, top_k: int = 20):
        if not self.docs:
            return []
        q_tokens = tokenize(text)
        if not q_tokens:
            return []
        import math

        scores = []
        N = len(self.docs)
        for d in self.docs:
            toks = tokenize(d["text"])
            if not toks:
                scores.append(0.0)
                continue
            tf = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            s = 0.0
            for qt in q_tokens:
                if qt not in tf:
                    continue
                df = max(1, self._df.get(qt, 1))
                idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
                # BM25 经典公式 (k1=1.5, b=0.75)
                num = tf[qt] * (1.5 + 1)
                den = tf[qt] + 1.5 * (1 - 0.75 + 0.75 * len(toks) / max(1, self._avgdl))
                s += idf * num / den
            scores.append(s)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        out = []
        for idx, sc in ranked:
            d = self.docs[idx]
            out.append({"id": d["id"], "document": d["text"], "metadata": d["metadata"], "score": float(sc)})
        return out

    def count(self):
        return len(self.docs)

    def load(self):
        return None


# ----- Fake Vector Store: 用 cos sim + numpy 都不行, 直接用 hash vector + 点积 -----
class FakeVectorStore:
    """in-memory 向量库: add / query / delete, 接口对齐 ChromaStore.

    不依赖 chromadb / numpy. 用 Python list 计算余弦.
    """

    def __init__(self, collection_name: str = "default", **kwargs):
        self.collection_name = collection_name
        self.docs: list[dict] = []  # [{id, document, embedding, metadata}]
        self._vec_dim = 64

    def add(self, ids, documents, embeddings, metadatas):
        # 模拟 upsert: 删掉旧 id 再加
        ids_set = set(ids)
        self.docs = [d for d in self.docs if d["id"] not in ids_set]
        for i in range(len(ids)):
            self.docs.append(
                {
                    "id": ids[i],
                    "document": documents[i],
                    "embedding": embeddings[i],
                    "metadata": metadatas[i],
                }
            )

    def query(self, query_embedding, top_k: int = 20, where=None):
        if not self.docs:
            return []
        scored = []
        q = query_embedding
        qn = sum(x * x for x in q) ** 0.5 or 1.0
        for d in self.docs:
            e = d["embedding"]
            en = sum(x * x for x in e) ** 0.5 or 1.0
            dot = sum(a * b for a, b in zip(q, e))
            sim = dot / (qn * en)
            scored.append((sim, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for sim, d in scored[:top_k]:
            out.append(
                {
                    "id": d["id"],
                    "document": d["document"],
                    "metadata": d["metadata"],
                    "distance": 1.0 - sim,  # chroma 接口: 越小越相关
                }
            )
        return out

    def count(self):
        return len(self.docs)

    def delete(self, ids):
        ids_set = set(ids)
        self.docs = [d for d in self.docs if d["id"] not in ids_set]

    def reset(self):
        self.docs = []


# ----- Fake Reranker: 简单 token overlap, 不依赖 sentence_transformers -----
class FakeReranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or "fake-cross-encoder"

    def rerank(self, query, candidates, top_k: int = 5):
        if not candidates:
            return []
        q_tokens = set(tokenize(query))
        if not q_tokens:
            return candidates[:top_k]
        scored = []
        for c in candidates:
            doc = c.get("document", "")
            d_tokens = set(tokenize(doc))
            inter = len(q_tokens & d_tokens)
            union = len(q_tokens | d_tokens) or 1
            score = inter / union  # Jaccard
            c2 = dict(c)
            c2["rerank_score"] = float(score)
            scored.append(c2)
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]


# ----- Fake LLM: 走 pipeline 那条 prompt, 把 contexts 拼成 [1] 引用答案 -----
class FakeLLM:
    class _Resp:
        def __init__(self, content: str):
            self.content = content

    def invoke(self, prompt):
        ctx = ""
        if "【检索结果】" in prompt and "【用户问题】" in prompt:
            try:
                ctx = prompt.split("【检索结果】", 1)[1].split("【用户问题】", 1)[0]
            except Exception:
                ctx = ""
        chunks = []
        buf = ""
        for line in ctx.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("[") and "]" in line:
                if buf:
                    chunks.append(buf.strip())
                buf = line.split("]", 1)[-1].strip()
            else:
                buf += " " + line
        if buf:
            chunks.append(buf.strip())
        if not chunks:
            return self._Resp("未找到相关信息。")
        body = " ".join(chunks[:3])[:300]
        return self._Resp(f"根据检索结果, 回答如下: {body} [1]")


# ----- Fake RAGAS Judge: 不同 metric 期待的 schema 各给一份 -----
class FakeRagasJudge:
    """实现 BaseRagasLLM 抽象, 让 RAGAS evaluate 跑通."""

    class _Gen:
        def __init__(self, text):
            self.text = text

    class _Resp:
        def __init__(self, text, n: int = 1):
            self.generations = [[FakeRagasJudge._Gen(text)] * n] if n > 1 else [[FakeRagasJudge._Gen(text)]]

    def __init__(self):
        self.run_config = None

    def _judge(self, prompt: Any) -> str:
        try:
            if hasattr(prompt, "to_string"):
                prompt = prompt.to_string()
            elif hasattr(prompt, "text"):
                prompt = prompt.text
            elif not isinstance(prompt, str):
                prompt = str(prompt)
        except Exception:
            prompt = str(prompt)
        p = prompt.lower()
        # context_precision: 单 {reason, verdict}
        if "verify if the context was useful" in p or "give verdict as" in p:
            return json.dumps({"reason": "fake: context useful", "verdict": 1}, ensure_ascii=False)
        # context_recall: list of {statement, attributed, reason}
        if "attributed to the given context" in p or "yes (1) or no (0)" in p:
            return json.dumps(
                [
                    {"statement": "fake-statement", "attributed": 1, "reason": "fake: attributed"},
                    {"statement": "fake-statement-2", "attributed": 1, "reason": "fake: attributed"},
                ],
                ensure_ascii=False,
            )
        # faithfulness_nli_verdict: list of {statement, reason, verdict}
        if (
            "judge the faithfulness of a series of statements" in p
            or "is the statement a hallucination" in p
            or "can be directly inferred based on the context" in p
        ):
            return json.dumps(
                [
                    {"statement": "fake-statement-A", "reason": "fake: faithful", "verdict": 1},
                    {"statement": "fake-statement-B", "reason": "fake: faithful", "verdict": 1},
                ],
                ensure_ascii=False,
            )
        # faithfulness_long_form: list of {sentence_index, simpler_statements}
        if "break down each sentence" in p or "fully understandable statements" in p:
            return json.dumps(
                [
                    {"sentence_index": 0, "simpler_statements": ["fake-fact-A1"]},
                    {"sentence_index": 1, "simpler_statements": ["fake-fact-B1"]},
                ],
                ensure_ascii=False,
            )
        # answer_relevancy: 单 {question, noncommittal}
        if "noncommittal" in p or "generate a question for the given answer" in p:
            return json.dumps({"question": "fake-question", "noncommittal": 0}, ensure_ascii=False)
        return json.dumps(
            {"statement": "fake-default", "attributed": 1, "reason": "fake-default", "verdict": 1},
            ensure_ascii=False,
        )

    def _resp_from_prompt(self, prompt, n: int = 1):
        return self._Resp(self._judge(prompt), n=n)

    # ragas BaseRagasLLM 抽象
    def generate_text(self, prompt, n: int = 1, **kwargs):
        return self._resp_from_prompt(prompt, n=n)

    async def agenerate_text(self, prompt, n: int = 1, **kwargs):
        return self._resp_from_prompt(prompt, n=n)

    def set_run_config(self, run_config):
        self.run_config = run_config
        return self

    def set_callbacks(self, *_args, **_kwargs):
        return self

    async def generate(self, prompt, **kwargs):
        return self._resp_from_prompt(prompt)

    async def agenerate(self, prompt, **kwargs):
        return self._resp_from_prompt(prompt)

    async def ainvoke(self, prompt, **kwargs):
        return self._resp_from_prompt(prompt)

    def invoke(self, prompt, **kwargs):
        return self._resp_from_prompt(prompt)

    def __call__(self, prompt, **kwargs):
        return self._resp_from_prompt(prompt)


# ----- Fake RAGAS Embeddings: RAGAS 内部有些指标 (answer_relevancy) 用 embeddings 算相似度 -----
class FakeRagasEmbeddings:
    """32 维 binary 词袋 (按字符 hash 落桶), 不依赖任何外部模型."""

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    def __call__(self, text):
        return self._vec(text)

    @staticmethod
    def _vec(text: str) -> list[float]:
        v = [0.0] * 32
        for ch in text.lower():
            v[ord(ch) % 32] = 1.0
        return v


# ----- 一键安装所有 fake (供集成测试 / 评估脚本共用) -----
def install_all_fakes(monkeypatch=None) -> None:
    """把 app.rag.{vector_store, bm25_store, embedding, reranker} + agents.llm_router
    里所有的重依赖都换成 fake. 既支持 pytest monkeypatch fixture, 也支持脚本内调用.

    用法:
        from tests._fakes import install_all_fakes
        install_all_fakes(monkeypatch)         # 测试里
        install_all_fakes()                     # 脚本里 (无 monkeypatch, 直接 setattr)
    """
    import os

    from app.agents import llm_router as lr_mod
    from app.rag import bm25_store as bm_mod
    from app.rag import embedding as emb_mod
    from app.rag import reranker as rk_mod
    from app.rag import vector_store as vs_mod

    os.environ["UNIKB_FAKE_EMBEDDING"] = "1"
    emb_mod.get_embedding_service.cache_clear()
    # 把真实 EmbeddingService 类换成 fake, 防止 get_embedding_service 因为
    # lru_cache 复用旧实例.
    if monkeypatch is not None:
        monkeypatch.setattr(emb_mod, "EmbeddingService", FakeEmbeddingService)
        monkeypatch.setattr(vs_mod, "ChromaStore", FakeVectorStore)
        monkeypatch.setattr(bm_mod, "BM25Store", FakeBM25Store)
        monkeypatch.setattr(rk_mod, "CrossEncoderReranker", FakeReranker)
        monkeypatch.setattr(lr_mod, "get_llm", lambda: FakeLLM())
    else:
        emb_mod.EmbeddingService = FakeEmbeddingService
        vs_mod.ChromaStore = FakeVectorStore
        bm_mod.BM25Store = FakeBM25Store
        rk_mod.CrossEncoderReranker = FakeReranker
        lr_mod.get_llm.cache_clear()
        lr_mod.get_llm = lambda: FakeLLM()
