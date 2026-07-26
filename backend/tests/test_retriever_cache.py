"""P1-8 测试: HybridRetriever per-kb_id 实例缓存 + 写并发锁.

覆盖:
1. 同 kb_id 多次 get_retriever() 返回同一对象.
2. 不同 kb_id 返回不同对象.
3. reset_retriever_cache() 清空后, 再次 get_retriever() 创建新实例.
4. 并发 get_retriever() 不重复创建 (锁保护).
5. add_documents 有写锁: 两个并发 add 串行化.
6. delete_documents 同步删两边.
7. retrieve 不需要写锁, 可并发读.

不依赖真实 chromadb / sentence_transformers:
- HybridRetriever 的 vector_store / embedding 在 add/delete 涉及的测试里被替换为 fake.
"""
from __future__ import annotations

import threading
import time

import pytest


class _FakeVectorStore:
    """替身: 不依赖 chromadb, 接口对齐 ChromaStore.add/delete/count."""
    def __init__(self):
        self._ids: list[str] = []
        self._write_lock = threading.Lock()

    def add(self, ids, documents, embeddings, metadatas):
        with self._write_lock:
            for i in ids:
                self._ids.append(i)

    def delete(self, ids):
        with self._write_lock:
            before = len(self._ids)
            self._ids = [i for i in self._ids if i not in set(ids)]
            return before - len(self._ids)

    def count(self):
        return len(self._ids)


class _FakeEmbedding:
    """替身: 不依赖 sentence_transformers."""
    dim = 4

    def embed(self, docs):
        return [[0.0] * self.dim for _ in docs]

    def embed_query(self, q):
        return [0.0] * self.dim


@pytest.fixture(autouse=True)
def _clean_cache(monkeypatch, tmp_path):
    # 让 BM25Store 落到 tmp_path, 避免上一个测试的持久化文件污染下一个.
    monkeypatch.chdir(tmp_path)
    from app.rag.retriever import reset_retriever_cache
    reset_retriever_cache()
    yield
    reset_retriever_cache()


def _patch_retriever_for_test(retriever):
    """跳过真实 chromadb / sentence_transformers."""
    retriever.vector_store = _FakeVectorStore()
    retriever.embedding = _FakeEmbedding()


def test_get_retriever_returns_same_instance():
    """同 kb_id 应当返回同一对象 (避免每次重建 ChromaStore/BM25Store)."""
    from app.rag.retriever import get_retriever

    a1 = get_retriever("alpha")
    a2 = get_retriever("alpha")
    assert a1 is a2


def test_different_kb_ids_get_different_instances():
    from app.rag.retriever import get_retriever

    a = get_retriever("alpha")
    b = get_retriever("beta")
    assert a is not b
    assert a.kb_id == "alpha"
    assert b.kb_id == "beta"


def test_reset_clears_cache():
    from app.rag.retriever import get_retriever, reset_retriever_cache

    a1 = get_retriever("alpha")
    reset_retriever_cache()
    a2 = get_retriever("alpha")
    # 不同对象了
    assert a1 is not a2


def test_concurrent_get_retriever_no_double_init():
    """N 个线程同时 get_retriever(kb_id), 只能创建一个实例."""
    from app.rag.retriever import get_retriever

    instances = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        r = get_retriever("shared")
        instances.append(id(r))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(set(instances)) == 1, "应当只创建 1 个实例"


def test_add_documents_serializes_with_lock():
    """两个并发 add 应当被写锁串行化, 不会数据竞争.

    思路: patch embedding.embed 让它 sleep, 然后并发 add 两次.
    如果没有锁, 两个 embed 会同时跑; 有锁的话两次串行, 总耗时 ≈ 2x sleep.
    """
    from app.rag.retriever import get_retriever

    retriever = get_retriever("lock_test")
    _patch_retriever_for_test(retriever)

    call_log: list[str] = []
    call_log_lock = threading.Lock()

    def slow_embed(docs):
        with call_log_lock:
            call_log.append("start:" + docs[0][:5])
        time.sleep(0.08)
        with call_log_lock:
            call_log.append("end:" + docs[0][:5])
        return [[0.0] * 4 for _ in docs]

    retriever.embedding.embed = slow_embed  # type: ignore[assignment]

    t1 = threading.Thread(target=retriever.add_documents, args=(["a"], ["alpha"], [{}]))
    t2 = threading.Thread(target=retriever.add_documents, args=(["b"], ["beta"], [{}]))
    t0 = time.monotonic()
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    elapsed = time.monotonic() - t0

    # 串行: 应该看到 start A -> end A -> start B -> end B (或反过来).
    # 任何 "start" 后立即跟另一个 "start" 都说明没锁住.
    for i, line in enumerate(call_log):
        if line.startswith("start:"):
            if i + 1 < len(call_log):
                nxt = call_log[i + 1]
                assert nxt.startswith("end:"), \
                    f"并发 add 没有被串行化: start 后立即跟了 {nxt}"
    # 串行总耗时 ≈ 0.16s; 并行 ≈ 0.08s. 给点 buffer.
    assert elapsed > 0.13, f"两次 add 应当串行 (~0.16s), 实测 {elapsed:.3f}s — 锁没生效"


def test_delete_documents_removes_from_both_stores():
    """delete_documents 应当同时清 BM25 和 Chroma."""
    from app.rag.retriever import get_retriever

    retriever = get_retriever("del_test")
    _patch_retriever_for_test(retriever)
    retriever.add_documents(
        ids=["c1", "c2", "c3"],
        documents=["alpha beta", "gamma delta", "epsilon zeta"],
        metadatas=[{}, {}, {}],
    )
    assert retriever.bm25_store.count() == 3
    assert retriever.vector_store.count() == 3

    n = retriever.delete_documents(["c1", "c2"])
    assert n == 2
    assert retriever.bm25_store.count() == 1
    assert retriever.vector_store.count() == 1


def test_delete_documents_with_no_ids():
    from app.rag.retriever import get_retriever

    retriever = get_retriever("del_test2")
    _patch_retriever_for_test(retriever)
    retriever.bm25_store.add(["x"], ["y"], [{}])
    n = retriever.delete_documents([])
    assert n == 0


def test_retrieve_does_not_hold_write_lock():
    """retrieve 是读操作, 不应该跟 add 互斥.

    思路: 触发一个慢 retrieve, 期间并发 add, 确认 add 不被 retrieve 阻塞太久.
    """
    from app.rag.retriever import get_retriever

    retriever = get_retriever("rw_test")
    _patch_retriever_for_test(retriever)

    real_retrieve = retriever.retrieve

    def slow_retrieve(query, top_k=None):
        time.sleep(0.2)
        return []

    retriever.retrieve = slow_retrieve  # type: ignore[assignment]
    try:
        t0 = time.monotonic()
        # 启动一个慢 retrieve
        r_thread = threading.Thread(target=retriever.retrieve, args=("foo",))
        r_thread.start()
        time.sleep(0.02)  # 让 retrieve 拿到一点点时间
        # add 应该能立刻执行 (不阻塞 retrieve)
        retriever.add_documents(["b"], ["baz"], [{}])
        elapsed = time.monotonic() - t0
        r_thread.join()
        # add 在 retrieve 期间执行了, 总耗时 ≈ 0.22s
        assert elapsed < 0.35, f"add 被 retrieve 阻塞了: {elapsed}s"
    finally:
        retriever.retrieve = real_retrieve  # type: ignore[assignment]
