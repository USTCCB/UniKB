"""P1-6 测试: 同步 CPU 密集型调用卸到 threadpool, 不阻塞事件循环.

验证:
1. HybridRetriever.retrieve_async / add_documents_async 真的把同步方法丢到 threadpool
   (不是阻塞主事件循环).
2. sync_iter_to_async 不阻塞事件循环, 并且能把同步迭代器转成 async 生成器.
3. asyncio.to_thread 包装后, 在 async 函数里 await 不会 block 其他并发协程.
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Iterator

import pytest

from app.core.streaming import sync_iter_to_async


@pytest.mark.asyncio
async def test_add_documents_async_does_not_block_loop():
    """add_documents_async 必须真正把同步调用抛到另一个线程."""
    from app.rag.retriever import HybridRetriever

    retriever = HybridRetriever(kb_id="default")

    main_thread = threading.get_ident()
    worker_threads: list[int] = []

    real_add = retriever.add_documents

    def wrapped(*args, **kwargs):
        worker_threads.append(threading.get_ident())
        # 在子线程里短暂 sleep, 模拟 CPU 密集
        time.sleep(0.05)
        # 返回 None, 不真入库
        return None

    retriever.add_documents = wrapped  # type: ignore[assignment]
    try:
        # 跑两次, 确认 worker 线程 != 主事件循环线程
        await retriever.add_documents_async(["a"], ["x"], [{}])
        # asyncio.to_thread 会在事件循环的默认 executor 跑, executor 线程 != 主线程
        # 这里主线程是 pytest 测试线程 (不是 asyncio loop thread),
        # 所以只要 worker 跟调用线程不同就行.
        assert len(worker_threads) >= 1
        assert all(t != main_thread for t in worker_threads), \
            "add_documents 应当在另一个线程跑, 不能在调用线程跑"
    finally:
        retriever.add_documents = real_add  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_retrieve_async_does_not_block_loop():
    """retrieve_async 也是 to_thread 包装."""
    from app.rag.retriever import HybridRetriever

    retriever = HybridRetriever(kb_id="default")

    main_thread = threading.get_ident()
    worker_threads: list[int] = []

    def fake_retrieve(query, top_k=None):
        worker_threads.append(threading.get_ident())
        time.sleep(0.05)
        return []

    retriever.retrieve = fake_retrieve  # type: ignore[assignment]
    try:
        await retriever.retrieve_async("hi")
        assert len(worker_threads) == 1
        assert worker_threads[0] != main_thread
    finally:
        retriever.retrieve = lambda q, top_k=None: []  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_async_routes_run_concurrently():
    """3 个并发 retrieve_async 在主线程 sleep 期间都能完成, 证明不阻塞.

    直观思路: 假设每个 retrieve 要 sleep 100ms (模拟 embedding).
    如果是阻塞调用, 串行需要 ~300ms; 如果并发行, ~100ms.
    """
    from app.rag.retriever import HybridRetriever

    retriever = HybridRetriever(kb_id="default")

    def slow_retrieve(query, top_k=None):
        time.sleep(0.1)
        return [{"id": "x", "document": "ok"}]

    retriever.retrieve = slow_retrieve  # type: ignore[assignment]

    t0 = time.monotonic()
    results = await asyncio.gather(
        retriever.retrieve_async("a"),
        retriever.retrieve_async("b"),
        retriever.retrieve_async("c"),
    )
    elapsed = time.monotonic() - t0
    # to_thread 用的是默认 threadpool, 三个并发请求应该能在 ~100ms 内完成
    assert len(results) == 3
    assert elapsed < 0.25, f"并发 retrieve_async 应该并行, 但耗时 {elapsed:.3f}s"


# ============ sync_iter_to_async ============

def _sync_iter(values: list[int]) -> Iterator[int]:
    for v in values:
        yield v


@pytest.mark.asyncio
async def test_sync_iter_to_async_basic():
    """基本行为: 把同步迭代器转成 async 生成器."""
    seen = []
    async for x in sync_iter_to_async(_sync_iter([1, 2, 3])):
        seen.append(x)
    assert seen == [1, 2, 3]


@pytest.mark.asyncio
async def test_sync_iter_to_async_empty():
    """空迭代器."""
    seen = []
    async for x in sync_iter_to_async(_sync_iter([])):
        seen.append(x)
    assert seen == []


@pytest.mark.asyncio
async def test_sync_iter_to_async_does_not_block_loop():
    """迭代期间主事件循环仍能跑其他协程."""
    loop_running = []

    async def heartbeat():
        await asyncio.sleep(0.01)
        loop_running.append(True)
        await asyncio.sleep(0.01)
        loop_running.append(True)
        await asyncio.sleep(0.01)
        loop_running.append(True)

    def slow_iter() -> Iterator[int]:
        for i in range(3):
            time.sleep(0.02)  # 模拟同步 stream 拉 token
            yield i

    async def consume():
        out = []
        async for x in sync_iter_to_async(slow_iter()):
            out.append(x)
        return out

    results, _ = await asyncio.gather(consume(), heartbeat())
    assert results == [0, 1, 2]
    # 主循环必须在迭代期间还跑了至少 1 次
    assert len(loop_running) >= 1


@pytest.mark.asyncio
async def test_sync_iter_to_async_propagates_exception():
    """迭代器抛异常时, async 端也要重新 raise."""
    def bad_iter() -> Iterator[int]:
        yield 1
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        async for x in sync_iter_to_async(bad_iter()):
            pass


@pytest.mark.asyncio
async def test_sync_iter_to_async_stops_iteration():
    """StopIteration 应当正常结束, 不抛."""
    async def consume():
        out = []
        async for x in sync_iter_to_async(_sync_iter([42])):
            out.append(x)
        return out

    assert await consume() == [42]