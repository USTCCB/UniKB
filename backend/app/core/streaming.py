"""把同步阻塞迭代器转换为 async 生成器, 不阻塞事件循环.

用法:
    async for x in sync_iter_to_async(blocking_iter()):
        ...

实现: 把同步迭代器搬进默认 threadpool, 通过 asyncio.Queue 跨线程传递.
每个 next() 都 spawn 一次性 task 拉下一个值, 不阻塞主事件循环.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Iterator, TypeVar

T = TypeVar("T")


async def sync_iter_to_async(it: Iterator[T]) -> AsyncIterator[T]:
    """在 threadpool 里消费同步迭代器, 不阻塞事件循环.

    适合: langchain ChatOpenAI / DeepseekChat 等同步 SDK 的 .stream() 调用,
    这些 SDK 没有原生 async stream, 直接 for-loop 会阻塞 FastAPI 事件循环.
    """
    loop = asyncio.get_running_loop()

    def _next_in_thread(queue: asyncio.Queue):
        try:
            item = next(it)
            loop.call_soon_threadsafe(queue.put_nowait, ("item", item))
        except StopIteration:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", e))

    queue: asyncio.Queue = asyncio.Queue(maxsize=128)
    # 启动首次 next() 在后台线程
    asyncio.get_running_loop().run_in_executor(None, _next_in_thread, queue)

    while True:
        kind, payload = await queue.get()
        if kind == "item":
            yield payload
            asyncio.get_running_loop().run_in_executor(None, _next_in_thread, queue)
        elif kind == "done":
            return
        else:  # error
            raise payload
