"""P1-7 测试: BM25 惰性重建.

覆盖:
1. add() 后立即 query, 应当能查到 (lazy build 触发).
2. 多次连续 add() 只在第一次 query 时 rebuild 一次.
3. 没有 docs 时 query 返回 [], 不抛.
4. dirty 状态: add() 后 _bm25 仍是 None (没真重建).
5. delete() 也走 dirty 路径, 再次 query 才重建.
6. 并发 query 也不会触发多次重建 (锁保证单次).
7. 显式 rebuild(force=True) 立即重建.
8. 持久化 + load 后 _dirty=True, 首次 query 触发重建.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest


@pytest.fixture
def store(monkeypatch, tmp_path):
    """每次给一个全新的 BM25Store (tmp_path 持久化目录)."""
    monkeypatch.chdir(tmp_path)
    from app.rag.bm25_store import BM25Store
    return BM25Store(persist_path=str(tmp_path / "bm25.pkl"))


def _add(store, ids, docs):
    metas = [{"i": i} for i in range(len(docs))]
    store.add(ids, docs, metas)


def test_add_then_query_triggers_lazy_build(store):
    """add() 不重建; 第一次 query 才触发."""
    assert store._bm25 is None
    _add(store, ["a"], ["你好世界 hello"])
    # dirty 已置, 但 _bm25 仍是 None
    assert store._dirty is True
    assert store._bm25 is None
    hits = store.query("你好")
    assert len(hits) == 1
    assert hits[0]["id"] == "a"
    # 重建后 dirty=False
    assert store._dirty is False
    assert store._bm25 is not None


def test_multiple_adds_only_rebuild_once(store):
    """连续 3 次 add(), 只在第一次 query 时重建一次."""
    _add(store, ["a"], ["foo bar"])
    _add(store, ["b"], ["baz qux"])
    _add(store, ["c"], ["hello world"])
    assert store._bm25 is None  # 还没 query
    assert store.count() == 3
    hits = store.query("foo")
    assert len(hits) >= 1
    # 一次 query 之后, _bm25 应该是被 build 过的对象
    assert store._bm25 is not None
    # 再 query 一次, 不应再次 build (dirty=False)
    bm25_before = store._bm25
    hits2 = store.query("hello")
    assert store._bm25 is bm25_before


def test_query_with_no_docs_returns_empty():
    """docs 为空时不抛, 直接返回 []."""
    from app.rag.bm25_store import BM25Store
    s = BM25Store()
    assert s.query("anything") == []
    assert s.count() == 0


def test_dirty_flag_set_after_add(store):
    """add() 必定把 dirty 置 True."""
    _add(store, ["x"], ["test"])
    assert store._dirty is True


def test_dirty_flag_reset_after_query(store):
    """query 之后 dirty 应该清掉."""
    _add(store, ["x"], ["test"])
    store.query("test")
    assert store._dirty is False


def test_delete_marks_dirty_and_triggers_rebuild(store):
    """delete() 也走 lazy 路径."""
    _add(store, ["a", "b"], ["alpha beta", "gamma delta"])
    # 先 query 一次建立索引
    store.query("alpha")
    assert store._bm25 is not None
    assert store._dirty is False
    # 删除 -> dirty
    n = store.delete(["a"])
    assert n == 1
    assert store.count() == 1
    assert store._dirty is True
    # 立即 query 会重建
    hits = store.query("alpha")
    assert all(h["id"] != "a" for h in hits)
    assert store._dirty is False


def test_delete_nonexistent_returns_zero(store):
    _add(store, ["a"], ["x"])
    store.query("x")
    n = store.delete(["zzz"])
    assert n == 0


def test_concurrent_queries_only_rebuild_once(store):
    """并发 query 时, 重建锁保证只 build 一次."""
    _add(store, [f"d{i}" for i in range(50)], [f"term{i}" for i in range(50)])
    assert store._bm25 is None  # dirty
    results: list = []
    errors: list = []

    def worker():
        try:
            results.append(store.query("term1"))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(results) == 8
    assert store._bm25 is not None
    # 所有 query 都返回同样的排序结果 (因为只 build 一次)
    first = [r[0]["id"] for r in results if r]
    assert len(set(first)) == 1  # 全部一致


def test_rebuild_force(store):
    """显式 rebuild(force=True) 立即重建."""
    _add(store, ["a"], ["x"])
    # 还没 query, _bm25 None
    assert store._bm25 is None
    store.rebuild(force=True)
    assert store._bm25 is not None
    assert store._dirty is False


def test_rebuild_no_force_clears_dirty(store):
    """不强制 rebuild 也会清掉 dirty 标记 (因为已经 build 了)."""
    _add(store, ["a"], ["x"])
    store.rebuild(force=False)
    assert store._dirty is False


def test_persist_and_reload_resets_dirty(store, tmp_path):
    """持久化 + load 后 _dirty=True, 首次 query 才 build."""
    _add(store, ["a", "b"], ["foo bar", "baz qux"])
    # 第一次 query 触发 build
    store.query("foo")
    # 现在 _bm25 应该存在, 写盘也保留了 docs
    assert Path(store.persist_path).exists()

    # 重新实例化, load 持久化
    from app.rag.bm25_store import BM25Store
    s2 = BM25Store(persist_path=store.persist_path)
    s2.load()
    assert s2.count() == 2
    # load 之后 dirty 应当为 True (设计上, 启动时按需重建)
    assert s2._dirty is True
    assert s2._bm25 is None
    hits = s2.query("foo")
    assert len(hits) >= 1
    assert s2._dirty is False
    assert s2._bm25 is not None


def test_query_keeps_returning_after_rebuild(store):
    """多次 rebuild + query 不影响结果稳定性."""
    for round_i in range(3):
        _add(store, [f"r{round_i}"], [f"text {round_i}"])
        hits = store.query("text")
        ids = {h["id"] for h in hits}
        assert f"r{round_i}" in ids