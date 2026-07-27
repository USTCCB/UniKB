"""测试 CrossEncoderReranker 的单例缓存.

要求 (来自 P1-3):
1. 连续两次调用 `get_reranker()` 返回同一对象 (is 判断).
2. 不同 model_name 产生不同实例.
3. 底层 CrossEncoder 只被实例化一次 (mock `CrossEncoder.__init__` 计数).
4. 业务代码 (`pipeline.py` / `chat.py` / `tools.py`) 都不再 `CrossEncoderReranker()`.
"""
from __future__ import annotations

import sys
import types

import pytest


@pytest.fixture
def fake_cross_encoder(monkeypatch):
    """把 sentence_transformers.CrossEncoder 替换成一个可计数的 fake."""
    created = {"count": 0}

    class _FakeCE:
        def __init__(self, model_name):
            created["count"] += 1
            self.model_name = model_name

        def predict(self, pairs, **_kwargs):
            # 默认给相同分数, 不影响排序断言.
            return [0.5 for _ in pairs]

    fake_mod = types.ModuleType("sentence_transformers")
    fake_mod.CrossEncoder = _FakeCE  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)
    # 清掉单例缓存, 让测试在干净状态跑.
    from app.rag import reranker as rk_mod

    rk_mod.reset_reranker_cache()
    yield created, _FakeCE
    rk_mod.reset_reranker_cache()


def test_get_reranker_returns_same_instance(fake_cross_encoder):
    """get_reranker() 同一 model_name 必须返回同一对象."""
    from app.rag.reranker import get_reranker

    a = get_reranker()
    b = get_reranker()
    assert a is b, "连续两次 get_reranker() 必须返回同一实例"


def test_get_reranker_loads_model_only_once(fake_cross_encoder):
    """底层 CrossEncoder 构造只发生一次, 即使 rerank 被调多次."""
    from app.rag.reranker import get_reranker

    count, _ = fake_cross_encoder
    rk = get_reranker()
    # 调几次 rerank, 底层 CrossEncoder 只构造一次 (懒加载 + 缓存).
    for _ in range(3):
        rk.rerank("query", [{"document": "x"}, {"document": "y"}], top_k=2)
    # 多次 get_reranker() 也不应重新构造.
    for _ in range(3):
        get_reranker().rerank("query", [{"document": "x"}], top_k=1)
    assert count["count"] == 1, f"底层 CrossEncoder 应只构造 1 次, 实际 {count['count']}"


def test_get_reranker_distinct_models(fake_cross_encoder):
    """不同 model_name 走不同实例 (按 model_name 分桶缓存)."""
    from app.rag.reranker import get_reranker

    a = get_reranker("model-A")
    b = get_reranker("model-A")
    c = get_reranker("model-B")
    assert a is b, "同一 model_name 应返回同一实例"
    assert a is not c, "不同 model_name 应返回不同实例"


def test_business_code_uses_singleton(monkeypatch):
    """pipeline.py / chat.py / tools.py 都用 get_reranker() 而不是 CrossEncoderReranker().

    我们静态扫描这几个文件, 不允许出现 `CrossEncoderReranker()` 这种直接实例化.
    """
    targets = [
        "app/rag/pipeline.py",
        "app/api/chat.py",
        "app/agents/tools.py",
    ]
    for path in targets:
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
        # 允许 import / 类型注解出现 `CrossEncoderReranker`, 但不允许直接调用.
        # 简单做法: 排除 `CrossEncoderReranker` 后面跟 `(` 但不是方法调用的.
        # 这里只允许 `get_reranker()` 和 (debug 用) `reset_reranker_cache`.
        bad = []
        for line in src.splitlines():
            # 跳过注释与 import 行.
            stripped = line.strip()
            if stripped.startswith(("#", "from ")):
                continue
            # 形式 `CrossEncoderReranker(...)` -- 直接实例化.
            if "CrossEncoderReranker(" in line:
                bad.append(line)
        assert not bad, (
            f"{path} 里仍然直接 CrossEncoderReranker() 实例化, 应该改成 get_reranker():\n"
            + "\n".join(bad)
        )


def test_get_reranker_can_be_reset(fake_cross_encoder):
    """reset_reranker_cache 后, 下一次 get_reranker 会重建对象."""
    from app.rag.reranker import get_reranker, reset_reranker_cache

    a = get_reranker()
    a.rerank("query", [{"document": "x"}], top_k=1)  # 触发底层模型加载
    count, _ = fake_cross_encoder
    assert count["count"] == 1

    reset_reranker_cache()
    b = get_reranker()
    b.rerank("query", [{"document": "x"}], top_k=1)  # 触发新对象底层模型加载
    assert a is not b
    assert count["count"] == 2, "reset 后重新构造一次底层 CrossEncoder"