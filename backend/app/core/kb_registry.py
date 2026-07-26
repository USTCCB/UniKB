"""知识库注册中心: 记录每个 kb_id 的 owner 与是否公共, 解决 IDOR 越权访问.

设计:
1. 数据结构极简: 持久化到 ./data/kb_registry.json, 一个 {kb_id: {owner, is_public, created_at}}.
2. 默认有一个公共 kb "default", 所有登录用户只读可访问.
3. 写操作 (upload) 会保证 kb 存在并属于当前用户; 若 kb 已被别人占用则返回 403.
4. 读操作 (list / chat) 对白名单公共 kb 直接放行; 对私有 kb 必须 owner == 当前用户.
5. 并发保护: 单进程内用 threading.Lock 串行化; 多进程写同一文件的最坏情况由
   操作系统原子写 (os.replace) 保证不会损坏文件, 偶发的覆盖竞争在 demo 场景可接受.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, status

from app.core.config import get_settings, settings
from app.core.logging import logger

# 注册表文件路径 (同 Chroma / BM25 等持久化目录一致)
_REGISTRY_PATH = Path("./data/kb_registry.json")
_LOCK = threading.Lock()


class KnowledgeBase:
    """单个知识库的元信息."""

    __slots__ = ("kb_id", "owner", "is_public", "created_at")

    def __init__(
        self,
        kb_id: str,
        owner: str,
        is_public: bool = False,
        created_at: Optional[float] = None,
    ):
        self.kb_id = kb_id
        self.owner = owner
        self.is_public = is_public
        self.created_at = created_at or time.time()

    def to_dict(self) -> dict:
        return {
            "owner": self.owner,
            "is_public": self.is_public,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, kb_id: str, data: dict) -> "KnowledgeBase":
        return cls(
            kb_id=kb_id,
            owner=data.get("owner", ""),
            is_public=bool(data.get("is_public", False)),
            created_at=float(data.get("created_at", 0.0)) or None,
        )


def _load_unlocked() -> dict[str, dict]:
    if not _REGISTRY_PATH.exists():
        return {}
    try:
        return json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"kb_registry.json 解析失败, 将重新初始化: {e}")
        return {}


def _save_unlocked(data: dict[str, dict]) -> None:
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _REGISTRY_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, _REGISTRY_PATH)


def _load_all() -> dict[str, dict]:
    """线程安全的读."""
    with _LOCK:
        return _load_unlocked()


def _save_all(data: dict[str, dict]) -> None:
    """线程安全的写 (atomic replace)."""
    with _LOCK:
        _save_unlocked(data)


def _public_kb_ids() -> set[str]:
    """从 settings 读公共 kb 白名单.

    接受两种格式:
    - 逗号分隔: "default,shared,public-docs"
    - JSON 数组: '["default","shared"]'
    默认 fallback 到 ["default"].

    注意: 通过 get_settings() 而非模块级 settings, 避免 lru_cache 复用导致
    测试改 env 不生效.
    """
    cfg = get_settings()
    raw = getattr(cfg, "public_kb_ids", None) or "default"
    items: list[str] = []
    if isinstance(raw, list):
        items = [str(x).strip() for x in raw if str(x).strip()]
    elif isinstance(raw, str):
        s = raw.strip()
        if not s:
            items = []
        elif s.startswith("["):
            try:
                import json as _json
                parsed = _json.loads(s)
                if isinstance(parsed, list):
                    items = [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                items = []
        if not items:
            items = [x.strip() for x in s.split(",") if x.strip()]
    return set(items or ["default"])


def is_public_kb(kb_id: str) -> bool:
    """判断某个 kb 是否在公共白名单 (无论是否在 registry 中登记)."""
    return kb_id in _public_kb_ids()


def get_kb(kb_id: str) -> Optional[KnowledgeBase]:
    data = _load_all()
    raw = data.get(kb_id)
    if not raw:
        return None
    return KnowledgeBase.from_dict(kb_id, raw)


def ensure_kb_for_read(kb_id: str, user: str) -> KnowledgeBase:
    """读权限校验: 公共 kb 直接放行; 私有 kb 要求 owner==user.

    Raises:
        404: kb 不存在 (私有 kb 用户无法感知其存在, 视为不存在)
        403: kb 存在但不属于当前用户
    """
    # 1) 公共 kb 白名单: 直接放行, 不需要事先注册
    if is_public_kb(kb_id):
        kb = get_kb(kb_id)
        if kb is None:
            # 公共 kb 没有显式注册过, 视为 owner="" (系统) 即可
            return KnowledgeBase(kb_id=kb_id, owner="", is_public=True)
        return kb

    # 2) 私有 kb: 必须存在且 owner 匹配
    kb = get_kb(kb_id)
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"知识库不存在: {kb_id}",
        )
    if kb.owner != user:
        # 不区分 "不存在" 与 "无权访问", 避免泄露 kb 是否存在
        logger.warning(f"KB access denied: user={user} tried kb_id={kb_id} owned by={kb.owner}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权访问该知识库",
        )
    return kb


def ensure_kb_for_write(kb_id: str, user: str) -> KnowledgeBase:
    """写权限校验: 若 kb 不存在则自动创建并归属当前用户; 若存在则必须 owner==user.

    Raises:
        403: kb 存在但属于其他用户 (包括公共 kb: 不允许往默认公共 kb 写)
    """
    public_set = _public_kb_ids()
    # 公共 kb 不允许普通用户写入 (保持只读); 如果业务需要可再放开.
    if kb_id in public_set:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"公共知识库 '{kb_id}' 不可写入, 请上传到自己的知识库",
        )

    with _LOCK:
        data = _load_unlocked()
        raw = data.get(kb_id)
        if raw is None:
            kb = KnowledgeBase(kb_id=kb_id, owner=user, is_public=False)
            data[kb_id] = kb.to_dict()
            _save_unlocked(data)
            logger.info(f"Created new KB: kb_id={kb_id} owner={user}")
            return kb
        kb = KnowledgeBase.from_dict(kb_id, raw)
        if kb.owner != user:
            logger.warning(
                f"KB write denied: user={user} tried to write kb_id={kb_id} owned by={kb.owner}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="无权访问该知识库",
            )
        return kb


def reset_for_tests() -> None:
    """仅供测试使用: 清空注册表, 让每个测试隔离."""
    with _LOCK:
        if _REGISTRY_PATH.exists():
            _REGISTRY_PATH.unlink()