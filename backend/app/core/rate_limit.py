"""进程内轻量限流器.

设计:
1. 不引入新三方依赖 (题目要求). 用标准库 threading + time.
2. 两个独立的桶:
   - LOGIN_FAIL_BUCKET: 同 username + IP 维度, 滑动窗口 (5min 内 5 次失败).
     超过则锁定 15min (LOGIN_LOCK_KEY 单独存解锁时间戳).
   - CHAT_TOKEN_BUCKET: 每用户 / 每 IP 固定窗口 (每分钟 N 次). 超额返回 429.
3. 存储是 dict + Lock; 测试时可调 reset() 清空.
4. 生产环境如果 Redis 已就绪, 未来可换 storage. 当前实现只覆盖单进程; 多进程
   部署会各算各的, 但 login 暴力破解只要 5 次失败就锁定, 单进程拦截足够防爆.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple

from loguru import logger

from app.core.config import get_settings


_FAIL_WINDOW: Dict[Tuple[str, str], Deque[float]] = {}
_FAIL_LOCK: Dict[Tuple[str, str], float] = {}  # 解锁时间戳
_CHAT_BUCKET: Dict[str, Deque[float]] = {}

_LOCK = threading.Lock()


def reset_for_tests() -> None:
    """仅供测试: 清空所有限流状态."""
    with _LOCK:
        _FAIL_WINDOW.clear()
        _FAIL_LOCK.clear()
        _CHAT_BUCKET.clear()


def set_enabled(enabled: bool) -> None:
    """仅供测试: 切换限流开关. 通过设 env var 让 get_settings() 重新解析.

    注意: 因为 Settings 是 pydantic BaseModel + BaseSettings, 直接 setattr 会
    触发 frozen/validated model 的报错. 这里用环境变量统一覆盖.
    """
    import os
    os.environ["RATE_LIMIT_ENABLED"] = "1" if enabled else "0"
    get_settings.cache_clear()


@dataclass
class RateLimitDecision:
    allowed: bool
    retry_after: float = 0.0
    reason: str = ""


def _now() -> float:
    return time.monotonic()


def _bucket_key(scope: str, identifier: str) -> Tuple[str, str]:
    return (scope, identifier)


def check_login_failure(username: str, ip: str) -> RateLimitDecision:
    """登录前询问: 该 (username, ip) 是否被锁定."""
    cfg = get_settings()
    if not cfg.rate_limit_enabled:
        return RateLimitDecision(allowed=True)
    with _LOCK:
        key = _bucket_key("login", f"{ip}|{username}")
        now = _now()
        # 1) 是否在锁定窗口内
        until = _FAIL_LOCK.get(key)
        if until and until > now:
            return RateLimitDecision(
                allowed=False,
                retry_after=until - now,
                reason="login locked: too many failures",
            )
        # 2) 滑动窗口计数
        dq = _FAIL_WINDOW.setdefault(key, deque())
        cutoff = now - cfg.rate_limit_login_window_sec
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= cfg.rate_limit_login_fail_max:
            # 锁定
            _FAIL_LOCK[key] = now + cfg.rate_limit_login_lock_sec
            logger.warning(f"login locked: {ip}/{username} after {len(dq)} failures")
            return RateLimitDecision(
                allowed=False,
                retry_after=cfg.rate_limit_login_lock_sec,
                reason="login locked: too many failures",
            )
    return RateLimitDecision(allowed=True)


def record_login_failure(username: str, ip: str) -> None:
    cfg = get_settings()
    if not cfg.rate_limit_enabled:
        return
    with _LOCK:
        key = _bucket_key("login", f"{ip}|{username}")
        dq = _FAIL_WINDOW.setdefault(key, deque())
        dq.append(_now())


def record_login_success(username: str, ip: str) -> None:
    """登录成功清掉该 (user, ip) 的失败计数, 不影响其他 IP/账号."""
    with _LOCK:
        key = _bucket_key("login", f"{ip}|{username}")
        _FAIL_WINDOW.pop(key, None)
        _FAIL_LOCK.pop(key, None)


def check_chat_quota(user: str, ip: str) -> RateLimitDecision:
    """chat/chat-stream 配额: 每个 user 每分钟 N 次.

    对已认证请求用 user, 未认证 (理论上 chat 都要求 auth) 退化为 IP.
    """
    cfg = get_settings()
    if not cfg.rate_limit_enabled:
        return RateLimitDecision(allowed=True)
    identifier = user or ip
    if not identifier:
        return RateLimitDecision(allowed=True)
    window = 60.0
    limit = cfg.rate_limit_chat_per_min
    with _LOCK:
        key = _bucket_key("chat", identifier)
        now = _now()
        dq = _CHAT_BUCKET.setdefault(key, deque())
        cutoff = now - window
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= limit:
            retry = window - (now - dq[0]) if dq else window
            logger.warning(f"chat rate limit hit: user={user} ip={ip}")
            return RateLimitDecision(
                allowed=False,
                retry_after=max(retry, 0.001),
                reason="chat quota exceeded",
            )
        dq.append(now)
    return RateLimitDecision(allowed=True)