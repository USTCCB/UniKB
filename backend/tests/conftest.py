"""共享 fixtures: 把 app 加入 sys.path 让 import 跑得通."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 让 `from app.xxx import ...` 跑得通, 不依赖 src layout
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def settings():
    """从 .env.example 强制加载一份测试用 settings."""
    from app.core.config import Settings

    return Settings(_env_file=ROOT / ".env.example")