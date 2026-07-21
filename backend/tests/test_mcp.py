"""测试 4: MCP server 构建 + tool 注册.

注意: app.mcp.server 依赖 `from mcp.server.fastmcp import FastMCP`,
但 mcp 1.1.0 还没有 fastmcp 子模块 (1.2+ 才有). 在 CI 上 ImportError 就 skip.
"""
from __future__ import annotations

import pytest

# 把整个模块跳过如果 mcp.server.fastmcp 不存在
fastmcp = pytest.importorskip("mcp.server.fastmcp")


def test_build_mcp_server_returns_fastmcp_instance():
    from app.mcp.server import build_mcp_server

    server = build_mcp_server(kb_id="default")
    assert isinstance(server, fastmcp.FastMCP)


def test_build_tools_returns_langchain_tools():
    from app.agents.tools import build_tools

    tools = build_tools(kb_id="default")
    names = {t.name for t in tools}
    assert "hybrid_search" in names


def test_mcp_server_wrapper_is_callable():
    from app.mcp.server import build_mcp_server

    server = build_mcp_server(kb_id="default")
    # FastMCP 实例至少有 .tool 装饰器
    assert callable(getattr(server, "tool", None)) or hasattr(server, "_tool_manager")