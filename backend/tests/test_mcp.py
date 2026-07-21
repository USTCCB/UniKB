"""测试 4: MCP server 构建 + tool 注册."""
from __future__ import annotations


def test_build_mcp_server_returns_fastmcp_instance():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        import pytest

        pytest.skip("mcp not installed")

    from app.mcp.server import build_mcp_server

    server = build_mcp_server(kb_id="default")
    assert isinstance(server, FastMCP)


def test_build_tools_returns_langchain_tools():
    from app.agents.tools import build_tools

    tools = build_tools(kb_id="default")
    names = {t.name for t in tools}
    # 至少要有 hybrid_search
    assert "hybrid_search" in names


def test_mcp_server_wrapper_is_callable():
    """Sanity: wrapper 函数可以被调用而不抛异常 (tool 实际执行可能失败, 但 build 不应爆)."""
    try:
        from app.mcp.server import build_mcp_server
    except ImportError:
        import pytest

        pytest.skip("mcp not installed")

    server = build_mcp_server(kb_id="default")
    # FastMCP 实例有 list_tools / run 等方法
    assert hasattr(server, "list_tools") or hasattr(server, "_tool_manager")