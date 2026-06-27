"""MCP server: exposes UniKB internal tools via Model Context Protocol.
支持 stdio 传输，方便 Claude Desktop / Cursor / Trae 等 MCP 客户端直接接入。
"""
from __future__ import annotations

from loguru import logger

from app.agents.tools import build_tools


def build_mcp_server(kb_id: str = "default"):
    """构建一个 MCP server 实例，工具集 = 内部 hybrid_search / calculator / current_date。"""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        logger.error("mcp not installed. `pip install mcp`")
        raise

    server = FastMCP("UniKB")

    tools = build_tools(kb_id)

    for t in tools:
        # 把 LangChain tool 适配为 MCP tool
        @server.tool(name=t.name, description=t.description)
        def _tool_wrapper(*args, **kwargs):
            try:
                return t.invoke(kwargs or (args[0] if args else {}))
            except Exception as e:
                logger.exception(f"MCP tool {t.name} failed: {e}")
                return f"Error: {e}"

    return server


def main():
    """stdio 入口：python -m app.mcp.server"""
    import sys
    kb_id = sys.argv[1] if len(sys.argv) > 1 else "default"
    logger.info(f"Starting UniKB MCP server (stdio) for kb={kb_id}")
    server = build_mcp_server(kb_id)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
