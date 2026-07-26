"""MCP server: exposes UniKB internal tools via Model Context Protocol.
支持 stdio 传输，方便 Claude Desktop / Cursor / Trae 等 MCP 客户端直接接入。
"""
from __future__ import annotations

from loguru import logger

def _register_tool(server, tool):
    """将单个 LangChain tool 注册为 MCP tool，避免循环闭包捕获最后一个工具。"""
    @server.tool(name=tool.name, description=tool.description)
    def tool_wrapper(*args, **kwargs):
        try:
            return tool.invoke(kwargs or (args[0] if args else {}))
        except Exception as e:
            logger.exception(f"MCP tool {tool.name} failed: {e}")
            return f"Error: {e}"

    return tool_wrapper


def build_mcp_server(kb_id: str = "default"):
    """构建一个 MCP server 实例，工具集 = 内部 hybrid_search / calculator / current_date。"""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        logger.error("mcp not installed. `pip install mcp`")
        raise

    # 延迟导入，保证只使用工具适配函数时不要求安装完整 Agent 依赖。
    from app.agents.tools import build_tools

    server = FastMCP("UniKB")

    tools = build_tools(kb_id)

    for tool in tools:
        _register_tool(server, tool)

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
