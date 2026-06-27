"""MCP 协议适配：把内部 tools 暴露为 MCP server 工具。"""
from app.mcp.server import build_mcp_server

__all__ = ["build_mcp_server"]
