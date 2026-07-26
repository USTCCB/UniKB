"""MCP 工具适配测试：不依赖真实 MCP 服务端或模型。"""
from __future__ import annotations


class FakeServer:
    def __init__(self):
        self.registered = {}

    def tool(self, *, name, description):
        def decorator(fn):
            self.registered[name] = {"description": description, "handler": fn}
            return fn
        return decorator


class FakeTool:
    def __init__(self, name):
        self.name = name
        self.description = f"{name} description"
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        return {"tool": self.name, "payload": payload}


def test_register_tool_keeps_each_tool_binding():
    from app.mcp.server import _register_tool

    server = FakeServer()
    search = FakeTool("hybrid_search")
    calculator = FakeTool("calculator")
    _register_tool(server, search)
    _register_tool(server, calculator)

    assert server.registered["hybrid_search"]["handler"](query="RAG") == {
        "tool": "hybrid_search", "payload": {"query": "RAG"},
    }
    assert server.registered["calculator"]["handler"](expression="1+1") == {
        "tool": "calculator", "payload": {"expression": "1+1"},
    }
    assert search.calls == [{"query": "RAG"}]
    assert calculator.calls == [{"expression": "1+1"}]


def test_register_tool_returns_error_message_on_failure():
    from app.mcp.server import _register_tool

    class BrokenTool(FakeTool):
        def invoke(self, payload):
            raise RuntimeError("boom")

    server = FakeServer()
    _register_tool(server, BrokenTool("broken"))
    assert server.registered["broken"]["handler"](value=1) == "Error: boom"
