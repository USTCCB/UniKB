"""测试 4: MCP 工具注册表."""
import pytest

from app.mcp.registry import ToolRegistry


def test_registry_register_and_list():
    reg = ToolRegistry()

    @reg.tool(name="echo", description="回显输入")
    def echo(text: str) -> str:
        return text

    tools = reg.list_tools()
    assert any(t.name == "echo" for t in tools)


def test_registry_invoke_returns_function_result():
    reg = ToolRegistry()

    @reg.tool(name="add", description="两数相加")
    def add(a: int, b: int) -> int:
        return a + b

    result = reg.invoke("add", {"a": 2, "b": 3})
    assert result == 5


def test_registry_unknown_tool_raises():
    reg = ToolRegistry()
    with pytest.raises(KeyError):
        reg.invoke("nope", {})