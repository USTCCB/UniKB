"""测试 calculator 沙箱 (P0-2).

覆盖:
1. 正常表达式能算 (整数 / 浮点 / 一元正负 / math.* / 嵌套).
2. 经典 eval 逃逸 payload 必须被拒绝 (不执行, 直接报错).
3. 工具入口 calculator() 返回字符串, 错误情况不抛异常.
"""
from __future__ import annotations

import pytest

from app.agents.safe_eval import SafeEvalError, safe_eval


# ============ 正常表达式 ============

@pytest.mark.parametrize(
    "expr, expected",
    [
        ("1 + 2", 3),
        ("10 - 4", 6),
        ("3 * 4", 12),
        ("10 / 4", 2.5),
        ("10 // 3", 3),
        ("10 % 3", 1),
        ("2 ** 10", 1024),
        ("-5 + 10", 5),
        ("+5", 5),
        ("-(2 ** 3)", -8),
        ("(1 + 2) * 3", 9),
        ("sqrt(16)", 4.0),
        ("sqrt(16) + 2", 6.0),
        ("max(1, 2, 3)", 3),
        ("min(1, 2, 3)", 1),
        ("abs(-7)", 7),
        ("round(3.14159, 2)", 3.14),
        ("pow(2, 8)", 256.0),
        ("floor(3.7)", 3),
        ("ceil(3.2)", 4),
        ("pi", pytest.approx(3.14159265, rel=1e-6)),
        ("e", pytest.approx(2.71828, rel=1e-6)),
        ("sin(0)", 0.0),
        ("cos(0)", 1.0),
        ("log(e)", pytest.approx(1.0, rel=1e-6)),
        ("max(1, 2) + min(3, 4)", 5),
    ],
)
def test_safe_eval_accepts_normal_expressions(expr, expected):
    assert safe_eval(expr) == expected


# ============ 沙箱逃逸 payload: 必须被拒绝 ============

# 一组经典的 eval 逃逸 payload, 每个必须抛 SafeEvalError, 而不是执行.
EVIL_PAYLOADS = [
    # 1. 对象内省链: ().__class__.__bases__[0].__subclasses__()
    "().__class__.__bases__[0].__subclasses__()",
    # 2. 直接调 __import__
    "__import__('os').system('echo pwned')",
    # 3. 通过 open 读文件
    "open('/etc/passwd').read()",
    # 4. 内建 eval
    "eval('1+1')",
    # 5. 内建 exec
    "exec('print(1)')",
    # 6. globals()['eval']
    "globals()['eval']('1+1')",
    # 7. lambda
    "(lambda: 1)()",
    # 8. 列表推导
    "[x for x in range(10)]",
    # 9. f-string
    "f'{__import__(\"os\").system(\"id\")}'",
    # 10. 属性访问 (math.pi 都不行, 我们要求纯 Name)
    "math.sqrt(16)",
    # 11. 下标访问
    "a[0]",
    # 12. 字典
    "{'a': 1}",
    # 13. 比较运算符
    "1 == 1",
    # 14. 字符串
    "'hello'",
    # 15. 关键字参数
    "round(number=3.7)",
    # 16. dunder 名字 (下划线开头)
    "__class__",
    # 17. 多表达式 / 链式赋值
    "a = 1",
    # 18. import 语句 (语法错误层面就拦住)
    "import os",
    # 19. 异常处理
    "1/0",
    # 20. walrus
    "(x := 1)",
    # 21. 星号 unpack
    "[*range(10)]",
    # 22. 三元
    "1 if True else 2",
    # 23. 未知函数 (不在白名单)
    "ord('a')",
    "chr(65)",
    "id(1)",
    "type(1)",
    # 24. 位运算 (本白名单未放行)
    "1 | 2",
    "1 & 2",
    "1 << 2",
    # 25. 字符串方法
    "'abc'.upper()",
    # 26. 类型注解 / 装饰器等
    "1 .real",
]


@pytest.mark.parametrize("payload", EVIL_PAYLOADS)
def test_safe_eval_rejects_evil_payloads(payload):
    """任何在白名单外的语法必须被拒绝, 不能执行."""
    with pytest.raises(SafeEvalError):
        safe_eval(payload)


# ============ DoS / 长度限制 ============

def test_safe_eval_rejects_overlong_expression():
    """超长表达式直接拒绝, 防止巨型 AST DoS."""
    huge = "+".join(["1"] * 5000)  # 远大于 _MAX_EXPR_LEN
    with pytest.raises(SafeEvalError):
        safe_eval(huge)


def test_safe_eval_rejects_empty_expression():
    with pytest.raises(SafeEvalError):
        safe_eval("")
    with pytest.raises(SafeEvalError):
        safe_eval("   ")


def test_safe_eval_rejects_non_string():
    with pytest.raises(SafeEvalError):
        safe_eval(123)  # type: ignore[arg-type]


# ============ calculator 工具入口: 拒绝时不抛异常, 返回错误字符串 ============

def test_calculator_tool_returns_error_string_on_evil(monkeypatch):
    """calculator() 工具对恶意输入返回错误字符串, 而不是把异常抛给 agent."""
    from app.agents.tools import build_tools

    # 用 kb_id=default 构造工具, 不需要真实 embedding/向量库 (我们只测 calculator).
    # 通过 monkeypatch 让 retriever/embedding 相关导入可延迟加载失败也不影响 calculator.
    try:
        tools = build_tools("__no_kb__")
    except Exception:
        # 如果 build_tools 因 embedding 加载失败而抛, 我们直接测 safe_eval 已经够了,
        # calculator 工具是薄壳.
        pytest.skip("build_tools requires embedding model in this env")

    calc = next(t for t in tools if t.name == "calculator")
    # LangChain @tool 对象: 通过 invoke 跑
    res = calc.invoke({"expression": "__import__('os').system('echo pwned')"})
    assert isinstance(res, str)
    assert "计算失败" in res
    # 关键: 不能让任何子进程跑起来. 由于上面的拒绝, system 根本不会被调用.
    assert "pwned" not in res


def test_calculator_tool_computes_normal_expression(monkeypatch):
    from app.agents.tools import build_tools

    try:
        tools = build_tools("__no_kb__")
    except Exception:
        pytest.skip("build_tools requires embedding model in this env")

    calc = next(t for t in tools if t.name == "calculator")
    res = calc.invoke({"expression": "sqrt(16) + 2"})
    assert res == "6.0"


# ============ 防止 __init__ 包裹 (绕过白名单技巧) ============

def test_safe_eval_blocks_dunder_attribute_access():
    """__class__ / __init__ 等 dunder 通过 Attribute 节点访问会被拒绝."""
    payloads = [
        "(1).__class__",
        "[].__class__",
        "str.__class__",
    ]
    for p in payloads:
        with pytest.raises(SafeEvalError):
            safe_eval(p)


def test_safe_eval_blocks_string_concatenation_and_call():
    """字符串 + str() 这种组合也被拒绝."""
    payloads = [
        "'hello' + 'world'",
        "str(1) + '2'",
    ]
    for p in payloads:
        with pytest.raises(SafeEvalError):
            safe_eval(p)


def test_safe_eval_blocks_nested_attribute_on_whitelisted_name():
    """sqrt 允许调用, 但 sqrt.__class__ 这种属性访问不行."""
    payloads = [
        "sqrt.__class__",
        "pi.real",
    ]
    for p in payloads:
        with pytest.raises(SafeEvalError):
            safe_eval(p)