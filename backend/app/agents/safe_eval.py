"""轻量级安全表达式求值器, 用于替换 calculator 工具里的 eval().

设计目标:
1. 不使用 eval / exec / compile(..., 'eval') -- 不给 Python 字节码解释器任何机会.
2. 解析为 ast 树, 只允许一组明确的节点类型 (白名单).
3. 拒绝任何 Attribute / Subscript / Lambda / comprehension / JoinedStr 等"逃逸载体".
4. Name / Call.func 的标识符必须只含 [A-Za-z0-9_], 不能以 _ 开头, 且必须在预定义白名单里.
5. 限制表达式长度 (避免 DoS 巨型 AST).

支持: 二元运算 (+ - * / // % **), 一元正负, 字面量数字 (int/float/布尔), 预定义函数调用.

不是目标: 完整 Python 表达式语言. Agent 拿来做算术和 math.* 就够了.
"""
from __future__ import annotations

import ast
import math
from typing import Any


# 允许的 AST 节点类型集合
_ALLOWED_NODE_TYPES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Call,
    # 运算符
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
}


# 允许的二元算子 -- 用作白名单对照, 不允许 ast.BitOr/BitAnd/LShift/RShift/And/Or 等
_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


# 允许的常量类型 -- 只放数字, 拒绝 str / bytes (防止拼 payload)
_ALLOWED_CONST_TYPES = (int, float, bool)


# 白名单函数表. 这些函数名可以在表达式里直接用, 例如 sqrt(16), log(100), floor(3.7).
# 注意: 这份表里不放任何 I/O / os / sys / subprocess / open 类函数, 也不放 __xxx__ 类.
_SAFE_FUNCTIONS: dict[str, Any] = {
    # 数学
    "sqrt": math.sqrt,
    "pow": math.pow,
    "exp": math.exp,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "degrees": math.degrees,
    "radians": math.radians,
    "ceil": math.ceil,
    "floor": math.floor,
    "fabs": math.fabs,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "fmod": math.fmod,
    "hypot": math.hypot,
    "trunc": math.trunc,
    # 常用内置
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    # 常用常量
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
    "nan": math.nan,
}


# 表达式长度上限 (字符数). AST 节点数上限在求值器内部另外统计.
_MAX_EXPR_LEN = 500
_MAX_AST_NODES = 200


class SafeEvalError(ValueError):
    """安全求值器抛出的错误, 包装语法错误/拒绝节点/未知名字等."""


def _validate_name(name: str) -> bool:
    """标识符合法性: 只允许字母数字下划线, 不能以 _ 开头 (防 dunder / 单下划线私有名)."""
    if not name or not isinstance(name, str):
        return False
    if name.startswith("_"):
        return False
    if not all(c.isalnum() or c == "_" for c in name):
        return False
    return True


def _eval_node(node: ast.AST, *, depth: int = 0) -> Any:
    """递归求值. depth 是为了防止意外深嵌套 (虽然白名单已限制)."""
    if depth > _MAX_AST_NODES:
        raise SafeEvalError("表达式嵌套过深")

    node_type = type(node)
    if node_type not in _ALLOWED_NODE_TYPES:
        raise SafeEvalError(f"不支持的语法: {node_type.__name__}")

    if isinstance(node, ast.Expression):
        return _eval_node(node.body, depth=depth + 1)

    if isinstance(node, ast.Constant):
        if not isinstance(node.value, _ALLOWED_CONST_TYPES):
            raise SafeEvalError(f"不允许的常量类型: {type(node.value).__name__}")
        return node.value

    if isinstance(node, ast.Name):
        if not _validate_name(node.id):
            raise SafeEvalError(f"非法名字: {node.id!r}")
        if node.id not in _SAFE_FUNCTIONS:
            raise SafeEvalError(f"未知名字: {node.id!r}")
        return _SAFE_FUNCTIONS[node.id]

    if isinstance(node, ast.Call):
        # Call.func 必须是 ast.Name (禁止 Attribute: 即拒绝 'math.sqrt' 这种形式).
        func = node.func
        if not isinstance(func, ast.Name):
            raise SafeEvalError("只允许调用白名单内的顶层函数, 不允许属性访问")
        if not _validate_name(func.id):
            raise SafeEvalError(f"非法函数名: {func.id!r}")
        if func.id not in _SAFE_FUNCTIONS:
            raise SafeEvalError(f"不允许的函数: {func.id!r}")
        # 不允许 keyword args (例如 abs(x=-1)), 简化攻击面
        if node.keywords:
            raise SafeEvalError("不支持关键字参数")
        if node.args:
            # 限制参数个数, 防 fanout 攻击
            if len(node.args) > 4:
                raise SafeEvalError("函数参数过多")
        func_obj = _SAFE_FUNCTIONS[func.id]
        if not callable(func_obj):
            # 比如 pi / e / inf 是常量, 不能调用
            raise SafeEvalError(f"{func.id!r} 不是函数")
        args = [_eval_node(a, depth=depth + 1) for a in node.args]
        try:
            return func_obj(*args)
        except Exception as e:
            raise SafeEvalError(f"{func.id} 计算失败: {e}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BINOPS:
            raise SafeEvalError(f"不支持的二元运算: {op_type.__name__}")
        left = _eval_node(node.left, depth=depth + 1)
        right = _eval_node(node.right, depth=depth + 1)
        try:
            return _ALLOWED_BINOPS[op_type](left, right)
        except Exception as e:
            raise SafeEvalError(f"二元运算失败: {e}")

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARYOPS:
            raise SafeEvalError(f"不支持的一元运算: {op_type.__name__}")
        operand = _eval_node(node.operand, depth=depth + 1)
        try:
            return _ALLOWED_UNARYOPS[op_type](operand)
        except Exception as e:
            raise SafeEvalError(f"一元运算失败: {e}")

    # 兜底 (理论不会到这里, 因为 _ALLOWED_NODE_TYPES 已经限制)
    raise SafeEvalError(f"意外节点: {node_type.__name__}")


def safe_eval(expression: str) -> Any:
    """对表达式做白名单 AST 求值. 任何不安全语法都抛 SafeEvalError."""
    if not isinstance(expression, str):
        raise SafeEvalError("表达式必须是字符串")
    if len(expression) > _MAX_EXPR_LEN:
        raise SafeEvalError(f"表达式过长 (上限 {_MAX_EXPR_LEN} 字符)")
    expr = expression.strip()
    if not expr:
        raise SafeEvalError("表达式为空")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise SafeEvalError(f"语法错误: {e.msg}")
    # 节点总数检查
    n_nodes = sum(1 for _ in ast.walk(tree))
    if n_nodes > _MAX_AST_NODES:
        raise SafeEvalError(f"表达式节点数过多 (上限 {_MAX_AST_NODES})")
    return _eval_node(tree)