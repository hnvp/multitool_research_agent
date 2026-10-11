from __future__ import annotations
import ast
import math
import operator as op
from langchain.tools import tool
from trace import run_tool_safely

_BINARY_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}

_UNARY_OPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}

_FUNCTIONS = {
    "sqrt": math.sqrt,
}


def _safe_eval(node: ast.AST):
    if isinstance(
        node,
        ast.Expression,
    ):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant) and isinstance(
        node.value,
        (int, float),
    ):
        return node.value

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _safe_eval(node.left)

        right = _safe_eval(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent is too large.")

        return _BINARY_OPS[type(node.op)](
            left,
            right,
        )

    if isinstance(node, ast.Call):
        if (
            not isinstance(node.func, ast.Name)
            or node.func.id not in _FUNCTIONS
            or len(node.args) != 1
            or node.keywords
        ):
            raise ValueError("Unsupported function. Only sqrt() is supported.")

        return _FUNCTIONS[node.func.id](_safe_eval(node.args[0]))

    raise ValueError(
        "Unsupported expression. Use numbers, " "+, -, *, /, //, %, ** and parentheses."
    )


def _calculate(
    expression: str,
):
    if len(expression) > 200:
        raise ValueError("Expression is too long.")

    tree = ast.parse(
        expression,
        mode="eval",
    )

    result = _safe_eval(tree)

    if isinstance(result, float) and result.is_integer():
        return int(result)

    return result


@tool
def calculator(
    expression: str,
) -> str:
    """Calculate a mathematical expression accurately.

    Use this tool for arithmetic such as addition,
    subtraction, multiplication, division, modulo,
    powers, floor division, square roots, and parentheses.
    """
    return run_tool_safely("calculator", _calculate, expression=expression)
