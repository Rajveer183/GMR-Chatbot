"""Calculator tool -- exact arithmetic without eval().

Who controls it: application code. The LLM only supplies an expression string;
it is parsed with Python's `ast` and evaluated by walking an explicit whitelist
of nodes, so names, attributes, imports, calls to anything not listed,
comprehensions, lambdas and strings can never run.

    allowed   numbers, + - * / // % **, unary + -, parentheses,
              round abs min max sqrt log log10 exp ceil floor pow
    limits    expression <= 300 chars, exponent <= 64, |values| <= 1e18,
              nesting depth <= 50
"""

from __future__ import annotations

import ast
import math
import operator

MAX_LEN = 300
MAX_EXPONENT = 64
MAX_ABS = 1e18
MAX_DEPTH = 50

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"round": round, "abs": abs, "min": min, "max": max, "sqrt": math.sqrt, "log": math.log,
          "log10": math.log10, "exp": math.exp, "ceil": math.ceil, "floor": math.floor, "pow": pow}


class CalculatorError(ValueError):
    """The expression is not allowed or cannot be evaluated."""


def _check(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalculatorError("only numbers are allowed")
    if isinstance(value, float) and not math.isfinite(value):
        raise CalculatorError("result is not a finite number")
    if abs(value) > MAX_ABS:
        raise CalculatorError("number too large")
    return value


def _eval(node: ast.AST, depth: int = 0):
    if depth > MAX_DEPTH:
        raise CalculatorError("expression too deeply nested")
    if isinstance(node, ast.Expression):
        return _eval(node.body, depth + 1)
    if isinstance(node, ast.Constant):
        return _check(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _check(_UNARY[type(node.op)](_eval(node.operand, depth + 1)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _eval(node.left, depth + 1), _eval(node.right, depth + 1)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalculatorError(f"exponent above {MAX_EXPONENT} is not allowed")
        try:
            return _check(_BINOPS[type(node.op)](left, right))
        except ZeroDivisionError:
            raise CalculatorError("division by zero") from None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        args = [_eval(a, depth + 1) for a in node.args]
        if node.func.id == "pow" and len(args) >= 2 and abs(args[1]) > MAX_EXPONENT:
            raise CalculatorError(f"exponent above {MAX_EXPONENT} is not allowed")
        try:
            return _check(_FUNCS[node.func.id](*args))
        except (TypeError, ValueError) as exc:
            raise CalculatorError(f"{node.func.id}(): {exc}") from None
    raise CalculatorError(f"'{type(node).__name__}' is not allowed in a calculation")


def evaluate(expression: str) -> float | int:
    """Evaluate one arithmetic expression; CalculatorError when it is not allowed."""
    text = str(expression or "").strip()
    if not text:
        raise CalculatorError("empty expression")
    if len(text) > MAX_LEN:
        raise CalculatorError(f"expression longer than {MAX_LEN} characters")
    # Thousands separators in a plain number ("4,214,678") -- but never inside a call's argument list.
    if "(" not in text:
        text = text.replace(",", "")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        raise CalculatorError("not a valid arithmetic expression") from None
    result = _eval(tree)
    return round(result, 10) if isinstance(result, float) else result
