"""Small, closed compile-time expression grammar; never executes Python code."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass


@dataclass(frozen=True)
class Expression:
    op: str
    left: object
    right: object


# Binding powers keep additions to the numeric grammar local to this table.
OPERATORS = {
    "+": (10, operator.add),
    "-": (10, operator.sub),
    "*": (20, operator.mul),
    "/": (20, operator.truediv),
    "//": (20, operator.floordiv),
    "%": (20, operator.mod),
}


def parse_expression(cur, minimum=0):
    kind, token = cur.next("numeric expression")
    if kind == "number":
        value = float(token) if any(c in token for c in ".eE") else int(token)
    elif kind == "ident":
        value = token
    elif token in ("+", "-"):
        value = Expression(token, 0, parse_expression(cur, 30))
    elif token == "(":
        if cur.version != "0.3":
            raise cur.error("numeric expressions require 'wgt 0.3'")
        value = parse_expression(cur)
        cur.expect("punct", ")")
    else:
        raise cur.error(f"expected numeric expression, found {token!r}")
    while cur.peek():
        kind, token = cur.peek()
        # The legacy lexer keeps negative literals intact for data rows/options.
        signed = kind == "number" and token.startswith("-")
        op = "+" if signed else token
        if op not in OPERATORS or OPERATORS[op][0] < minimum:
            break
        if not signed:
            cur.next("operator")
        right = parse_expression(cur, OPERATORS[op][0] + 1)
        value = Expression(op, value, right)
    if isinstance(value, Expression) and cur.version != "0.3":
        raise cur.error("numeric expressions require 'wgt 0.3'")
    return value


def evaluate(value, env, error):
    if isinstance(value, str):
        if value not in env:
            raise error(f"unknown parameter {value!r}")
        return env[value]
    if isinstance(value, Expression):
        left = evaluate(value.left, env, error)
        right = evaluate(value.right, env, error)
        try:
            value = OPERATORS[value.op][1](left, right)
        except (ArithmeticError, ValueError):
            raise error("invalid numeric expression (division by zero or overflow)") from None
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise error("numeric expression must be finite")
    return value


def logical_lines(source, strip_comment, error):
    """Join bracketed physical lines, preserving the statement's start line."""
    pending = []
    stack = []
    start = 1
    for lineno, raw in enumerate(source.splitlines(), 1):
        text = strip_comment(raw)
        if not pending:
            start = lineno
        quoted = False
        for ch in text:
            if ch == '"':
                quoted = not quoted
            elif not quoted:
                if ch in "([":
                    stack.append(ch)
                elif ch in ")]":
                    if not stack or stack.pop() != {")": "(", "]": "["}[ch]:
                        raise error("unmatched closing bracket", lineno)
        pending.append(text if len(pending) == 0 else text.strip())
        if not stack:
            yield start, " ".join(pending), lineno != start
            pending = []
    if stack:
        raise error("unclosed bracket", start)
