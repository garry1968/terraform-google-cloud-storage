"""Tools and the tool router.

The router is the only path from the model to a tool. It enforces
tool_permissions (default deny) on every call and turns routing rules into
hints for the planner and responder.
"""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .memory import MemoryStore


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    func: Callable[[dict[str, Any]], str]


@dataclass
class ToolResult:
    name: str
    input: dict[str, Any]
    ok: bool
    output: str


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def safe_eval(expression: str) -> float:
    """Evaluate plain arithmetic. No names, calls, or attribute access."""
    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value,
                                                         (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        raise ValueError("only arithmetic is allowed")

    if len(expression) > 200:
        raise ValueError("expression too long")
    return walk(ast.parse(expression.replace("^", "**"), mode="eval"))


def _calculator(args: dict[str, Any]) -> str:
    result = safe_eval(str(args["expression"]))
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


def _word_count(args: dict[str, Any]) -> str:
    text = str(args["text"])
    return f"words: {len(text.split())}, characters: {len(text)}"


def _clock(args: dict[str, Any]) -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _schema(**properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def default_tools(memory: MemoryStore) -> dict[str, Tool]:
    def memory_search(args: dict[str, Any]) -> str:
        hits = memory.search(str(args["query"]))
        return "\n".join(f"- {h['request']} -> {h['answer']}"
                         for h in hits) or "no matches"

    tools = [
        Tool("calculator",
             "Evaluate an arithmetic expression exactly, e.g. '17 * 23'. "
             "Supports + - * / // % ** and parentheses.",
             _schema(expression={"type": "string"}), _calculator),
        Tool("word_count",
             "Count the words and characters in a piece of text.",
             _schema(text={"type": "string"}), _word_count),
        Tool("clock", "Return the current UTC date and time.",
             _schema(), _clock),
        Tool("memory_search",
             "Search past conversations for related requests and answers.",
             _schema(query={"type": "string"}), memory_search),
    ]
    return {t.name: t for t in tools}


class ToolRouter:
    def __init__(self, tools: dict[str, Tool]):
        self.tools = tools

    def allowed(self, config: dict[str, Any]) -> list[str]:
        permissions = config["tool_permissions"]
        return [n for n in self.tools if permissions.get(n) == "allow"]

    def definitions(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        """Claude tool definitions for permitted tools only."""
        return [{"name": t.name, "description": t.description,
                 "input_schema": t.input_schema, "strict": True}
                for n, t in self.tools.items() if n in self.allowed(config)]

    def describe(self, config: dict[str, Any]) -> str:
        return "\n".join(f"- {n}: {self.tools[n].description}"
                         for n in self.allowed(config)) or "(no tools)"

    def execute(self, name: str, args: dict[str, Any],
                config: dict[str, Any]) -> ToolResult:
        if name not in self.tools:
            return ToolResult(name, args, False, f"unknown tool: {name}")
        if name not in self.allowed(config):
            return ToolResult(name, args, False,
                              f"permission denied: {name} is not allowed")
        try:
            return ToolResult(name, args, True, self.tools[name].func(args))
        except Exception as exc:  # tool bugs must not crash the loop
            return ToolResult(name, args, False, f"tool error: {exc}")

    def route_hints(self, text: str,
                    config: dict[str, Any]) -> list[dict[str, str]]:
        allowed = self.allowed(config)
        hints = []
        for rule in config["routing_rules"]:
            if rule["tool"] in allowed and re.search(rule["pattern"], text,
                                                     re.IGNORECASE):
                hints.append({"rule": rule["id"], "tool": rule["tool"],
                              "hint": rule["hint"]})
        return hints
