"""Planner: turns a request into a short, tool-aware plan."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .config import prompt_for
from .llm import LLM, LLMError, LLMRefusal
from .tools import ToolRouter

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "approach": {"type": "string"},
        "complexity": {"type": "string", "enum": ["low", "medium", "high"]},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "tool": {"type": "string"},
                },
                "required": ["description", "tool"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["approach", "complexity", "steps"],
    "additionalProperties": False,
}


def sections(**named: str) -> str:
    """Render 'Name:\\nbody' blocks separated by blank lines."""
    return "\n\n".join(f"{name.replace('_', ' ').capitalize()}:\n{body}"
                       for name, body in named.items() if body)


def render_hints(hints: list[dict[str, str]]) -> str:
    return "\n".join(f"Suggested tool: {h['tool']} - {h['hint']}"
                     for h in hints)


def render_memories(memories: list[dict[str, Any]]) -> str:
    return "\n".join(f"- Q: {m['request']}\n  A: {m['answer']}"
                     for m in memories)


@dataclass
class PlanStep:
    description: str
    tool: str | None = None


@dataclass
class Plan:
    approach: str
    complexity: str
    steps: list[PlanStep]
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        return "\n".join(
            f"{i}. {s.description}" + (f" [tool: {s.tool}]" if s.tool else "")
            for i, s in enumerate(self.steps, 1))

    def to_dict(self) -> dict[str, Any]:
        return {"approach": self.approach, "complexity": self.complexity,
                "steps": [s.__dict__ for s in self.steps],
                "notes": self.notes}


class Planner:
    def __init__(self, llm: LLM, router: ToolRouter):
        self.llm = llm
        self.router = router

    def plan(self, request: str, hints: list[dict[str, str]],
             memories: list[dict[str, Any]],
             config: dict[str, Any]) -> Plan:
        max_steps = config["planner"]["max_steps"]
        prompt = sections(
            request=request,
            available_tools=self.router.describe(config),
            routing_hints=render_hints(hints),
            relevant_past_successes=render_memories(memories),
            limits=f"Use at most {max_steps} steps. Use tool 'none' for "
                   "steps that need no tool.",
        )
        try:
            response = self.llm.generate(
                purpose="plan", system=prompt_for(config, "planner"),
                messages=[{"role": "user", "content": prompt}],
                schema=PLAN_SCHEMA)
            data = json.loads(response.text)
            raw_steps = data["steps"]
        except LLMRefusal:
            raise
        except (LLMError, ValueError, KeyError, TypeError) as exc:
            return Plan("direct", "low",
                        [PlanStep("Answer the request directly")],
                        notes=[f"planner fallback: {exc}"])

        allowed = set(self.router.allowed(config))
        steps, notes = [], []
        for raw in raw_steps[:max_steps]:
            tool = (raw.get("tool") or "").strip()
            if tool.lower() in ("", "none"):
                tool = None
            elif tool not in allowed:
                notes.append(f"dropped unpermitted tool {tool!r}")
                tool = None
            steps.append(PlanStep(str(raw.get("description", "")), tool))
        if len(raw_steps) > max_steps:
            notes.append(f"plan truncated to {max_steps} steps")
        complexity = data.get("complexity", "low")
        if complexity not in ("low", "medium", "high"):
            complexity = "low"
        return Plan(str(data.get("approach", "")), complexity,
                    steps or [PlanStep("Answer the request directly")], notes)
