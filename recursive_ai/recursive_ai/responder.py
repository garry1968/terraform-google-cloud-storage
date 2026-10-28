"""Responder: executes a plan, calling tools through the router."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .config import prompt_for
from .llm import LLM
from .planner import Plan, render_hints, render_memories, sections
from .tools import ToolResult, ToolRouter

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "confidence", "evidence"],
    "additionalProperties": False,
}


@dataclass
class Draft:
    answer: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class Responder:
    def __init__(self, llm: LLM, router: ToolRouter):
        self.llm = llm
        self.router = router

    def respond(self, request: str, plan: Plan, hints: list[dict[str, str]],
                memories: list[dict[str, Any]], config: dict[str, Any],
                feedback: list[str] | None = None,
                previous: str | None = None) -> Draft:
        prompt = sections(
            request=request,
            plan=plan.render(),
            routing_hints=render_hints(hints),
            relevant_past_successes=render_memories(memories),
            previous_answer=previous or "",
            issues_to_fix="\n".join(f"- {i}" for i in feedback or []),
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        tools = self.router.definitions(config)
        purpose = "revise" if feedback else "respond"
        results: list[ToolResult] = []

        for _ in range(config["planner"]["max_tool_rounds"]):
            response = self.llm.generate(
                purpose=purpose, system=prompt_for(config, "responder"),
                messages=messages, tools=tools or None,
                schema=RESPONSE_SCHEMA)
            if not response.tool_calls:
                return self._parse(response.text, results)

            # Append-only history: the assistant turn goes back unchanged,
            # then every tool result in a single user turn.
            messages.append({"role": "assistant", "content": response.content})
            blocks = []
            for call in response.tool_calls:
                result = self.router.execute(call.name, call.input, config)
                results.append(result)
                block = {"type": "tool_result", "tool_use_id": call.id,
                         "content": result.output}
                if not result.ok:
                    block["is_error"] = True
                blocks.append(block)
            messages.append({"role": "user", "content": blocks})

        return Draft("I ran out of tool steps before finishing.", 0.1,
                     tool_results=results, notes=["tool round limit reached"])

    @staticmethod
    def _parse(text: str, results: list[ToolResult]) -> Draft:
        try:
            data = json.loads(text)
            confidence = min(1.0, max(0.0, float(data["confidence"])))
            return Draft(str(data["answer"]), confidence,
                         [str(e) for e in data.get("evidence", [])], results)
        except (ValueError, KeyError, TypeError):
            return Draft(text.strip() or "(no answer)", 0.3,
                         tool_results=results,
                         notes=["response was not valid JSON"])
