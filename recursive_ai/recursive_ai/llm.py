"""Model access.

ClaudeLLM calls the Claude API. OfflineLLM is a deterministic stand-in so
the whole loop (including learning and approvals) can be demonstrated and
tested without an API key. It is pattern matching, not intelligence.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMError(RuntimeError):
    """The model call failed or returned something unusable."""


class LLMRefusal(LLMError):
    """The model declined the request (stop_reason == "refusal")."""


class BudgetExceeded(RuntimeError):
    """The run reached its API spending cap. Raised before the next call.

    Deliberately not an LLMError: components that degrade gracefully on a
    model error must not keep going once the money is spent.
    """


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    # Raw content blocks, appended unchanged to the history in tool loops.
    content: list[Any] = field(default_factory=list)


class LLM(Protocol):
    def generate(self, *, purpose: str, system: str,
                 messages: list[dict[str, Any]],
                 tools: list[dict[str, Any]] | None = None,
                 schema: dict[str, Any] | None = None) -> LLMResponse: ...


# Effort per call site. Claude Opus 5.5 defaults to "medium", so set it.
DEFAULT_EFFORT = {
    "plan": "medium",
    "respond": "high",
    "revise": "high",
    "evaluate": "medium",
    "propose": "high",
    "refine": "high",
}

# US dollars per million tokens: (input, output, cache read). Cache writes
# bill at 1.25x input. A model missing here is priced as the dearest one, so
# the spending cap errs on the safe side.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-fable-5-1": (10.00, 50.00, 0.25),
}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0  # estimated US dollars

    def add(self, usage: Any, model: str) -> None:
        price_in, price_out, price_read = next(
            (p for name, p in PRICES.items() if model.startswith(name)),
            max(PRICES.values()))

        def count(name: str) -> int:
            return getattr(usage, name, 0) or 0

        fresh = count("input_tokens")
        written = count("cache_creation_input_tokens")
        read = count("cache_read_input_tokens")
        out = count("output_tokens")
        self.input_tokens += fresh + written + read
        self.output_tokens += out
        self.cost += (fresh * price_in + written * price_in * 1.25
                      + read * price_read + out * price_out) / 1_000_000


class Budget:
    """Caps the model calls and estimated spend of one run. Wraps any LLM."""

    def __init__(self, llm: LLM, max_calls: int, max_cost: float):
        self.llm = llm
        self.max_calls = max_calls
        self.max_cost = max_cost
        self.calls = 0

    @property
    def cost(self) -> float:
        usage = getattr(self.llm, "usage", None)
        return usage.cost if usage else 0.0

    def generate(self, **kwargs: Any) -> LLMResponse:
        if self.calls >= self.max_calls:
            raise BudgetExceeded(
                f"reached the cap of {self.max_calls} API calls")
        if self.cost >= self.max_cost:
            raise BudgetExceeded(
                f"reached the ${self.max_cost:.2f} spending cap")
        self.calls += 1
        return self.llm.generate(**kwargs)

    def summary(self) -> str:
        return f"API use: {self.calls} calls, about ${self.cost:.2f}"


class ClaudeLLM:
    def __init__(self, model: str = "claude-opus-5-5",
                 effort: dict[str, str] | None = None,
                 max_tokens: int = 16000, client: Any = None):
        import anthropic  # imported lazily so offline mode needs no SDK

        self._anthropic = anthropic
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.effort = {**DEFAULT_EFFORT, **(effort or {})}
        self.max_tokens = max_tokens
        self.usage = Usage()

    def generate(self, *, purpose, system, messages, tools=None,
                 schema=None) -> LLMResponse:
        output_config: dict[str, Any] = {
            "effort": self.effort.get(purpose, "medium")}
        if schema:
            output_config["format"] = {"type": "json_schema",
                                       "schema": schema}
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": messages,
            "output_config": output_config,
            "cache_control": {"type": "ephemeral"},
            # Server-side fallback: a declined request is retried on a
            # recommended fallback model instead of failing outright.
            "betas": ["server-side-fallback-2026-07-01"],
            "fallbacks": "default",
        }
        if tools:
            kwargs["tools"] = tools

        anthropic = self._anthropic
        try:
            response = self.client.beta.messages.create(**kwargs)
        except anthropic.RateLimitError as exc:
            raise LLMError(f"rate limited: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"API error {exc.status_code}: {exc.message}") \
                from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"connection error: {exc}") from exc

        # Recorded before the checks below: a declined or truncated
        # response is still billed.
        self.usage.add(response.usage,
                       getattr(response, "model", None) or self.model)
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None)
            raise LLMRefusal(f"model declined (category: {category})")
        if response.stop_reason == "max_tokens":
            raise LLMError("response truncated at max_tokens")

        text = "".join(b.text for b in response.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, b.input)
                 for b in response.content if b.type == "tool_use"]
        return LLMResponse(text, calls, response.stop_reason,
                           list(response.content))


# --------------------------------------------------------------------------
# Offline stand-in
# --------------------------------------------------------------------------

_WORD_OPS = [
    (r"\bmultiplied by\b", "*"), (r"\btimes\b", "*"),
    (r"\bdivided by\b", "/"), (r"\bplus\b", "+"), (r"\bminus\b", "-"),
]


def _section(text: str, name: str) -> str:
    match = re.search(rf"{name}:\n(.*?)(?:\n\n[A-Z][\w ]*:\n|\Z)", text,
                      re.DOTALL)
    return match.group(1).strip() if match else ""


def _message_text(message: dict[str, Any]) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content
                     if isinstance(b, dict))


def _json(payload: dict[str, Any]) -> LLMResponse:
    text = json.dumps(payload)
    return LLMResponse(text, content=[{"type": "text", "text": text}])


class OfflineLLM:
    """Deterministic rules that mimic each call site's output shape.

    It answers only when a routing hint points it at a tool, so learning a
    new routing rule visibly changes its behaviour.
    """

    def generate(self, *, purpose, system, messages, tools=None,
                 schema=None) -> LLMResponse:
        handler = getattr(self, f"_{purpose}", None)
        if handler is None:
            raise LLMError(f"offline model has no handler for {purpose!r}")
        return handler(system, messages, tools or [])

    def _plan(self, system, messages, tools):
        text = _message_text(messages[0])
        steps = [{"description": f"Use {tool} to get an exact result",
                  "tool": tool}
                 for tool in re.findall(r"Suggested tool: (\w+)", text)]
        steps.append({"description": "Write the answer", "tool": "none"})
        return _json({"approach": "direct", "complexity": "low",
                      "steps": steps})

    def _respond(self, system, messages, tools):
        last = messages[-1]["content"]
        if isinstance(last, list) and last and \
                last[0].get("type") == "tool_result":
            outputs = [b["content"] for b in last]
            return _json({"answer": f"The answer is {outputs[0]}.",
                          "confidence": 0.9,
                          "evidence": [f"tool returned {o}" for o in outputs]})

        first = _message_text(messages[0])
        request = _section(first, "Request")
        names = {t["name"] for t in tools}
        for tool in re.findall(r"Suggested tool: (\w+)", first):
            if tool not in names:
                continue
            if tool == "calculator":
                expr = request
                for pattern, symbol in _WORD_OPS:
                    expr = re.sub(pattern, symbol, expr, flags=re.IGNORECASE)
                found = re.findall(r"[\d.\s+\-*/()%^]{3,}", expr)
                if not found:
                    continue
                args = {"expression": max(found, key=len).strip()}
            elif tool == "word_count":
                args = {"text": request.split(":", 1)[-1].strip()}
            elif tool == "clock":
                args = {}
            else:
                continue
            call = ToolCall("toolu_" + uuid.uuid4().hex[:12], tool, args)
            return LLMResponse("", [call], "tool_use",
                               [{"type": "tool_use", "id": call.id,
                                 "name": tool, "input": args}])

        return _json({"answer": "I'm not certain; I could not verify this "
                                "offline.",
                      "confidence": 0.3, "evidence": []})

    _revise = _respond

    def _evaluate(self, system, messages, tools):
        answer = _section(_message_text(messages[0]), "Answer")
        if "not certain" in answer:
            return _json({"correctness": 0.2, "completeness": 0.3,
                          "relevance": 0.6,
                          "issues": ["The answer does not resolve the "
                                     "request."]})
        return _json({"correctness": 0.9, "completeness": 0.9,
                      "relevance": 0.9, "issues": []})

    def _propose(self, system, messages, tools):
        text = _message_text(messages[0])
        proposals = []
        if "ungrounded_math" in text and '"math-words"' not in text:
            rule = {"id": "math-words",
                    "pattern": r"\d+(\.\d+)?\s+(times|plus|minus|"
                               r"multiplied by|divided by)\s+\d+",
                    "tool": "calculator",
                    "hint": "Use the calculator for arithmetic written in "
                            "words."}
            proposals.append({
                "target": "routing_rules", "operation": "append",
                "value_json": json.dumps(rule),
                "rationale": "Arithmetic phrased in words was answered "
                             "without the calculator.",
                "risk": "low"})
            # Deliberately under-rated: shows the gate ignores a low claim.
            proposals.append({
                "target": "planner.max_steps", "operation": "set",
                "value_json": "8",
                "rationale": "Allow longer plans for multi-step maths.",
                "risk": "low"})
        elif "low_confidence" in text:
            proposals.append({
                "target": "prompt_addenda.responder", "operation": "append",
                "value_json": json.dumps(
                    "When unsure, say what extra information would let you "
                    "answer."),
                "rationale": "Low-confidence answers gave the user no next "
                             "step.",
                "risk": "low"})
        return _json({"proposals": proposals})

    def _refine(self, system, messages, tools):
        return _json({"proposals": []})
