"""Evaluator: scores an answer with an LLM judge plus deterministic checks.

Each problem found is tagged with a failure mode so the learning controller
can group evidence across episodes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .config import prompt_for
from .llm import LLM, LLMError
from .planner import sections
from .responder import Draft
from .safety import ContentScreen

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correctness": {"type": "number"},
        "completeness": {"type": "number"},
        "relevance": {"type": "number"},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["correctness", "completeness", "relevance", "issues"],
    "additionalProperties": False,
}

_MATH_SYMBOLS = re.compile(r"\d\s*[+*/%^x]\s*\d|\d\s+-\s+\d")
_MATH_WORDS = re.compile(
    r"\b(times|plus|minus|divided|multiplied|sum|product|percent|squared|"
    r"average)\b", re.IGNORECASE)


def looks_computational(text: str) -> bool:
    if _MATH_SYMBOLS.search(text):
        return True
    return len(re.findall(r"\d+(?:\.\d+)?", text)) >= 2 \
        and bool(_MATH_WORDS.search(text))


@dataclass
class Issue:
    mode: str
    detail: str


@dataclass
class Evaluation:
    score: float
    passed: bool
    metrics: dict[str, float]
    issues: list[Issue] = field(default_factory=list)

    @property
    def modes(self) -> list[str]:
        return sorted({i.mode for i in self.issues})

    def to_dict(self) -> dict[str, Any]:
        return {"score": round(self.score, 3), "passed": self.passed,
                "metrics": {k: round(v, 3) for k, v in self.metrics.items()},
                "issues": [i.__dict__ for i in self.issues]}


def _clamp(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


class Evaluator:
    def __init__(self, llm: LLM, screen: ContentScreen | None = None):
        self.llm = llm
        self.screen = screen or ContentScreen()

    def evaluate(self, request: str, draft: Draft, config: dict[str, Any],
                 expected: dict[str, Any] | None = None) -> Evaluation:
        metrics: dict[str, float] = {}
        issues: list[Issue] = []

        self._judge(request, draft, config, metrics, issues)

        metrics["confidence"] = draft.confidence
        if draft.confidence < config["self_correction"]["min_confidence"]:
            issues.append(Issue("low_confidence",
                                f"confidence {draft.confidence:.2f}"))

        if draft.tool_results:
            failed = [r for r in draft.tool_results if not r.ok]
            metrics["tool_reliability"] = 1 - len(failed) / len(
                draft.tool_results)
            for r in failed:
                mode = ("permission_denied" if "permission denied" in r.output
                        else "tool_error")
                issues.append(Issue(mode, f"{r.name}: {r.output}"))
        else:
            metrics["tool_reliability"] = 1.0

        # Numbers derived from a calculation should rest on tool evidence.
        metrics["grounding"] = 1.0
        if looks_computational(request) and not any(
                r.ok for r in draft.tool_results):
            metrics["grounding"] = 0.5
            issues.append(Issue("ungrounded_math",
                                "computed a numeric answer without a tool"))

        if expected and expected.get("contains"):
            wanted = expected["contains"]
            hits = sum(1 for w in wanted if w.lower() in draft.answer.lower())
            metrics["expected"] = hits / len(wanted)
            if hits < len(wanted):
                issues.append(Issue("expected_mismatch",
                                    f"answer missing {wanted}"))

        unsafe = self.screen.check_output(draft.answer, config)
        safe = unsafe is None
        if not safe:
            issues.append(Issue("unsafe_output", f"matched {unsafe!r}"))

        weights = config["evaluator"]["weights"]
        total = sum(weights.get(k, 0) for k in metrics)
        score = sum(v * weights.get(k, 0) for k, v in metrics.items()) / total \
            if total else 0.0
        if not safe:
            score = 0.0
        passed = (score >= config["evaluator"]["pass_threshold"] and safe
                  and metrics.get("expected", 1.0) == 1.0)
        return Evaluation(score, passed, metrics, issues)

    def _judge(self, request, draft, config, metrics, issues) -> None:
        tool_trace = "\n".join(
            f"- {r.name}({json.dumps(r.input)}) -> {r.output}"
            for r in draft.tool_results)
        prompt = sections(request=request, answer=draft.answer,
                          tool_results=tool_trace,
                          stated_evidence="\n".join(draft.evidence))
        try:
            response = self.llm.generate(
                purpose="evaluate", system=prompt_for(config, "evaluator"),
                messages=[{"role": "user", "content": prompt}],
                schema=JUDGE_SCHEMA)
            data = json.loads(response.text)
        except (LLMError, ValueError) as exc:
            issues.append(Issue("judge_unavailable", str(exc)))
            return
        for key in ("correctness", "completeness", "relevance"):
            metrics[key] = _clamp(data.get(key))
        for text in data.get("issues", []):
            issues.append(Issue("judge_flagged", str(text)))
