"""Baseline evaluation: does the system beat its own underlying model?

Runs generated tasks with known answers (tasks.py) under three conditions,
all on the same model and settings:

- plain: one call, no tools.
- tools: the responder with its tools and default settings. No planner,
  self-correction, memory or learned changes.
- full: the whole pipeline with the current learned settings and memory.

The full condition runs on a throwaway copy of the state folder and no
condition can learn, so an evaluation never changes the system.
"""

from __future__ import annotations

import copy
import shutil
import time
from collections.abc import Iterator
from dataclasses import dataclass
from math import comb
from pathlib import Path

from .config import DEFAULT_CONFIG
from .llm import LLM, LLMError
from .memory import MemoryStore
from .orchestrator import Orchestrator
from .planner import Plan, PlanStep
from .responder import Responder
from .tasks import Task, grade
from .tools import ToolRouter, default_tools

CONDITIONS = ("plain", "tools", "full")
PLAIN_SYSTEM = "You are a careful assistant. Answer the user's question."


@dataclass
class TrialResult:
    task: str
    category: str
    condition: str
    trial: int
    correct: bool
    cost: float  # estimated US dollars
    seconds: float
    calls: int = 0  # model calls, including tool rounds and revisions
    error: str = ""
    reply: str = ""  # kept so every grade can be checked by hand


class Baseline:
    def __init__(self, llm: LLM, state_dir: Path, workdir: Path):
        self.llm = llm
        full_dir = workdir / "full"
        if state_dir.exists():
            shutil.copytree(state_dir, full_dir,
                            ignore=shutil.ignore_patterns(".git"))
        self.full = Orchestrator(llm, full_dir)
        self.tools_config = copy.deepcopy(DEFAULT_CONFIG)
        memory = MemoryStore(workdir / "empty-memory.json",
                             self.tools_config["memory_schema"]["version"])
        self.router = ToolRouter(default_tools(memory))
        self.responder = Responder(llm, self.router)

    @property
    def cost(self) -> float:
        return getattr(self.llm, "cost", 0.0)

    @property
    def calls(self) -> int:
        return getattr(self.llm, "calls", 0)

    def answer(self, condition: str, task: Task) -> str:
        if condition == "plain":
            return self.llm.generate(
                purpose="respond", system=PLAIN_SYSTEM,
                messages=[{"role": "user", "content": task.prompt}]).text
        if condition == "tools":
            config = self.tools_config
            plan = Plan("direct", "low", [PlanStep("Answer the request")])
            hints = self.router.route_hints(task.prompt, config)
            return self.responder.respond(task.prompt, plan, hints, [],
                                          config).answer
        result = self.full.run_pipeline(task.prompt, self.full.store.current)
        return result.draft.answer

    def run(self, tasks: list[Task], conditions: tuple[str, ...],
            trials: int) -> Iterator[TrialResult]:
        """Conditions take turns on each task, so a run stopped by the
        spending cap still compares them on the same tasks."""
        for trial in range(1, trials + 1):
            for task in tasks:
                for condition in conditions:
                    cost, calls = self.cost, self.calls
                    start = time.monotonic()
                    try:
                        reply = self.answer(condition, task)
                        correct, error = grade(task, reply), ""
                    except LLMError as exc:
                        reply, correct, error = "", False, str(exc)
                    yield TrialResult(task.id, task.category, condition,
                                      trial, correct, self.cost - cost,
                                      time.monotonic() - start,
                                      self.calls - calls, error, reply)


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test on paired wins and losses."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def _matching(results: list[TrialResult], **fields: object
              ) -> list[TrialResult]:
    return [r for r in results
            if all(getattr(r, k) == v for k, v in fields.items())]


def summarize(results: list[TrialResult],
              conditions: tuple[str, ...]) -> list[str]:
    if not results:
        return ["No tasks finished."]
    lines = ["Condition   Correct     Cost/task  Calls/task  Time/task  "
             "Errors"]
    for c in conditions:
        rs = _matching(results, condition=c)
        if rs:
            right, n = sum(r.correct for r in rs), len(rs)
            lines.append(
                f"{c:<10}{right:>4}/{n:<3}{right / n:>5.0%}   "
                f"${sum(r.cost for r in rs) / n:.4f}    "
                f"{sum(r.calls for r in rs) / n:5.1f}     "
                f"{sum(r.seconds for r in rs) / n:5.1f}s   "
                f"{sum(1 for r in rs if r.error):>4}")

    lines.append("By category:")
    for cat in dict.fromkeys(r.category for r in results):
        cells = []
        for c in conditions:
            rs = _matching(results, category=cat, condition=c)
            cells.append(f"{c} {sum(r.correct for r in rs)}/{len(rs)}")
        lines.append(f"  {cat:<9} " + "   ".join(cells))

    outcome = {(r.task, r.trial, r.condition): r.correct for r in results}
    pairs = {(r.task, r.trial) for r in results}
    for c in conditions:
        if c == "plain" or "plain" not in conditions:
            continue
        both = [p for p in pairs if (*p, c) in outcome
                and (*p, "plain") in outcome]
        wins = sum(outcome[(*p, c)] and not outcome[(*p, "plain")]
                   for p in both)
        losses = sum(outcome[(*p, "plain")] and not outcome[(*p, c)]
                     for p in both)
        lines.append(f"{c} vs plain: better on {wins}, worse on {losses} "
                     f"(sign test p = {sign_test(wins, losses):.2f})")

    if len({r.trial for r in results}) > 1:
        for c in conditions:
            per_task: dict[str, set[bool]] = {}
            for r in _matching(results, condition=c):
                per_task.setdefault(r.task, set()).add(r.correct)
            if per_task:
                steady = sum(len(v) == 1 for v in per_task.values())
                lines.append(f"{c}: same result on every trial for "
                             f"{steady}/{len(per_task)} tasks")
    return lines
