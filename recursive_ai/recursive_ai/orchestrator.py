"""Orchestrator: wires the components and runs the per-request pipeline.

Per request: input -> plan -> respond (tools) -> evaluate -> self-correct ->
output, then hands the episode to the learning controller.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ConfigStore
from .evaluator import Evaluation, Evaluator
from .interface import InputRejected, parse_input
from .learning import LearningController, LearningReport, Outcome, signature
from .llm import LLM, BudgetExceeded, LLMError, LLMRefusal
from .memory import MemoryStore
from .output import OutputLayer, Reply
from .planner import Plan, Planner
from .practice import PRACTICE_SET
from .responder import Draft, Responder
from .safety import ApprovalQueue, AuditLog, ChangeProposal, ContentScreen
from .tools import ToolRouter, default_tools


@dataclass
class PipelineResult:
    request: str
    plan: Plan
    draft: Draft
    evaluation: Evaluation
    attempts: int
    revisions: int


@dataclass
class PracticeResult:
    case: dict[str, Any]
    result: PipelineResult | None = None
    report: LearningReport | None = None
    error: str = ""


class Orchestrator:
    def __init__(self, llm: LLM, state_dir: Path | str):
        state_dir = Path(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        self.store = ConfigStore(state_dir)
        self.audit = AuditLog(state_dir / "audit.jsonl")
        self.memory = MemoryStore(
            state_dir / "memory.json",
            self.store.current["memory_schema"]["version"])
        self.router = ToolRouter(default_tools(self.memory))
        self.screen = ContentScreen()
        self.planner = Planner(llm, self.router)
        self.responder = Responder(llm, self.router)
        self.evaluator = Evaluator(llm, self.screen)
        self.queue = ApprovalQueue(state_dir / "approvals.json", self.audit)
        self.learner = LearningController(llm, self.store, self.memory,
                                          self.queue, self.audit,
                                          self._score_case)
        self.output = OutputLayer()

    # Steps 1-2 and self-correction ------------------------------------------

    def handle(self, raw: str) -> Reply:
        config = self.store.current
        try:
            request = parse_input(raw, config)
        except InputRejected as exc:
            return self.output.rejected(str(exc))
        if self.screen.check_input(request.text, config):
            self.audit.record("input_blocked", request=request.text[:200])
            return self.output.rejected(
                "requests to change safety or approval controls go through "
                "the human approval queue, not chat")

        try:
            result = self.run_pipeline(request.text, config)
        except LLMRefusal as exc:
            self.memory.log_pattern("error", "refusal", str(exc))
            return self.output.rejected(str(exc))
        except LLMError as exc:
            self.memory.log_pattern("error", "llm_error", str(exc))
            return self.output.error(str(exc))

        episode = self.memory.add_episode(self._episode(result, config),
                                          config["memory_policy"])
        report, regenerated = None, False
        if config["learning"]["enabled"]:
            report = self.learner.after_episode(episode, result.evaluation)
            if report.with_status("kept"):
                result, regenerated = self._retry(result)

        withheld = self.screen.check_output(result.draft.answer,
                                            self.store.current) is not None
        return self.output.render(
            result, report, config["self_correction"]["min_confidence"],
            regenerated, withheld)

    def run_pipeline(self, request: str, config: dict[str, Any],
                     expected: dict[str, Any] | None = None) -> PipelineResult:
        """Plan, answer and self-correct, then grade.

        Self-correction sees only what it would see in real use. A known
        answer is checked once at the end, so it never leaks into a revision.
        """
        hints = self.router.route_hints(request, config)
        memories = self.memory.retrieve(request, config["memory_policy"])
        plan = self.planner.plan(request, hints, memories, config)
        policy = config["self_correction"]

        # Compare several candidates when the planner says it would help.
        n = policy["candidates"]
        if plan.complexity == "high":
            n = max(n, 2)
        attempts = []
        for _ in range(n):
            draft = self.responder.respond(request, plan, hints, memories,
                                           config)
            attempts.append((draft, self.evaluator.evaluate(
                request, draft, config)))
        best_draft, best_eval = max(attempts, key=lambda a: a[1].score)

        revisions = 0
        while revisions < policy["max_revisions"] and (
                not best_eval.passed
                or best_draft.confidence < policy["min_confidence"]):
            feedback = [i.detail for i in best_eval.issues] or [
                "Confidence is low; check the answer against the evidence."]
            draft = self.responder.respond(request, plan, hints, memories,
                                           config, feedback=feedback,
                                           previous=best_draft.answer)
            evaluation = self.evaluator.evaluate(request, draft, config)
            revisions += 1
            if evaluation.score >= best_eval.score:
                best_draft, best_eval = draft, evaluation

        if expected:
            best_eval = self.evaluator.check_expected(
                best_eval, best_draft.answer, expected, config)
        return PipelineResult(request, plan, best_draft, best_eval,
                              n + revisions, revisions)

    def _retry(self, result: PipelineResult) -> tuple[PipelineResult, bool]:
        """Re-answer under the improved config; keep whichever scores best."""
        config = self.store.current
        try:
            retry = self.run_pipeline(result.request, config)
        except (LLMError, BudgetExceeded):
            return result, False
        if retry.evaluation.score <= result.evaluation.score:
            return result, False
        self.memory.add_episode(self._episode(retry, config),
                                config["memory_policy"])
        return retry, True

    # Unattended learning ----------------------------------------------------

    def practice(self, limit: int,
                 cases: list[dict[str, Any]] | None = None
                 ) -> Iterator[PracticeResult]:
        """Answer questions with known answers and learn from the mistakes.

        Works through the least recently practised cases first, so repeated
        runs cover the whole set.
        """
        self.memory.add_practice_cases(PRACTICE_SET if cases is None
                                       else cases)
        for case in self.memory.next_practice_cases(limit):
            config = self.store.current
            try:
                result = self.run_pipeline(case["request"], config,
                                           case["expected"])
            except LLMError as exc:
                mode = "refusal" if isinstance(exc, LLMRefusal) \
                    else "llm_error"
                self.memory.log_pattern("error", mode, str(exc))
                self.memory.mark_practiced(case)
                yield PracticeResult(case, error=str(exc))
                continue
            self.memory.mark_practiced(case)
            episode = self.memory.add_episode(self._episode(result, config),
                                              config["memory_policy"])
            report = None
            if config["learning"]["enabled"]:
                report = self.learner.after_episode(episode,
                                                    result.evaluation)
            yield PracticeResult(case, result, report)
            if report and report.stopped:
                return

    def _score_case(self, case: dict[str, Any],
                    config: dict[str, Any]) -> float:
        result = self.run_pipeline(case["request"], config,
                                   case.get("expected"))
        return result.evaluation.score

    @staticmethod
    def _episode(result: PipelineResult,
                 config: dict[str, Any]) -> dict[str, Any]:
        ev = result.evaluation
        return {
            "request": result.request,
            "plan": result.plan.to_dict(),
            "answer": result.draft.answer,
            "confidence": result.draft.confidence,
            "score": round(ev.score, 3),
            "passed": ev.passed,
            "metrics": ev.to_dict()["metrics"],
            "issues": [i.__dict__ for i in ev.issues],
            "revisions": result.revisions,
            "tool_calls": [{"name": r.name, "input": r.input, "ok": r.ok,
                            "output": r.output}
                           for r in result.draft.tool_results],
            "config_version": config["version"],
        }

    # Human controls (called from the CLI only) -----------------------------

    def pending_approvals(self) -> list[ChangeProposal]:
        return self.queue.pending()

    def approve(self, proposal_id: str, approver: str,
                note: str = "") -> Outcome:
        self.queue.decide(proposal_id, True, approver, note)
        return self.learner.apply_approved(proposal_id)

    def reject(self, proposal_id: str, approver: str,
               note: str = "") -> ChangeProposal:
        proposal = self.queue.decide(proposal_id, False, approver, note)
        self.memory.log_pattern("error", "change_rejected",
                                signature(proposal))
        return proposal

    def rollback(self, version: int, approver: str) -> int:
        new_version = self.store.rollback(version, f"manual by {approver}")
        self.audit.record("manual_rollback", to_version=version,
                          new_version=new_version, by=approver)
        return new_version

    def benchmark(self) -> Iterator[tuple[str, float]]:
        """Score each regression case in turn, so a run that stops at the
        spending cap still shows the cases it finished."""
        config = self.store.current
        for case in self.memory.benchmarks:
            yield case["id"], self.learner.run_suite([case],
                                                     config).scores[case["id"]]
