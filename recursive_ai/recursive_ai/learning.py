"""Learning controller: the recursive improvement loop.

failure -> diagnose -> propose -> classify risk -> auto-apply (low) or queue
for a human (medium/high) -> retest on the regression suite -> keep, refine,
or roll back.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .config import ConfigError, ConfigStore, apply_change, prompt_for
from .evaluator import Evaluation
from .llm import LLM, LLMError
from .memory import MemoryStore
from .planner import sections
from .safety import (
    ApprovalQueue,
    AuditLog,
    ChangeProposal,
    Risk,
    RiskAssessment,
    classify_change,
)

PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "operation": {"type": "string",
                                  "enum": ["set", "append", "remove"]},
                    "value_json": {"type": "string"},
                    "rationale": {"type": "string"},
                    "risk": {"type": "string",
                             "enum": ["low", "medium", "high"]},
                },
                "required": ["target", "operation", "value_json",
                             "rationale", "risk"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["proposals"],
    "additionalProperties": False,
}

# Sections shown to the proposer. tool_permissions is shown so it knows what
# a routing rule may target; changing it is still a major change.
VISIBLE_SECTIONS = ("prompt_addenda", "routing_rules", "memory_policy",
                    "self_correction", "tool_permissions", "planner")

# (benchmark case, config) -> score in 0-1.
CaseRunner = Callable[[dict[str, Any], dict[str, Any]], float]

# Pattern modes recording changes that did not survive, so they are not
# proposed again.
FAILED_CHANGE_MODES = ("change_rolled_back", "change_rejected")


def signature(proposal: ChangeProposal) -> str:
    return json.dumps({"target": proposal.target,
                       "operation": proposal.operation,
                       "value": proposal.value}, sort_keys=True)


@dataclass
class SuiteResult:
    scores: dict[str, float]

    @property
    def mean(self) -> float:
        return sum(self.scores.values()) / len(self.scores) \
            if self.scores else 0.0


@dataclass
class Outcome:
    proposal_id: str
    target: str
    risk: str
    status: str
    detail: str = ""


@dataclass
class LearningReport:
    failure_modes: list[str] = field(default_factory=list)
    outcomes: list[Outcome] = field(default_factory=list)

    def with_status(self, *statuses: str) -> list[Outcome]:
        return [o for o in self.outcomes if o.status in statuses]


class LearningController:
    def __init__(self, llm: LLM, store: ConfigStore, memory: MemoryStore,
                 queue: ApprovalQueue, audit: AuditLog, run_case: CaseRunner):
        self.llm = llm
        self.store = store
        self.memory = memory
        self.queue = queue
        self.audit = audit
        self.run_case = run_case

    # Steps 3-8: evaluate outcome, find failure modes, propose, route ---------

    def after_episode(self, episode: dict[str, Any],
                      evaluation: Evaluation) -> LearningReport:
        report = LearningReport(failure_modes=evaluation.modes)
        if evaluation.passed:
            self.memory.log_pattern("success", "passed",
                                    episode["request"][:200], episode["id"])
            return report

        for issue in evaluation.issues:
            self.memory.log_pattern("error", issue.mode, issue.detail,
                                    episode["id"])
        config = self.store.current
        if not config["learning"]["enabled"]:
            return report

        # The failing request joins the regression suite, so any change it
        # motivates is retested against it.
        case = self.memory.add_benchmark(episode["request"], "failure")
        for proposal in self.propose(episode, evaluation, config):
            proposal.evidence = [episode["id"]]
            proposal.cases = [case["id"]]
            report.outcomes.append(self.route(proposal))
        return report

    def propose(self, episode: dict[str, Any], evaluation: Evaluation,
                config: dict[str, Any]) -> list[ChangeProposal]:
        limit = config["learning"]["max_proposals_per_episode"]
        visible = {k: config[k] for k in VISIBLE_SECTIONS}
        failed = self.failed_changes()
        prompt = sections(
            failed_request=episode["request"],
            answer_given=episode["answer"],
            failure_modes=", ".join(evaluation.modes),
            issues="\n".join(f"- [{i.mode}] {i.detail}"
                             for i in evaluation.issues),
            recurring_failures=json.dumps(
                {mode: n for mode, n
                 in self.memory.pattern_counts("error").most_common(10)
                 if mode not in FAILED_CHANGE_MODES}),
            current_config=json.dumps(visible, indent=2),
            changes_already_tried_and_failed="\n".join(sorted(failed)[-10:]),
            limits=f"Propose at most {limit} changes. Targets are dotted "
                   "config paths, e.g. prompt_addenda.responder or "
                   "memory_policy.retrieval_k. Return an empty list if no "
                   "config change would help.",
        )
        try:
            response = self.llm.generate(
                purpose="propose", system=prompt_for(config, "learner"),
                messages=[{"role": "user", "content": prompt}],
                schema=PROPOSAL_SCHEMA)
            raw = json.loads(response.text)["proposals"]
        except (LLMError, ValueError, KeyError, TypeError) as exc:
            self.audit.record("proposal_failed", error=str(exc))
            return []
        waiting = {signature(p) for p in self.queue.pending()}
        proposals = []
        for item in raw[:limit]:
            proposal = self._parse(item)
            if proposal is None:
                continue
            if signature(proposal) in failed | waiting:
                self.audit.record("proposal_skipped_repeat",
                                  change=signature(proposal))
            else:
                proposals.append(proposal)
        return proposals

    def failed_changes(self) -> set[str]:
        return {p["detail"] for p in self.memory.data["patterns"]["error"]
                if p["mode"] in FAILED_CHANGE_MODES}

    def _parse(self, raw: dict[str, Any]) -> ChangeProposal | None:
        try:
            value = json.loads(raw["value_json"])
            return ChangeProposal(target=str(raw["target"]),
                                  operation=str(raw["operation"]),
                                  value=value,
                                  rationale=str(raw.get("rationale", "")),
                                  proposed_risk=str(raw.get("risk", "low")))
        except (ValueError, KeyError, TypeError) as exc:
            self.audit.record("proposal_invalid", raw=raw, error=str(exc))
            return None

    def _assess(self, proposal: ChangeProposal,
                config: dict[str, Any]) -> RiskAssessment:
        assessment = classify_change(proposal, config)
        proposal.risk = assessment.risk.name.lower()
        proposal.risk_reasons = assessment.reasons
        return assessment

    @staticmethod
    def _may_auto_apply(assessment: RiskAssessment,
                        config: dict[str, Any]) -> bool:
        ceiling = config["learning"]["auto_apply_max_risk"]
        if ceiling == "none" or assessment.locked:
            return False
        # HIGH never auto-applies, whatever the config says.
        return assessment.risk < Risk.HIGH \
            and assessment.risk <= Risk.parse(ceiling)

    def route(self, proposal: ChangeProposal) -> Outcome:
        config = self.store.current
        assessment = self._assess(proposal, config)
        self.audit.record("proposed", proposal=proposal.id,
                          target=proposal.target, operation=proposal.operation,
                          value=proposal.value, risk=proposal.risk,
                          reasons=assessment.reasons,
                          claimed_risk=proposal.proposed_risk)

        def outcome(status: str, detail: str) -> Outcome:
            return Outcome(proposal.id, proposal.target, proposal.risk,
                           status, detail)

        if assessment.locked:
            proposal.note("rejected", reason="locked section")
            self.audit.record("rejected_locked", proposal=proposal.id)
            return outcome("rejected", "; ".join(assessment.reasons))
        try:
            apply_change(config, proposal.target, proposal.operation,
                         proposal.value)
        except ConfigError as exc:
            proposal.note("invalid", error=str(exc))
            self.audit.record("proposal_invalid", proposal=proposal.id,
                              error=str(exc))
            return outcome("invalid", str(exc))

        if self._may_auto_apply(assessment, config):
            return self.apply_and_validate(proposal, approved=False)

        self.queue.submit(proposal)
        return outcome("awaiting_approval", "; ".join(assessment.reasons))

    # Steps 9-10: retest, then keep / refine / roll back ---------------------

    def apply_approved(self, proposal_id: str) -> Outcome:
        """Apply a change a human approved. Only the CLI calls this."""
        proposal = self.queue.get(proposal_id)
        if proposal.status != "approved":
            raise ValueError(f"{proposal_id} is {proposal.status}, "
                             "not approved")
        if self._assess(proposal, self.store.current).locked:
            proposal.note("rejected", reason="locked section")
            self.queue.update(proposal)
            return Outcome(proposal.id, proposal.target, proposal.risk,
                           "rejected", "locked section")
        outcome = self.apply_and_validate(proposal, approved=True)
        self.queue.update(proposal)
        return outcome

    def run_suite(self, cases: list[dict[str, Any]],
                  config: dict[str, Any]) -> SuiteResult:
        scores = {}
        for case in cases:
            try:
                scores[case["id"]] = round(self.run_case(case, config), 3)
            except LLMError as exc:
                self.audit.record("benchmark_error", case=case["id"],
                                  error=str(exc))
                scores[case["id"]] = 0.0
        return SuiteResult(scores)

    @staticmethod
    def verdict(baseline: SuiteResult, candidate: SuiteResult,
                policy: dict[str, Any], approved: bool) -> tuple[str, str]:
        tolerance = policy["regression_tolerance"]
        regressed = [cid for cid, score in baseline.scores.items()
                     if candidate.scores.get(cid, 0.0) < score - tolerance]
        if regressed:
            return "rollback", f"regressed on {', '.join(regressed)}"
        gain = candidate.mean - baseline.mean
        summary = f"suite score {baseline.mean:.2f} -> {candidate.mean:.2f}"
        # A human already judged the benefit; the retest only guards
        # against breakage.
        if approved or gain >= policy["min_improvement"]:
            return "keep", summary
        return "refine", f"no measurable improvement ({summary})"

    def apply_and_validate(self, proposal: ChangeProposal,
                           approved: bool) -> Outcome:
        config = self.store.current
        policy = config["learning"]
        cases = self.memory.select_benchmarks(proposal.cases,
                                              policy["max_benchmark_cases"])
        baseline = self.run_suite(cases, config)
        refinements = 0

        while True:
            try:
                candidate_config = apply_change(config, proposal.target,
                                                proposal.operation,
                                                proposal.value)
            except ConfigError as exc:
                proposal.note("failed", error=str(exc))
                self.audit.record("apply_failed", proposal=proposal.id,
                                  error=str(exc))
                return Outcome(proposal.id, proposal.target, proposal.risk,
                               "failed", str(exc))

            how = "approved" if approved else "auto-applied"
            version = self.store.commit(
                candidate_config, f"{how} {proposal.id}: {proposal.target}")
            self.audit.record("applied", proposal=proposal.id,
                              version=version, risk=proposal.risk, how=how)

            candidate = self.run_suite(cases, self.store.current)
            decision, detail = self.verdict(baseline, candidate, policy,
                                            approved)
            self.audit.record("retested", proposal=proposal.id,
                              baseline=baseline.scores,
                              candidate=candidate.scores, decision=decision)

            if decision == "keep":
                proposal.note("kept", version=version, detail=detail)
                self.memory.log_pattern("success", "change_kept",
                                        signature(proposal))
                return Outcome(proposal.id, proposal.target, proposal.risk,
                               "kept", detail)

            self.store.rollback(config["version"], f"{proposal.id}: {detail}")
            self.audit.record("rolled_back", proposal=proposal.id,
                              to_version=config["version"], reason=detail)

            if decision != "refine" or approved \
                    or refinements >= policy["max_refinements"]:
                break
            refined = self.refine(proposal, detail, config)
            if refined is None:
                break
            refinements += 1
            assessment = self._assess(refined, config)
            if not self._may_auto_apply(assessment, config):
                refined.cases = proposal.cases
                refined.evidence = proposal.evidence
                self.queue.submit(refined)
                proposal.note("superseded", by=refined.id)
                return Outcome(refined.id, refined.target, refined.risk,
                               "awaiting_approval",
                               "refinement needs approval: "
                               + "; ".join(assessment.reasons))
            proposal.note("refined", target=refined.target,
                          value=refined.value)
            proposal.target, proposal.operation = (refined.target,
                                                   refined.operation)
            proposal.value, proposal.risk = refined.value, refined.risk

        proposal.note("rolled_back", detail=detail)
        self.memory.log_pattern("error", "change_rolled_back",
                                signature(proposal))
        return Outcome(proposal.id, proposal.target, proposal.risk,
                       "rolled_back", detail)

    def refine(self, proposal: ChangeProposal, result: str,
               config: dict[str, Any]) -> ChangeProposal | None:
        prompt = sections(
            change_tried=json.dumps({"target": proposal.target,
                                     "operation": proposal.operation,
                                     "value": proposal.value}),
            rationale=proposal.rationale,
            retest_result=result,
            current_config=json.dumps(
                {k: config[k] for k in VISIBLE_SECTIONS}, indent=2),
            limits="Return at most one revised change that is more likely "
                   "to help, or an empty list.",
        )
        try:
            response = self.llm.generate(
                purpose="refine", system=prompt_for(config, "learner"),
                messages=[{"role": "user", "content": prompt}],
                schema=PROPOSAL_SCHEMA)
            raw = json.loads(response.text)["proposals"]
        except (LLMError, ValueError, KeyError, TypeError):
            return None
        return self._parse(raw[0]) if raw else None
