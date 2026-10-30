"""Output layer: renders the final answer with confidence and learning notes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .learning import LearningReport
    from .orchestrator import PipelineResult, PracticeResult

_LABELS = {
    "kept": "[kept]",
    "rolled_back": "[rolled back]",
    "awaiting_approval": "[needs your approval]",
    "rejected": "[rejected]",
    "invalid": "[invalid]",
    "failed": "[failed]",
}


@dataclass
class Reply:
    text: str
    answer: str
    ok: bool = True
    confidence: float | None = None
    score: float | None = None
    passed: bool | None = None
    learning: LearningReport | None = None


class OutputLayer:
    def render(self, result: PipelineResult,
               report: LearningReport | None = None,
               min_confidence: float = 0.6, regenerated: bool = False,
               withheld: bool = False) -> Reply:
        draft, ev = result.draft, result.evaluation
        answer = ("[answer withheld: it matched the output safety policy]"
                  if withheld else draft.answer)
        lines = [answer, "",
                 f"(confidence {draft.confidence:.2f} | quality "
                 f"{ev.score:.2f} | {'passed' if ev.passed else 'not passed'})"]
        if not ev.passed or draft.confidence < min_confidence:
            lines.append("Caution: low confidence or failed checks - verify "
                         "this answer.")
        if result.revisions:
            lines.append(f"Self-correction: revised {result.revisions} "
                         f"time(s).")
        if regenerated:
            lines.append("Answer regenerated after a kept improvement.")
        if report and (report.outcomes or report.stopped):
            lines.append("Learning:")
            lines += [f"  {line}" for line in self.learning_lines(report)]
            if report.with_status("awaiting_approval"):
                lines.append("  Review with: python -m recursive_ai approvals")
        return Reply("\n".join(lines), answer, True, draft.confidence,
                     ev.score, ev.passed, report)

    @staticmethod
    def learning_lines(report: LearningReport) -> list[str]:
        lines = [f"{_LABELS.get(o.status, o.status)} {o.proposal_id}"
                 f" ({o.risk}) {o.target} - {o.detail}"
                 for o in report.outcomes]
        if report.stopped:
            lines.append(f"[stopped] {report.stopped}")
        return lines

    def practice_lines(self, number: int, item: PracticeResult) -> list[str]:
        case_id = item.case["id"]
        if item.result is None:
            return [f"{number}. {case_id}  error - {item.error}"]
        ev = item.result.evaluation
        lines = [f"{number}. {case_id}  {'passed' if ev.passed else 'FAILED'}"
                 f"  (quality {ev.score:.2f})"]
        if not ev.passed:
            expected = item.case["expected"]
            wanted = expected.get("terms", []) + expected.get("contains", [])
            answer = " ".join(item.result.draft.answer.split())
            lines.append(f"   wanted: {', '.join(wanted)}  |  answered: "
                         f"{answer[:100]}")
        if item.report:
            lines += [f"   {line}"
                      for line in self.learning_lines(item.report)]
        return lines

    @staticmethod
    def practice_summary(items: list[PracticeResult]) -> list[str]:
        answered = [i.result for i in items if i.result]
        passed = sum(r.evaluation.passed for r in answered)
        outcomes = [o for i in items if i.report for o in i.report.outcomes]

        def count(status: str) -> int:
            return sum(o.status == status for o in outcomes)

        lines = [f"Practised {len(items)}: {passed} passed, "
                 f"{len(answered) - passed} failed, "
                 f"{len(items) - len(answered)} errors.",
                 f"Changes: {count('kept')} kept, {count('rolled_back')} "
                 f"rolled back, {count('awaiting_approval')} waiting for "
                 "your approval."]
        if count("awaiting_approval"):
            lines.append("Review them with: python -m recursive_ai approvals")
        return lines

    @staticmethod
    def rejected(reason: str) -> Reply:
        return Reply(f"Request not processed: {reason}", "", ok=False)

    @staticmethod
    def error(reason: str) -> Reply:
        return Reply(f"Something went wrong: {reason}", "", ok=False)
