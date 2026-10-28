"""Output layer: renders the final answer with confidence and learning notes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .learning import LearningReport
    from .orchestrator import PipelineResult

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
        if report and report.outcomes:
            lines.append("Learning:")
            for o in report.outcomes:
                lines.append(f"  {_LABELS.get(o.status, o.status)} {o.proposal_id}"
                             f" ({o.risk}) {o.target} - {o.detail}")
            if report.with_status("awaiting_approval"):
                lines.append("  Review with: python -m recursive_ai approvals")
        return Reply("\n".join(lines), answer, True, draft.confidence,
                     ev.score, ev.passed, report)

    @staticmethod
    def rejected(reason: str) -> Reply:
        return Reply(f"Request not processed: {reason}", "", ok=False)

    @staticmethod
    def error(reason: str) -> Reply:
        return Reply(f"Something went wrong: {reason}", "", ok=False)
