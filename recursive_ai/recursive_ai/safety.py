"""Safety layer: content screening, change risk classification, the human
approval queue, and the audit log.

The risk rules live in code, not in the config, so the learning loop cannot
edit the rules that decide whether its own changes need a human.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any

from .config import ConfigError, get_path


class Risk(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2

    @classmethod
    def parse(cls, value: str | None) -> Risk:
        try:
            return cls[str(value).upper()]
        except KeyError:
            return cls.LOW


# Sections the learning loop may never touch, even with approval. Only a
# human editing config.json directly can change them.
LOCKED_SECTIONS = {"learning", "version"}

# Major changes: always need human approval.
MAJOR_SECTIONS = {
    "safety_policy": "safety policy change",
    "tool_permissions": "tool permission change",
    "memory_schema": "memory schema change",
    "planner": "core planner change",
    "prompts": "base prompt rewrite (core behaviour)",
    "evaluator": "evaluation metric change (could make self-grading lenient)",
}

# Numeric knobs that may auto-apply when they stay inside these bounds and
# move by at most max_step: path -> (min, max, max_step).
SAFE_NUMERIC_BOUNDS = {
    "memory_policy.retrieval_k": (1, 8, 2),
    "memory_policy.success_threshold": (0.5, 0.95, 0.1),
    "memory_policy.max_episodes": (100, 2000, 500),
    "self_correction.max_revisions": (0, 3, 1),
    "self_correction.candidates": (1, 3, 1),
    "self_correction.min_confidence": (0.4, 0.9, 0.1),
}

# Text that tries to steer around oversight escalates any change to HIGH.
SUSPICIOUS_TEXT = re.compile(
    r"ignore (all |any )?(previous|prior|above)|disregard|bypass|jailbreak|"
    r"disable|override|without (approval|permission)|system prompt|"
    r"safety|permission|approv|oversight|audit",
    re.IGNORECASE,
)

MAX_LOW_RISK_TEXT = 300
MAX_LOW_RISK_ADDENDA = 10


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ChangeProposal:
    target: str
    operation: str
    value: Any
    rationale: str
    proposed_risk: str = "low"
    evidence: list[str] = field(default_factory=list)  # episode ids
    cases: list[str] = field(default_factory=list)  # benchmark ids to retest
    id: str = field(default_factory=lambda: "C-" + uuid.uuid4().hex[:8])
    risk: str = ""
    risk_reasons: list[str] = field(default_factory=list)
    status: str = "proposed"
    created: str = field(default_factory=_now)
    history: list[dict[str, Any]] = field(default_factory=list)

    def note(self, status: str, **detail: Any) -> None:
        self.status = status
        self.history.append({"ts": _now(), "status": status, **detail})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChangeProposal:
        return cls(**data)


@dataclass
class RiskAssessment:
    risk: Risk
    reasons: list[str]
    locked: bool = False


def _text_of(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def classify_change(proposal: ChangeProposal,
                    config: dict[str, Any]) -> RiskAssessment:
    """Deterministic, fail-closed risk rules.

    The proposer's own risk estimate can raise the result but never lower it.
    """
    section = proposal.target.split(".")[0]
    reasons: list[str] = []

    if section in LOCKED_SECTIONS:
        return RiskAssessment(Risk.HIGH, [f"'{section}' is locked: only a "
                                          "human can edit it directly"],
                              locked=True)

    if section in MAJOR_SECTIONS:
        risk = Risk.HIGH
        reasons.append(MAJOR_SECTIONS[section])
    elif section == "prompt_addenda":
        risk, reasons = _classify_addendum(proposal, config)
    elif section == "routing_rules":
        risk, reasons = _classify_routing(proposal, config)
    elif proposal.target in SAFE_NUMERIC_BOUNDS:
        risk, reasons = _classify_numeric(proposal, config)
    elif section in ("memory_policy", "self_correction"):
        risk = Risk.MEDIUM
        reasons.append(f"{section} change outside the auto-apply list")
    else:
        risk = Risk.HIGH
        reasons.append(f"unknown target '{proposal.target}' (fail closed)")

    if risk < Risk.HIGH and SUSPICIOUS_TEXT.search(_text_of(proposal.value)):
        risk = Risk.HIGH
        reasons.append("value mentions safety/approval/permissions")

    claimed = Risk.parse(proposal.proposed_risk)
    if claimed > risk:
        risk = claimed
        reasons.append(f"proposer rated it {claimed.name.lower()}")

    return RiskAssessment(risk, reasons)


def _classify_addendum(proposal, config):
    if proposal.operation == "remove":
        return Risk.LOW, ["removes a learned guidance line"]
    if proposal.operation != "append":
        return Risk.MEDIUM, ["replaces a whole guidance list"]
    if not isinstance(proposal.value, str):
        return Risk.HIGH, ["guidance must be plain text"]
    if len(proposal.value) > MAX_LOW_RISK_TEXT:
        return Risk.MEDIUM, ["long guidance text"]
    try:
        existing = get_path(config, proposal.target)
    except ConfigError:
        return Risk.HIGH, [f"unknown target '{proposal.target}'"]
    if len(existing) >= MAX_LOW_RISK_ADDENDA:
        return Risk.MEDIUM, ["many learned lines already; needs review"]
    return Risk.LOW, ["appends a short guidance line"]


def _classify_routing(proposal, config):
    if proposal.target != "routing_rules":
        return Risk.MEDIUM, ["edits inside a routing rule"]
    if proposal.operation != "append":
        return Risk.MEDIUM, ["removes or replaces routing rules"]
    rule = proposal.value
    if not isinstance(rule, dict):
        return Risk.HIGH, ["routing rule must be an object"]
    if config["tool_permissions"].get(rule.get("tool")) != "allow":
        return Risk.HIGH, [f"routes to a tool that is not permitted: "
                           f"{rule.get('tool')!r}"]
    try:
        re.compile(str(rule.get("pattern", "")))
    except re.error:
        return Risk.MEDIUM, ["invalid regex"]
    return Risk.LOW, ["adds a hint for an already-permitted tool"]


def _classify_numeric(proposal, config):
    low, high, max_step = SAFE_NUMERIC_BOUNDS[proposal.target]
    value = proposal.value
    if proposal.operation != "set" or isinstance(value, bool) \
            or not isinstance(value, (int, float)):
        return Risk.MEDIUM, ["non-numeric change to a numeric setting"]
    current = get_path(config, proposal.target)
    if not low <= value <= high:
        return Risk.MEDIUM, [f"outside safe bounds {low}-{high}"]
    if abs(value - current) > max_step:
        return Risk.MEDIUM, [f"step larger than {max_step}"]
    return Risk.LOW, ["small numeric tune within safe bounds"]


class ContentScreen:
    """Regex screen for inputs and outputs, driven by safety_policy."""

    @staticmethod
    def _match(patterns: list[str], text: str) -> str | None:
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return pattern
        return None

    def check_input(self, text: str, config: dict[str, Any]) -> str | None:
        return self._match(config["safety_policy"]["blocked_input_patterns"],
                           text)

    def check_output(self, text: str, config: dict[str, Any]) -> str | None:
        return self._match(config["safety_policy"]["blocked_output_patterns"],
                           text)


class AuditLog:
    """Append-only JSONL record of every learning and approval event."""

    def __init__(self, path: Path):
        self.path = path

    def record(self, event: str, **data: Any) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps({"ts": _now(), "event": event, **data},
                               default=str) + "\n")

    def tail(self, n: int = 20) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text().splitlines()[-n:]
        return [json.loads(line) for line in lines]


class ApprovalQueue:
    """Medium/high-risk proposals wait here for a human decision.

    Nothing in the automated loop calls approve(); only the CLI does.
    """

    def __init__(self, path: Path, audit: AuditLog):
        self.path = path
        self.audit = audit
        self._items: dict[str, ChangeProposal] = {}
        if path.exists():
            for data in json.loads(path.read_text()):
                proposal = ChangeProposal.from_dict(data)
                self._items[proposal.id] = proposal

    def submit(self, proposal: ChangeProposal) -> None:
        proposal.note("awaiting_approval", risk=proposal.risk)
        self._items[proposal.id] = proposal
        self._save()
        self.audit.record("queued_for_approval", proposal=proposal.id,
                          target=proposal.target, risk=proposal.risk)

    def pending(self) -> list[ChangeProposal]:
        return [p for p in self._items.values()
                if p.status == "awaiting_approval"]

    def get(self, proposal_id: str) -> ChangeProposal:
        if proposal_id not in self._items:
            raise KeyError(f"no proposal {proposal_id}")
        return self._items[proposal_id]

    def decide(self, proposal_id: str, approved: bool, approver: str,
               note: str = "") -> ChangeProposal:
        proposal = self.get(proposal_id)
        if proposal.status != "awaiting_approval":
            raise ValueError(f"{proposal_id} is {proposal.status}, "
                             "not awaiting approval")
        status = "approved" if approved else "rejected"
        proposal.note(status, by=approver, note=note)
        self._save()
        self.audit.record(status, proposal=proposal_id, by=approver,
                          note=note)
        return proposal

    def update(self, proposal: ChangeProposal) -> None:
        self._items[proposal.id] = proposal
        self._save()

    def _save(self) -> None:
        self.path.write_text(json.dumps(
            [p.to_dict() for p in self._items.values()], indent=2,
            default=str))
