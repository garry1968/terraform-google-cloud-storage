"""Versioned system configuration.

Everything the learning loop is allowed to change lives in one JSON document.
Every change is committed as a new numbered snapshot, so any change can be
diffed, audited, and rolled back. History is append-only: a rollback is
itself a new version that copies an older snapshot.
"""

from __future__ import annotations

import copy
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "version": 1,
    # Base prompts define core behaviour. Rewriting one is a major change.
    "prompts": {
        "planner": (
            "You are the planning component of an assistant. Break the user's "
            "request into the smallest set of concrete steps that produces a "
            "correct, well-supported answer. Name a tool for a step only when "
            "it materially improves accuracy, and only from the tools listed. "
            "Rate complexity 'high' only when several approaches are plausible "
            "and comparing candidate answers would help."
        ),
        "responder": (
            "You are a careful assistant. Follow the plan, use tools when they "
            "make the answer more reliable, and answer the request directly. "
            "Report a confidence between 0 and 1 that reflects how well "
            "supported the answer is; lower it when you rely on recall you "
            "cannot verify. List the evidence the answer rests on. When "
            "revising an earlier answer, fix the listed issues and keep what "
            "was right."
        ),
        "evaluator": (
            "You grade an assistant's answer. Score correctness, completeness "
            "and relevance from 0 to 1. Judge correctness against the request "
            "and any tool results shown; do not reward a confident tone. List "
            "each concrete problem as a short issue, or an empty list if there "
            "are none."
        ),
        "learner": (
            "You improve an AI assistant's configuration from evidence of its "
            "failures. Propose a few small, targeted changes that would have "
            "prevented the failures shown. Prefer the low-risk channels: "
            "appending a short guidance line to prompt_addenda.<component>, "
            "appending a routing rule for an already-permitted tool, or "
            "nudging a memory_policy or self_correction number within its "
            "bounds. Never propose changes that weaken safety or oversight. "
            "Give each value as JSON in value_json and an honest risk: changes "
            "to safety_policy, tool_permissions, memory_schema, planner, "
            "prompts or evaluator are high risk and go to a human."
        ),
    },
    # Short learned guidance lines appended to a base prompt. This is the
    # main low-risk channel for automatic improvement.
    "prompt_addenda": {"planner": [], "responder": [], "evaluator": []},
    # Regex -> tool hints. Hints only suggest a tool; permissions still apply.
    "routing_rules": [
        {
            "id": "arithmetic",
            "pattern": r"\d+(\.\d+)?\s*([+*/x%^]|\s-\s)\s*\d+",
            "tool": "calculator",
            "hint": "Use the calculator for arithmetic instead of mental math.",
        },
        {
            "id": "word-count",
            "pattern": r"\b(count|how many)\b.*\b(words|characters)\b",
            "tool": "word_count",
            "hint": "Use word_count to count words or characters exactly.",
        },
        {
            "id": "clock",
            "pattern": r"\b(what time|today's date|current date|current time)\b",
            "tool": "clock",
            "hint": "Use the clock for the current date or time.",
        },
    ],
    "memory_policy": {
        "retrieval_k": 3,
        "max_episodes": 500,
        "success_threshold": 0.8,
    },
    "memory_schema": {
        "version": 1,
        "episode_fields": [
            "id", "ts", "request", "plan", "answer", "confidence", "score",
            "passed", "metrics", "issues", "revisions", "tool_calls",
            "config_version",
        ],
    },
    "self_correction": {
        "max_revisions": 2,
        "candidates": 1,
        "min_confidence": 0.6,
    },
    "safety_policy": {
        "max_input_chars": 8000,
        "blocked_input_patterns": [
            r"(disable|turn off|bypass|skip)\W+(\w+\W+){0,3}"
            r"(approval|safety|oversight)",
        ],
        "blocked_output_patterns": [
            r"sk-ant-[A-Za-z0-9_-]{8,}",
            r"(?i)api[_-]?key\s*[:=]\s*\S{12,}",
        ],
    },
    # Default deny: a tool missing from this map cannot run.
    "tool_permissions": {
        "calculator": "allow",
        "word_count": "allow",
        "clock": "allow",
        "memory_search": "allow",
    },
    "planner": {"max_steps": 5, "max_tool_rounds": 6},
    "evaluator": {
        "pass_threshold": 0.7,
        "weights": {
            "correctness": 0.35,
            "completeness": 0.15,
            "relevance": 0.15,
            "confidence": 0.10,
            "tool_reliability": 0.10,
            "grounding": 0.15,
            "expected": 0.50,
        },
    },
    # Locked: the learning loop can never propose changes here.
    "learning": {
        "enabled": True,
        "auto_apply_max_risk": "low",
        "min_improvement": 0.01,
        "regression_tolerance": 0.05,
        "max_refinements": 1,
        "max_benchmark_cases": 5,
        "max_proposals_per_episode": 2,
    },
}


class ConfigError(ValueError):
    """A change or a config document is invalid."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_path(config: dict[str, Any], path: str) -> Any:
    node: Any = config
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            raise ConfigError(f"unknown config path: {path}")
        node = node[key]
    return node


def apply_change(
    config: dict[str, Any], target: str, operation: str, value: Any
) -> dict[str, Any]:
    """Return a new config with one change applied. The input is not mutated.

    operation is "set" (replace a value), "append" (add to a list) or
    "remove" (drop a list item, matched by its "id" for dict items).
    """
    new = copy.deepcopy(config)
    *parents, leaf = target.split(".")
    node: Any = new
    for key in parents:
        if not isinstance(node, dict) or key not in node:
            raise ConfigError(f"unknown config path: {target}")
        node = node[key]
    if not isinstance(node, dict) or leaf not in node:
        raise ConfigError(f"unknown config path: {target}")

    if operation == "set":
        current = node[leaf]
        if isinstance(current, (dict, list)) != isinstance(value, (dict, list)):
            raise ConfigError(f"type mismatch setting {target}")
        node[leaf] = value
    elif operation in ("append", "remove"):
        items = node[leaf]
        if not isinstance(items, list):
            raise ConfigError(f"{target} is not a list")
        if operation == "append":
            if isinstance(value, dict) and "id" in value:
                if any(isinstance(i, dict) and i.get("id") == value["id"]
                       for i in items):
                    raise ConfigError(f"duplicate id {value['id']!r}")
            if value in items:
                raise ConfigError(f"{target} already contains this value")
            items.append(value)
        else:
            key = value.get("id") if isinstance(value, dict) else value
            kept = [i for i in items
                    if (i.get("id") if isinstance(i, dict) else i) != key]
            if len(kept) == len(items):
                raise ConfigError(f"{target} has no item {key!r}")
            node[leaf] = kept
    else:
        raise ConfigError(f"unknown operation: {operation}")

    validate_config(new)
    return new


def prompt_for(config: dict[str, Any], component: str) -> str:
    """Base prompt for a component plus any learned guidance lines."""
    prompt = config["prompts"][component]
    addenda = config["prompt_addenda"].get(component, [])
    if addenda:
        lines = "\n".join(f"- {line}" for line in addenda)
        prompt += f"\n\nLearned guidance:\n{lines}"
    return prompt


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def validate_config(config: dict[str, Any]) -> None:
    """Structural and bounds checks. Raises ConfigError on the first problem."""
    for section in DEFAULT_CONFIG:
        _require(section in config, f"missing section: {section}")

    for name, text in config["prompts"].items():
        _require(isinstance(text, str) and text.strip() != "",
                 f"prompts.{name} must be non-empty text")
    for name, lines in config["prompt_addenda"].items():
        _require(isinstance(lines, list) and len(lines) <= 20,
                 f"prompt_addenda.{name} must be a list of at most 20 lines")
        _require(all(isinstance(x, str) and 0 < len(x) <= 500 for x in lines),
                 f"prompt_addenda.{name} lines must be 1-500 chars")

    for rule in config["routing_rules"]:
        _require(isinstance(rule, dict)
                 and {"id", "pattern", "tool", "hint"} <= rule.keys(),
                 "routing rules need id, pattern, tool and hint")
        try:
            re.compile(rule["pattern"])
        except re.error as exc:
            raise ConfigError(f"routing rule {rule['id']}: bad regex: {exc}")

    mp = config["memory_policy"]
    _require(isinstance(mp["retrieval_k"], int) and 0 <= mp["retrieval_k"] <= 20,
             "memory_policy.retrieval_k must be an int in 0-20")
    _require(isinstance(mp["max_episodes"], int) and mp["max_episodes"] >= 10,
             "memory_policy.max_episodes must be an int >= 10")
    _require(0 <= mp["success_threshold"] <= 1,
             "memory_policy.success_threshold must be in 0-1")

    sc = config["self_correction"]
    _require(isinstance(sc["max_revisions"], int) and 0 <= sc["max_revisions"] <= 5,
             "self_correction.max_revisions must be an int in 0-5")
    _require(isinstance(sc["candidates"], int) and 1 <= sc["candidates"] <= 5,
             "self_correction.candidates must be an int in 1-5")
    _require(0 <= sc["min_confidence"] <= 1,
             "self_correction.min_confidence must be in 0-1")

    for tool, permission in config["tool_permissions"].items():
        _require(permission in ("allow", "deny"),
                 f"tool_permissions.{tool} must be 'allow' or 'deny'")

    _require(0 <= config["evaluator"]["pass_threshold"] <= 1,
             "evaluator.pass_threshold must be in 0-1")
    _require(config["learning"]["auto_apply_max_risk"] in ("none", "low", "medium"),
             "learning.auto_apply_max_risk must be none, low or medium")


class ConfigStore:
    """Persists the current config plus a numbered snapshot per version."""

    def __init__(self, state_dir: Path):
        self.path = state_dir / "config.json"
        self.history_dir = state_dir / "config_history"
        self.history_dir.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._config = json.loads(self.path.read_text())
            validate_config(self._config)
        else:
            self._config = copy.deepcopy(DEFAULT_CONFIG)
            self._write(self._config, "initial default config")

    @property
    def current(self) -> dict[str, Any]:
        return copy.deepcopy(self._config)

    @property
    def version(self) -> int:
        return self._config["version"]

    def commit(self, config: dict[str, Any], reason: str) -> int:
        validate_config(config)
        config = copy.deepcopy(config)
        config["version"] = self._latest_version() + 1
        self._write(config, reason)
        return config["version"]

    def rollback(self, version: int, reason: str) -> int:
        snapshot = self._snapshot_path(version)
        if not snapshot.exists():
            raise ConfigError(f"no snapshot for version {version}")
        old = json.loads(snapshot.read_text())["config"]
        return self.commit(old, f"rollback to v{version}: {reason}")

    def history(self) -> list[dict[str, Any]]:
        entries = []
        for path in sorted(self.history_dir.glob("v*.json"),
                           key=lambda p: int(p.stem[1:])):
            data = json.loads(path.read_text())
            entries.append({k: data[k] for k in ("version", "ts", "reason")})
        return entries

    def _latest_version(self) -> int:
        versions = [int(p.stem[1:]) for p in self.history_dir.glob("v*.json")]
        return max(versions, default=0)

    def _snapshot_path(self, version: int) -> Path:
        return self.history_dir / f"v{version}.json"

    def _write(self, config: dict[str, Any], reason: str) -> None:
        record = {"version": config["version"], "ts": _now(),
                  "reason": reason, "config": config}
        self._snapshot_path(config["version"]).write_text(
            json.dumps(record, indent=2))
        self.path.write_text(json.dumps(config, indent=2))
        self._config = config
