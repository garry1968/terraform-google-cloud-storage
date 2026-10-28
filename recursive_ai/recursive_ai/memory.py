"""Memory: past episodes, success/error patterns, and the regression suite.

The on-disk schema is versioned. If the stored schema does not match the
config's memory_schema.version, loading stops: migrating memory is a major
change that needs a human.
"""

from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STOPWORDS = {
    "the", "and", "for", "are", "was", "what", "how", "why", "who", "this",
    "that", "with", "from", "you", "your", "can", "does", "into", "about",
    "is", "of", "to", "in", "a", "an", "it", "be", "on", "or", "me",
}

SEED_BENCHMARKS = [
    {"id": "seed-math", "request": "What is 17 * 23?",
     "expected": {"contains": ["391"]}, "source": "seed"},
    {"id": "seed-words",
     "request": "How many words are in this sentence: the quick brown fox "
                "jumps",
     "expected": {"contains": ["5"]}, "source": "seed"},
    {"id": "seed-explain",
     "request": "In one sentence, what is a cache in computing?",
     "expected": None, "source": "seed"},
]


class MemorySchemaError(RuntimeError):
    """Stored memory uses a schema this config does not accept."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower())
            if len(w) > 1 and w not in STOPWORDS}


class MemoryStore:
    def __init__(self, path: Path, schema_version: int):
        self.path = path
        if path.exists():
            data = json.loads(path.read_text())
            if data.get("schema_version") != schema_version:
                raise MemorySchemaError(
                    f"memory file is schema v{data.get('schema_version')} but "
                    f"config expects v{schema_version}; a human-approved "
                    "migration is required")
            self.data = data
        else:
            self.data = {
                "schema_version": schema_version,
                "episodes": [],
                "patterns": {"success": [], "error": []},
                "benchmarks": [dict(b) for b in SEED_BENCHMARKS],
            }
            self.save()

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2, default=str))

    # Episodes -------------------------------------------------------------

    def add_episode(self, episode: dict[str, Any],
                    policy: dict[str, Any]) -> dict[str, Any]:
        episode = {"id": "E-" + uuid.uuid4().hex[:8], "ts": _now(), **episode}
        episodes = self.data["episodes"]
        episodes.append(episode)
        del episodes[:-policy["max_episodes"]]
        self.save()
        return episode

    def retrieve(self, text: str, policy: dict[str, Any]) -> list[dict]:
        """Top-k past successful episodes by keyword overlap."""
        query = tokens(text)
        if not query or policy["retrieval_k"] == 0:
            return []
        scored = []
        for ep in self.data["episodes"]:
            if ep.get("score", 0) < policy["success_threshold"]:
                continue
            overlap = len(query & tokens(ep["request"])) / len(query)
            if overlap > 0:
                scored.append((overlap, ep))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [ep for _, ep in scored[:policy["retrieval_k"]]]

    def search(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        """Keyword search over all episodes, used by the memory_search tool."""
        q = tokens(query)
        hits = [(len(q & tokens(ep["request"])), ep)
                for ep in self.data["episodes"]]
        hits = [h for h in hits if h[0] > 0]
        hits.sort(key=lambda pair: pair[0], reverse=True)
        return [{"request": ep["request"], "answer": ep["answer"],
                 "score": ep.get("score")} for _, ep in hits[:limit]]

    # Patterns -------------------------------------------------------------

    def log_pattern(self, kind: str, mode: str, detail: str,
                    episode_id: str | None = None) -> None:
        self.data["patterns"][kind].append({
            "ts": _now(), "mode": mode, "detail": detail,
            "episode": episode_id})
        del self.data["patterns"][kind][:-1000]
        self.save()

    def pattern_counts(self, kind: str) -> Counter:
        return Counter(p["mode"] for p in self.data["patterns"][kind])

    # Regression suite -----------------------------------------------------

    @property
    def benchmarks(self) -> list[dict[str, Any]]:
        return self.data["benchmarks"]

    def add_benchmark(self, request: str, source: str,
                      expected: dict | None = None) -> dict[str, Any]:
        for case in self.benchmarks:
            if case["request"].strip().lower() == request.strip().lower():
                return case
        case = {"id": "B-" + uuid.uuid4().hex[:8], "request": request,
                "expected": expected, "source": source}
        self.benchmarks.append(case)
        self.save()
        return case

    def select_benchmarks(self, must_include: list[str],
                          limit: int) -> list[dict[str, Any]]:
        """The cases that triggered a change, then seeds, then other cases."""
        by_id = {c["id"]: c for c in self.benchmarks}
        chosen = [by_id[i] for i in must_include if i in by_id]
        rest = sorted((c for c in self.benchmarks if c not in chosen),
                      key=lambda c: c["source"] != "seed")
        for case in rest:
            if len(chosen) >= max(limit, len(must_include)):
                break
            chosen.append(case)
        return chosen
