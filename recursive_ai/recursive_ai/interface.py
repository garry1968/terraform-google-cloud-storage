"""Input layer: turns raw user input into a validated request."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


class InputRejected(ValueError):
    """The input is empty, too long, or blocked by the safety policy."""


@dataclass
class UserRequest:
    text: str
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_input(raw: str, config: dict[str, Any],
                session_id: str | None = None) -> UserRequest:
    # Drop control characters but keep line structure.
    text = "".join(ch for ch in (raw or "")
                   if ch.isprintable() or ch in "\n\t").strip()
    if not text:
        raise InputRejected("empty request")
    limit = config["safety_policy"]["max_input_chars"]
    if len(text) > limit:
        raise InputRejected(f"request is longer than {limit} characters")
    request = UserRequest(text)
    if session_id:
        request.session_id = session_id
    return request
