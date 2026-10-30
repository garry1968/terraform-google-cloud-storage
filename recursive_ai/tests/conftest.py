import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recursive_ai.llm import LLMResponse, OfflineLLM  # noqa: E402
from recursive_ai.orchestrator import Orchestrator  # noqa: E402


class ScriptedLLM(OfflineLLM):
    """OfflineLLM, except calls for a scripted purpose return queued JSON."""

    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []

    def generate(self, *, purpose, system, messages, tools=None,
                 schema=None):
        self.calls.append(purpose)
        if self.script.get(purpose):
            text = json.dumps(self.script[purpose].pop(0))
            return LLMResponse(text, content=[{"type": "text",
                                               "text": text}])
        return super().generate(purpose=purpose, system=system,
                                messages=messages, tools=tools,
                                schema=schema)


@pytest.fixture(autouse=True)
def no_saved_memory(monkeypatch):
    # Tests must never load from, or save to, a real memory repository.
    monkeypatch.delenv("RECURSIVE_AI_MEMORY_REPO", raising=False)


@pytest.fixture
def make_ai(tmp_path):
    def make(llm=None):
        return Orchestrator(llm or OfflineLLM(), tmp_path / "state")
    return make
