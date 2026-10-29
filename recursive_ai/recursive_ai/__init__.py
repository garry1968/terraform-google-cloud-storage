"""A modular assistant that improves itself, with human-gated major changes."""

from .llm import ClaudeLLM, OfflineLLM
from .orchestrator import Orchestrator
from .safety import Risk, classify_change

__all__ = ["ClaudeLLM", "OfflineLLM", "Orchestrator", "Risk",
           "classify_change"]
