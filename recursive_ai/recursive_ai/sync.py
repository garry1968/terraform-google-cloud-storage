"""Saves the state folder to a git repository, so what the system learns
outlives the machine it ran on.

The state folder (config, memory, approvals, audit log) becomes a clone of
that repository. The CLI brings it up to date before a command and pushes it
after any command that can change it. Use a private repository: the memory
holds every question asked and every answer given.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

BRANCH = "main"

# Used only where git has no identity configured.
FALLBACK_IDENTITY = ["-c", "user.name=recursive_ai",
                     "-c", "user.email=recursive-ai@localhost"]


class SyncError(RuntimeError):
    """A git step failed. Local state is left as it was."""


def _git(state_dir: Path, *args: str, check: bool = True
         ) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(["git", "-C", str(state_dir), *args],
                                capture_output=True, text=True)
    except FileNotFoundError:
        raise SyncError("git is not installed") from None
    if check and result.returncode != 0:
        step = next(a for a in args if not a.startswith("-")
                    and "=" not in a)
        raise SyncError(f"git {step} failed: "
                        f"{(result.stderr or result.stdout).strip()}")
    return result


def is_synced(state_dir: Path) -> bool:
    return (state_dir / ".git").exists()


def _remote_has_branch(state_dir: Path) -> bool:
    return bool(_git(state_dir, "ls-remote", "--heads", "origin",
                     BRANCH).stdout.strip())


def load(state_dir: Path, repo: str | None = None) -> str:
    """Connect the state folder to the repository, or bring it up to date.

    Returns what happened, for the CLI to print.
    """
    if is_synced(state_dir):
        if not _remote_has_branch(state_dir):
            return "Memory repository is connected; nothing saved yet."
        _git(state_dir, "pull", "-q", "--ff-only", "origin", BRANCH)
        return "Memory is up to date with the saved copy."
    if not repo:
        raise SyncError("no memory repository given: pass --repo or set "
                        "RECURSIVE_AI_MEMORY_REPO")

    state_dir.mkdir(parents=True, exist_ok=True)
    has_local_state = any(state_dir.iterdir())
    _git(state_dir, "init", "-q", "-b", BRANCH)
    try:
        _git(state_dir, "remote", "add", "origin", repo)
        _git(state_dir, "fetch", "-q", "origin")
        if not _remote_has_branch(state_dir):
            return "Memory repository is connected; nothing saved yet."
        if has_local_state:
            raise SyncError(
                f"this folder has its own memory and so does {repo}. To use "
                "the saved one, rename this folder and run load again")
        _git(state_dir, "checkout", "-q", BRANCH)
    except SyncError:
        # Leave the folder exactly as it was before load.
        shutil.rmtree(state_dir / ".git")
        raise
    return "Loaded the saved memory."


def save(state_dir: Path, message: str) -> bool:
    """Commit any changes and push them, along with any commits an earlier
    failed push left behind. Returns True if anything new was uploaded."""
    if _git(state_dir, "status", "--porcelain").stdout.strip():
        identity = [] if _git(state_dir, "config", "user.email",
                              check=False).returncode == 0 \
            else FALLBACK_IDENTITY
        _git(state_dir, "add", "-A")
        _git(state_dir, *identity, "commit", "-q", "-m", message)
    if _git(state_dir, "rev-parse", "--verify", "-q", "HEAD",
            check=False).returncode != 0:
        return False  # nothing has ever been saved here
    pushed = _git(state_dir, "push", "--porcelain", "-u", "origin",
                  f"HEAD:{BRANCH}").stdout
    # Porcelain marks a ref that was already up to date with "=".
    return not any(line.startswith("=\t") for line in pushed.splitlines())
