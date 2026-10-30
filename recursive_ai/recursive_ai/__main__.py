"""Command-line interface.

    python -m recursive_ai ask "What is 12 times 7?"
    python -m recursive_ai chat
    python -m recursive_ai approvals
    python -m recursive_ai approve C-1234abcd --note "looks right"
    python -m recursive_ai reject C-1234abcd
    python -m recursive_ai history
    python -m recursive_ai rollback 3
    python -m recursive_ai audit
    python -m recursive_ai benchmark
    python -m recursive_ai practice --cases 5 --max-cost 1
    python -m recursive_ai baseline --split dev --per-category 2 --trials 1
    python -m recursive_ai load --repo https://github.com/you/ai-memory
    python -m recursive_ai save

Add --offline to use the deterministic stand-in model (no API key needed).
Commands that call the model stop at --max-cost (estimated US dollars) or
--max-calls, whichever comes first, and exit with code 3.

After load, every command that can change the memory saves it to that
repository when it finishes, and exits with code 4 if saving failed. With
RECURSIVE_AI_MEMORY_REPO set, a new machine loads the memory by itself.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from . import sync
from .baseline import CONDITIONS, Baseline, summarize
from .llm import LLM, Budget, BudgetExceeded, ClaudeLLM, OfflineLLM
from .orchestrator import Orchestrator
from .tasks import SPLITS, make_tasks

# Commands that call the model. Read-only commands never do.
MODEL_COMMANDS = {"ask", "chat", "approve", "benchmark", "practice",
                  "baseline"}
# Commands that never change the memory, so there is nothing to save.
READ_ONLY_COMMANDS = {"approvals", "history", "audit", "baseline"}
MEMORY_REPO_ENV = "RECURSIVE_AI_MEMORY_REPO"
BUDGET_STOPPED = 3
MEMORY_NOT_SAVED = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recursive_ai",
                                     description=__doc__.split("\n")[0])
    parser.add_argument("--state-dir", default=".recursive_ai_state",
                        help="where config, memory and logs are stored")
    parser.add_argument("--offline", action="store_true",
                        help="use the deterministic offline model")
    parser.add_argument("--model", default="claude-opus-5-5")
    sub = parser.add_subparsers(dest="command", required=True)

    budget = argparse.ArgumentParser(add_help=False)
    budget.add_argument("--max-cost", type=float, default=1.00,
                        help="stop before spending more than this many US "
                             "dollars (estimated) in this run")
    budget.add_argument("--max-calls", type=int, default=300,
                        help="stop after this many API calls in this run")

    sub.add_parser("ask", parents=[budget],
                   help="answer one request").add_argument("text")
    sub.add_parser("chat", parents=[budget], help="interactive session")
    sub.add_parser("approvals", help="list changes awaiting approval")
    for name in ("approve", "reject"):
        p = sub.add_parser(name, parents=[budget] if name == "approve" else [],
                           help=f"{name} a queued change")
        p.add_argument("proposal_id")
        p.add_argument("--by", default=getpass.getuser())
        p.add_argument("--note", default="")
    sub.add_parser("history", help="list config versions")
    p = sub.add_parser("rollback", help="restore an earlier config version")
    p.add_argument("version", type=int)
    p.add_argument("--by", default=getpass.getuser())
    p = sub.add_parser("audit", help="show recent audit events")
    p.add_argument("-n", type=int, default=20)
    sub.add_parser("benchmark", parents=[budget],
                   help="run the regression suite")
    p = sub.add_parser("practice", parents=[budget],
                       help="answer practice questions with known answers "
                            "and learn from the mistakes")
    p.add_argument("--cases", type=int, default=5,
                   help="how many questions to practise this run")
    p = sub.add_parser("baseline", parents=[budget],
                       help="compare plain Claude, Claude with tools and the "
                            "full system on generated tasks; changes nothing")
    p.add_argument("--split", choices=SPLITS, default="dev",
                   help="holdout tasks are for milestone comparisons only")
    p.add_argument("--per-category", type=_positive, default=2)
    p.add_argument("--trials", type=_positive, default=1)
    p.add_argument("--conditions", type=_conditions,
                   default=",".join(CONDITIONS),
                   help=f"comma-separated, from {', '.join(CONDITIONS)}")
    p.add_argument("--out", help="also write every result to this JSON file")
    p = sub.add_parser("load", help="connect the memory to a git repository, "
                                    "or bring it up to date")
    p.add_argument("--repo", default=os.environ.get(MEMORY_REPO_ENV),
                   help="URL of a private git repository (default: "
                        f"${MEMORY_REPO_ENV})")
    sub.add_parser("save", help="save the memory to its repository now")
    return parser


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return value


def _conditions(text: str) -> tuple[str, ...]:
    chosen = tuple(c.strip() for c in text.split(",") if c.strip())
    if not chosen or any(c not in CONDITIONS for c in chosen):
        raise argparse.ArgumentTypeError(
            f"choose from {', '.join(CONDITIONS)}")
    return chosen


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    state_dir = Path(args.state_dir)
    if args.command in ("load", "save"):
        return memory_command(state_dir, args)
    load_before(state_dir)

    if args.offline or args.command not in MODEL_COMMANDS:
        llm = OfflineLLM()
    else:
        llm = ClaudeLLM(model=args.model)
    budget = None
    if args.command in MODEL_COMMANDS:
        llm = budget = Budget(llm, args.max_calls, args.max_cost)

    code = 0
    try:
        if args.command == "baseline":
            run_baseline(llm, args)
        else:
            run(Orchestrator(llm, args.state_dir), args)
    except BudgetExceeded as exc:
        print(f"Stopped: {exc}. Any change still being tested was rolled "
              "back, and an approved one went back in the queue. Raise the "
              "cap with --max-cost or --max-calls.")
        code = BUDGET_STOPPED
    if budget:
        print(budget.summary())
    if args.command not in READ_ONLY_COMMANDS \
            and not save_after(state_dir, args.command) and code == 0:
        code = MEMORY_NOT_SAVED
    return code


def memory_command(state_dir: Path, args: argparse.Namespace) -> int:
    try:
        if args.command == "load":
            print(sync.load(state_dir, args.repo))
        elif not sync.is_synced(state_dir):
            print("Memory is not connected to a repository yet. Run: "
                  "python -m recursive_ai load --repo <URL>")
            return 1
        # After load, this uploads a memory that existed before connecting.
        if sync.save(state_dir, f"recursive_ai {args.command}"):
            print("Memory saved.")
        elif args.command == "save":
            print("Nothing new to save.")
    except sync.SyncError as exc:
        print(f"Memory error: {exc}")
        return 1
    return 0


def load_before(state_dir: Path) -> None:
    """Bring saved memory up to date, or load it onto a new machine."""
    repo = os.environ.get(MEMORY_REPO_ENV)
    synced = sync.is_synced(state_dir)
    if not synced and not (repo and not state_dir.exists()):
        if repo:
            print(f"Memory is not being saved. To connect it to {repo}, "
                  "run: python -m recursive_ai load")
        return
    try:
        message = sync.load(state_dir, repo)
    except sync.SyncError as exc:
        print(f"Memory not updated from its repository ({exc}). Using the "
              "copy on this machine.")
        return
    if not synced:
        print(message)


def save_after(state_dir: Path, command: str) -> bool:
    """Save the memory if it is connected. False only if saving failed."""
    if not sync.is_synced(state_dir):
        return True
    try:
        if sync.save(state_dir, f"recursive_ai {command}"):
            print("Memory saved.")
    except sync.SyncError as exc:
        print(f"Memory NOT saved ({exc}). It is still on this machine; "
              "run: python -m recursive_ai save")
        return False
    return True


def run_baseline(llm: LLM, args: argparse.Namespace) -> None:
    conditions = args.conditions
    tasks = make_tasks(args.split, args.per_category)
    results = []
    with tempfile.TemporaryDirectory() as workdir:
        baseline = Baseline(llm, Path(args.state_dir), Path(workdir))
        try:
            for result in baseline.run(tasks, conditions, args.trials):
                results.append(result)
                if result.error:
                    print(f"  {result.condition} error: {result.error}")
                if result.condition == conditions[-1]:
                    marks = "  ".join(
                        f"{r.condition} {'right' if r.correct else 'WRONG'}"
                        for r in results[-len(conditions):])
                    print(f"{result.task} (trial {result.trial}): {marks}")
        finally:
            print("\n".join(summarize(results, conditions)))
            if args.out:
                Path(args.out).write_text(json.dumps({
                    "model": "offline" if args.offline else args.model,
                    "split": args.split, "per_category": args.per_category,
                    "trials": args.trials, "conditions": conditions,
                    "finished": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"),
                    "results": [asdict(r) for r in results]}, indent=1))


def run(ai: Orchestrator, args: argparse.Namespace) -> None:
    if args.command == "ask":
        print(ai.handle(args.text).text)
    elif args.command == "chat":
        print("Type a request, or 'quit' to exit.")
        while True:
            try:
                line = input("> ")
            except (EOFError, KeyboardInterrupt):
                break
            if line.strip().lower() in ("quit", "exit"):
                break
            if line.strip():
                print(ai.handle(line).text, "\n")
    elif args.command == "approvals":
        pending = ai.pending_approvals()
        if not pending:
            print("No changes awaiting approval.")
        for p in pending:
            print(f"{p.id}  risk={p.risk}  {p.operation} {p.target}\n"
                  f"  value: {json.dumps(p.value)}\n"
                  f"  why: {p.rationale}\n"
                  f"  risk reasons: {'; '.join(p.risk_reasons)}\n")
    elif args.command == "approve":
        outcome = ai.approve(args.proposal_id, args.by, args.note)
        print(f"{outcome.proposal_id}: {outcome.status} - {outcome.detail}")
    elif args.command == "reject":
        p = ai.reject(args.proposal_id, args.by, args.note)
        print(f"{p.id}: rejected")
    elif args.command == "history":
        for entry in ai.store.history():
            marker = "*" if entry["version"] == ai.store.version else " "
            print(f"{marker} v{entry['version']}  {entry['ts']}  "
                  f"{entry['reason']}")
    elif args.command == "rollback":
        print(f"Now at v{ai.rollback(args.version, args.by)}")
    elif args.command == "audit":
        for event in ai.audit.tail(args.n):
            print(json.dumps(event))
    elif args.command == "benchmark":
        scores = []
        try:
            for case_id, score in ai.benchmark():
                scores.append(score)
                print(f"{case_id}: {score:.2f}")
        finally:
            if scores:
                print(f"mean: {sum(scores) / len(scores):.2f} "
                      f"over {len(scores)} cases")
    elif args.command == "practice":
        items = []
        try:
            for item in ai.practice(args.cases):
                items.append(item)
                print("\n".join(ai.output.practice_lines(len(items), item)))
        finally:
            print("\n".join(ai.output.practice_summary(items)))


if __name__ == "__main__":
    sys.exit(main())
