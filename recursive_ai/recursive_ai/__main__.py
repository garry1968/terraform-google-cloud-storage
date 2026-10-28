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

Add --offline to use the deterministic stand-in model (no API key needed).
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys

from .llm import ClaudeLLM, OfflineLLM
from .orchestrator import Orchestrator

# Commands that call the model. Read-only commands never do.
MODEL_COMMANDS = {"ask", "chat", "approve", "benchmark"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recursive_ai",
                                     description=__doc__.split("\n")[0])
    parser.add_argument("--state-dir", default=".recursive_ai_state",
                        help="where config, memory and logs are stored")
    parser.add_argument("--offline", action="store_true",
                        help="use the deterministic offline model")
    parser.add_argument("--model", default="claude-opus-5-5")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ask", help="answer one request").add_argument("text")
    sub.add_parser("chat", help="interactive session")
    sub.add_parser("approvals", help="list changes awaiting approval")
    for name in ("approve", "reject"):
        p = sub.add_parser(name, help=f"{name} a queued change")
        p.add_argument("proposal_id")
        p.add_argument("--by", default=getpass.getuser())
        p.add_argument("--note", default="")
    sub.add_parser("history", help="list config versions")
    p = sub.add_parser("rollback", help="restore an earlier config version")
    p.add_argument("version", type=int)
    p.add_argument("--by", default=getpass.getuser())
    p = sub.add_parser("audit", help="show recent audit events")
    p.add_argument("-n", type=int, default=20)
    sub.add_parser("benchmark", help="run the regression suite")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.offline or args.command not in MODEL_COMMANDS:
        llm = OfflineLLM()
    else:
        llm = ClaudeLLM(model=args.model)
    ai = Orchestrator(llm, args.state_dir)

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
        result = ai.benchmark()
        for case_id, score in result.scores.items():
            print(f"{case_id}: {score:.2f}")
        print(f"mean: {result.mean:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
