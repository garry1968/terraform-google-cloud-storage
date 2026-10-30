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

Add --offline to use the deterministic stand-in model (no API key needed).
Commands that call the model stop at --max-cost (estimated US dollars) or
--max-calls, whichever comes first, and exit with code 3.
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys

from .llm import Budget, BudgetExceeded, ClaudeLLM, OfflineLLM
from .orchestrator import Orchestrator

# Commands that call the model. Read-only commands never do.
MODEL_COMMANDS = {"ask", "chat", "approve", "benchmark", "practice"}
BUDGET_STOPPED = 3


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.offline or args.command not in MODEL_COMMANDS:
        llm = OfflineLLM()
    else:
        llm = ClaudeLLM(model=args.model)
    budget = None
    if args.command in MODEL_COMMANDS:
        llm = budget = Budget(llm, args.max_calls, args.max_cost)
    ai = Orchestrator(llm, args.state_dir)

    code = 0
    try:
        run(ai, args)
    except BudgetExceeded as exc:
        print(f"Stopped: {exc}. Any change still being tested was rolled "
              "back, and an approved one went back in the queue. Raise the "
              "cap with --max-cost or --max-calls.")
        code = BUDGET_STOPPED
    if budget:
        print(budget.summary())
    return code


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
