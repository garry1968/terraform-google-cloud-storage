# Recursive AI

A modular assistant built on Claude that improves itself from its own
mistakes. Small, low-risk improvements apply automatically, but only if a
retest shows they help. Major or risky changes wait for a human.

## Quick start

```bash
cd recursive_ai
pip install -e ".[test]"

# No API key needed: deterministic offline model
python -m recursive_ai --offline ask "What is 12 times 7?"
python -m recursive_ai approvals
python -m recursive_ai --offline approve <ID> --note "looks right"

# Real model (Claude Opus 5.5). Uses ANTHROPIC_API_KEY or an `ant auth login` profile
python -m recursive_ai ask "Explain how a B-tree stays balanced."
python -m recursive_ai chat

# Learn without a human asking: practise questions with known answers
python -m recursive_ai practice --cases 5 --max-cost 1

pytest -q
```

The offline model is pattern matching, not intelligence. It exists so the
full loop (learning, retest, rollback, approvals) runs and is tested without
an API key.

## Components

| Layer | File | Job |
|---|---|---|
| Input / interface | `interface.py`, `__main__.py` | Validate input, CLI |
| Orchestrator | `orchestrator.py` | Run the pipeline, hand episodes to learning |
| Planner | `planner.py` | Short, tool-aware plan (structured output) |
| Responder | `responder.py` | Execute the plan, call tools |
| Memory | `memory.py` | Episodes, success/error patterns, regression suite |
| Tool router | `tools.py` | The only path to tools; default-deny permissions |
| Evaluator | `evaluator.py` | LLM judge + deterministic checks, tagged failure modes |
| Learning controller | `learning.py` | Propose, apply, retest, keep / refine / roll back |
| Safety / approval gate | `safety.py` | Risk rules, approval queue, audit log, content screen |
| Output | `output.py` | Answer + confidence + learning notes |
| Config | `config.py` | Versioned config; every change is a snapshot |
| Model | `llm.py` | `ClaudeLLM`, `OfflineLLM`, and the `Budget` spending cap |
| Practice set | `practice.py` | Questions with known answers for unattended learning |
| Saving | `sync.py` | Keeps the state folder in a git repository |

## The loop

```mermaid
flowchart LR
  A[Request] --> B[Plan] --> C[Respond + tools] --> D[Evaluate]
  D -- low confidence / issues --> C
  D --> E{Passed?}
  E -- yes --> F[Log success]
  E -- no --> G[Log failure modes<br/>add case to suite] --> H[Propose changes]
  H --> I{Risk}
  I -- low --> J[Apply] --> K[Retest suite]
  I -- medium/high --> Q[Human approval queue] -- approved --> J
  K -- better, no regressions --> L[Keep]
  K -- no gain --> M[Refine once] --> J
  K -- regression --> N[Roll back]
```

1. Receive request. 2. Plan and respond. 3. Evaluate with metrics.
4. Tag failure modes. 5. Propose a config change. 6. Classify risk in code.
7. Low risk: apply. 8. Medium/high: queue for a human.
9. Retest on the regression suite (always includes the failing case).
10. Keep if it scores better with no per-case regression; otherwise refine
once, then roll back. Changes that were rolled back or rejected are not
proposed again.

Self-correction per request: revise when confidence is low or checks fail
(up to `self_correction.max_revisions`); compare several candidates when the
planner rates a request high-complexity.

## Unattended learning

`practice` works through questions with known answers (40 in `practice.py`,
plus the seed cases), least recently practised first, and learns from the
ones it gets wrong. Self-correction never sees the known answer: it is
checked once, at the end, as a whole word or number ("391" does not match
"3910"). The issue it raises does not quote the answer, so the learner
cannot memorise it. Retests prefer cases with known answers, so a change has
to fix real answers, not just please the LLM judge.

An issue that keeps coming back on answers that pass (`ungrounded_math`,
`low_confidence`, `tool_error`, `permission_denied`) also triggers learning
once it has come up `learning.recurring_issue_threshold` times (default 3)
since it was last addressed. Like the rest of `learning`, only a human can
change that number.

## Baseline evaluation

`baseline` tests one question: does the system beat the model it runs on?
It runs generated tasks with known answers (maths, logic, ARC-style grid
puzzles, maze planning, reading code, and finding a hidden rule) under three
conditions, all on the same model and settings:

| Condition | What runs |
|---|---|
| `plain` | One call to the model, no tools |
| `tools` | The responder with its tools and default settings |
| `full` | The whole pipeline with the learned settings and memory |

It prints accuracy, cost and time per task for each condition, results by
category, and how often each condition beat `plain` on the same task, with a
sign test. With `--trials 2` or more it also shows how consistent each one is.

- Tasks are generated from seeds, so they are new to every model. Each
  possible task belongs to exactly one split, `dev` or `holdout`, by a hash
  of its text. Use `dev` while improving the system; keep `holdout` for
  milestone comparisons.
- Nothing is learned or saved: `full` runs on a throwaway copy of the state
  folder, and the learning code never imports the tasks.
- It spends money like any model command and stops at `--max-cost`. Try it
  for free with `--offline`.

```bash
python -m recursive_ai baseline --split dev --per-category 2 --trials 1 --out results.json
```

## Spending cap

Every command that calls the model stops at `--max-cost` (US dollars,
default 1.00) or `--max-calls` (default 300), whichever comes first, and
exits with code 3. A call starts only if the run would stay within
`--max-cost` even if that call cost the most it can: all `max_tokens` of
output, plus the request counted at 2 characters per token. So a run never
goes over its cap, but it may stop with up to one call's worth unspent
(about $0.33 on Claude Opus 5.5, under $0.01 on Claude Haiku 5.5).

Spend is estimated from each response's token usage at the list prices in
`llm.py`, so it is only as accurate as that table. `--max-calls` counts
every model call, including tool rounds, self-checks and revisions. When the
SDK retries a failed request (up to twice, after rate-limit or server
errors), that still counts as one call. A change whose retest is cut short
is rolled back and not counted as failed; an approved one goes back in the
approval queue. Learning is the expensive part: one failure can retest two
changes on up to `learning.max_benchmark_cases` cases each. With
`learning.auto_apply_max_risk` set to `none` (proposals only), nothing is
retested until you approve it.

## Risk rules

The rules are in `safety.py`, not in the config, so the system cannot edit
them. The proposer's own risk rating can raise the risk, never lower it.

| Change | Risk | Path |
|---|---|---|
| Append a short guidance line to `prompt_addenda.*` | Low | Auto, retested |
| Add a routing rule for an already-permitted tool | Low | Auto, retested |
| Small tune of `memory_policy` / `self_correction` within bounds | Low | Auto, retested |
| Long text, removals of rules, out-of-bounds numbers | Medium | Human |
| `safety_policy`, `tool_permissions`, `memory_schema`, `planner` | High | Human |
| `prompts` (base prompts), `evaluator` (metrics) | High | Human |
| Any value mentioning safety, approval, permissions, bypass | High | Human |
| Unknown target | High | Human (fail closed) |
| `learning` section, `version` | Locked | Rejected; humans edit `config.json` directly |

Other guarantees:

- High risk never auto-applies, whatever `learning.auto_apply_max_risk` says.
- Only the CLI `approve` command applies a queued change. Approved changes
  are still retested and rolled back if they break the suite.
- Every proposal, apply, retest, approval and rollback goes to
  `audit.jsonl`. Config history is append-only (`history`, `rollback N`).
- Chat requests to bypass approval or safety are refused.
- A memory file with a different schema version will not load.

## State

Stored in `--state-dir` (default `.recursive_ai_state/`): `config.json`,
`config_history/`, `memory.json`, `approvals.json`, `audit.jsonl`.

### Saving it permanently

The state folder lives on one machine. To keep it, connect it to a git
repository. Use a private one: the memory holds every question and answer.

```bash
python -m recursive_ai load --repo https://github.com/you/ai-memory
```

The folder becomes a clone of that repository. From then on, every command
that can change it pulls first and pushes when it finishes (`save` does it
by hand). A failed push keeps everything locally, prints a warning and
exits with code 4. Set `RECURSIVE_AI_MEMORY_REPO` to the same URL and a new
machine loads the memory the first time any command runs. `load` will not
mix two memories: if the folder and the repository both hold one, it stops
and changes nothing.

## Claude API settings

`ClaudeLLM` uses `claude-opus-5-5` with structured outputs, strict tool
schemas, prompt caching, an explicit effort level per call site, and the
server-side refusal fallback (`fallbacks: "default"`). A refusal or API error
is reported to the user and logged as a failure pattern. Each retest runs
the pipeline on up to `learning.max_benchmark_cases` cases, so learning
spends extra API calls when it triggers.
