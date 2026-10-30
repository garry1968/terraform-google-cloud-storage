"""Tests for the generated tasks and the baseline evaluation."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from recursive_ai.__main__ import BUDGET_STOPPED, main
from recursive_ai.baseline import (
    CONDITIONS,
    Baseline,
    TrialResult,
    sign_test,
    summarize,
)
from recursive_ai.llm import Budget, ClaudeLLM, OfflineLLM
from recursive_ai.tasks import (
    CATEGORIES,
    SOLVED,
    _coin_ways,
    _colourings,
    _ice_moves,
    _jug_steps,
    _key_maze_moves,
    _tiles_by_distance,
    families,
    grade,
    make_tasks,
)


def as_reply(task, answer):
    if task.kind == "grid":
        rows = "\n".join(" ".join(str(v) for v in row) for row in answer)
        return f"Working it out...\nAnswer:\n{rows}"
    if task.kind == "names":
        return f"Working it out...\nAnswer: {', '.join(answer)}"
    return f"Working it out...\nAnswer: {answer}"


def test_splits_share_no_kind_of_problem_and_are_reproducible():
    dev, holdout = make_tasks("dev", 20), make_tasks("holdout", 20)
    assert len(dev) == 20 * len(CATEGORIES)
    for cat in CATEGORIES:
        assert len(families(cat, "dev")) >= 2
        assert len(families(cat, "holdout")) >= 2
    assert not {t.family for t in dev} & {t.family for t in holdout}
    assert not {t.prompt for t in dev} & {t.prompt for t in holdout}
    assert len({t.prompt for t in dev}) == len(dev)
    assert [t.prompt for t in make_tasks("dev", 20)] == \
        [t.prompt for t in dev]


def test_known_answers_grade_right_and_others_wrong():
    for task in make_tasks("dev", 12) + make_tasks("holdout", 12):
        assert grade(task, as_reply(task, task.answer)), task.id
        wrong = {"number": lambda a: a + 1,
                 "text": lambda a: a + "x",
                 "name": lambda a: next(n for n in task.choices if n != a),
                 "names": lambda a: sorted(set(task.choices) - set(a)),
                 "grid": lambda a: [[(v + 1) % 10 for v in row]
                                    for row in a]}[task.kind](task.answer)
        assert not grade(task, as_reply(task, wrong)), task.id


def test_grader_reads_common_answer_formats():
    number, name, names, grid, text = (
        next(t for t in make_tasks("dev", 30) if t.kind == kind)
        for kind in ("number", "name", "names", "grid", "text"))
    assert grade(number, f"So it is...\nAnswer: **{number.answer:,}**")
    assert grade(number, str(number.answer))
    assert not grade(number, f"Not {number.answer}: it is "
                             f"{number.answer + 1}.")
    rows = "\n".join(" ".join(str(v) for v in r) for r in grid.answer)
    assert grade(grid, f"The rule flips it.\nAnswer:\n```\n{rows}\n```")
    other = next(n for n in name.choices if n != name.answer)
    assert not grade(name, f"Answer: {name.answer} or {other}")
    assert grade(names, "Answer: " + " and ".join(reversed(names.answer)))
    extra = next(n for n in names.choices if n not in names.answer)
    assert not grade(names, "Answer: " + ", ".join(names.answer + [extra]))
    assert grade(text, f"Answer: `{text.answer}`")


def test_search_answers_match_hand_solved_puzzles():
    assert _jug_steps(3, 5, 4) == 6
    assert _coin_ways([1, 2, 5], 5) == 4
    # A-B-C is a triangle (6 ways); D and E are free (3 ways each).
    assert _colourings([("A", "B"), ("B", "C"), ("A", "C")]) == 6 * 3 * 3
    layouts = _tiles_by_distance()
    assert layouts[0] == [SOLVED]
    assert sum(len(v) for v in layouts.values()) == 181_440
    assert (1, 2, 3, 4, 5, 6, 7, 0, 8) in layouts[1]
    # Right, down, left: the slide passes nothing it could stop on sooner.
    assert _ice_moves(["S..#", "#...", "G..."], (0, 0)) == 3
    # 4 moves to the key, then 6 back through the door to G.
    assert _key_maze_moves(["S.D.G", "#.###", "k...."], (0, 0)) == 10
    assert _key_maze_moves(["S.D.G", "#.###", "....."], (0, 0)) is None


def test_baseline_runs_every_condition_and_changes_nothing(make_ai,
                                                           tmp_path):
    ai = make_ai()
    ai.handle("What is 17 * 23?")
    state = tmp_path / "state"
    before = {p: p.read_bytes() for p in state.rglob("*") if p.is_file()}

    llm = Budget(OfflineLLM(), max_calls=10_000, max_cost=1.0)
    tasks = make_tasks("dev", 1)
    work = tmp_path / "work"
    work.mkdir()
    results = list(Baseline(llm, state, work).run(tasks, CONDITIONS, 2))

    assert len(results) == len(tasks) * len(CONDITIONS) * 2
    assert {r.condition for r in results} == set(CONDITIONS)
    calls = {c: {r.calls for r in results if r.condition == c}
             for c in CONDITIONS}
    assert calls["plain"] == {1}
    assert min(calls["full"]) >= 3  # plan, answer and self-check at least
    after = {p: p.read_bytes() for p in state.rglob("*") if p.is_file()}
    assert after == before


def test_summary_pairs_conditions_on_the_same_tasks():
    def result(task, condition, correct):
        return TrialResult(task, "math", condition, 1, correct, 0.01, 1.0)

    results = [result(f"t{i}", c, c == "full" or i == 0)
               for i in range(8) for c in ("plain", "full")]
    lines = summarize(results, ("plain", "full"))
    assert any("full vs plain: better on 7, worse on 0" in line
               for line in lines)


def test_sign_test():
    assert sign_test(0, 0) == 1.0
    assert sign_test(5, 5) == 1.0
    assert sign_test(8, 0) == pytest.approx(2 / 256)


def test_learning_code_never_sees_the_tasks():
    code = ("import sys, recursive_ai.orchestrator, recursive_ai.learning; "
            "print('recursive_ai.tasks' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True,
                         cwd=Path(__file__).resolve().parents[1]).stdout
    assert out.strip() == "False"


def test_cli_baseline_reports_saves_and_stops_at_the_cap(tmp_path, capsys):
    out = tmp_path / "results.json"
    args = ["--offline", "--state-dir", str(tmp_path / "state"), "baseline",
            "--per-category", "1"]
    assert main(args + ["--out", str(out)]) == 0
    assert "full vs plain" in capsys.readouterr().out
    assert len(json.loads(out.read_text())["results"]) == \
        len(CATEGORIES) * len(CONDITIONS)

    assert main(args + ["--max-calls", "5"]) == BUDGET_STOPPED
    printed = capsys.readouterr().out
    assert "Stopped: reached the cap of 5 API calls" in printed
    assert "Condition" in printed


@pytest.mark.parametrize("model,fallback", [("claude-opus-5-5", True),
                                            ("claude-haiku-5-5", False)])
def test_fallback_is_only_sent_to_models_that_have_one(model, fallback):
    sent = {}
    usage = SimpleNamespace(input_tokens=1000, output_tokens=1000,
                            cache_creation_input_tokens=0,
                            cache_read_input_tokens=0)

    def create(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(
            stop_reason="end_turn", model=model, usage=usage,
            content=[SimpleNamespace(type="text", text="hi")])

    client = SimpleNamespace(beta=SimpleNamespace(
        messages=SimpleNamespace(create=create)))
    llm = ClaudeLLM(model=model, client=client)
    llm.generate(purpose="plan", system="", messages=[
        {"role": "user", "content": "hi"}])
    assert ("fallbacks" in sent) is fallback
    expected = 0.024 if fallback else 0.0006  # $/M: 4+20, or 0.10+0.50
    assert llm.usage.cost == pytest.approx(expected)
