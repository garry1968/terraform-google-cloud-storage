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
from recursive_ai.tasks import CATEGORIES, grade, make_tasks, split_of


def as_reply(task, answer):
    if task.kind == "grid":
        rows = "\n".join(" ".join(str(v) for v in row) for row in answer)
        return f"Working it out...\nAnswer:\n{rows}"
    return f"Working it out...\nAnswer: {answer}"


def test_splits_never_share_a_task_and_are_reproducible():
    dev, holdout = make_tasks("dev", 20), make_tasks("holdout", 20)
    assert len(dev) == 20 * len(CATEGORIES)
    assert not {t.prompt for t in dev} & {t.prompt for t in holdout}
    assert len({t.prompt for t in dev}) == len(dev)
    assert all(split_of(t.prompt) == "holdout" for t in holdout)
    assert [t.prompt for t in make_tasks("dev", 20)] == \
        [t.prompt for t in dev]


def test_known_answers_grade_right_and_others_wrong():
    for task in make_tasks("dev", 15) + make_tasks("holdout", 15):
        assert grade(task, as_reply(task, task.answer)), task.id
        wrong = {"number": lambda a: a + 1,
                 "text": lambda a: a + "x",
                 "name": lambda a: next(n for n in task.choices if n != a),
                 "grid": lambda a: [[(v + 1) % 10 for v in row]
                                    for row in a]}[task.kind](task.answer)
        assert not grade(task, as_reply(task, wrong)), task.id


def test_grader_reads_common_answer_formats():
    number, name, grid, text = (
        next(t for t in make_tasks("dev", 30) if t.kind == kind)
        for kind in ("number", "name", "grid", "text"))
    assert grade(number, f"So it is...\nAnswer: **{number.answer:,}**")
    assert grade(number, str(number.answer))
    assert not grade(number, f"Not {number.answer}: it is "
                             f"{number.answer + 1}.")
    rows = "\n".join(" ".join(str(v) for v in r) for r in grid.answer)
    assert grade(grid, f"The rule flips it.\nAnswer:\n```\n{rows}\n```")
    other = next(n for n in name.choices if n != name.answer)
    assert not grade(name, f"Answer: {name.answer} or {other}")
    assert grade(text, f"Answer: `{text.answer}`")


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
