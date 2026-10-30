"""Tests for unattended practice, recurring-issue learning and the budget."""

from types import SimpleNamespace

import pytest
from conftest import ScriptedLLM

from recursive_ai.__main__ import BUDGET_STOPPED, main
from recursive_ai.evaluator import has_term
from recursive_ai.learning import RECURRING_MARK
from recursive_ai.llm import Budget, BudgetExceeded, ClaudeLLM, OfflineLLM
from recursive_ai.practice import PRACTICE_SET
from recursive_ai.safety import ChangeProposal

CASES = [
    {"id": "p-times", "request": "What is 12 times 7?",
     "expected": {"terms": ["84"]}},
    {"id": "p-sum", "request": "What is 17 + 25?",
     "expected": {"terms": ["42"]}},
]


def test_practice_learns_from_a_wrong_answer(make_ai):
    ai = make_ai()
    items = {i.case["id"]: i for i in ai.practice(10, CASES)}

    times = items["p-times"]
    assert not times.result.evaluation.passed
    assert [o.target for o in times.report.with_status("kept")] == [
        "routing_rules"]
    assert items["p-sum"].result.evaluation.passed
    assert "math-words" in [r["id"] for r in ai.store.current["routing_rules"]]


def test_practice_rotates_through_the_set(make_ai):
    ai = make_ai()
    first = [i.case["id"] for i in ai.practice(1, CASES)]
    second = [i.case["id"] for i in ai.practice(1, CASES)]
    assert first != second


def test_known_answer_never_reaches_self_correction(make_ai):
    wrong = {"answer": "12 times 7 is 85.", "confidence": 0.9,
             "evidence": []}
    llm = ScriptedLLM({"respond": [wrong]})
    ai = make_ai(llm)
    result = ai.run_pipeline("What is 12 times 7?", ai.store.current,
                             {"terms": ["84"]})

    assert "revise" not in llm.calls
    assert not result.evaluation.passed
    assert "expected_mismatch" in result.evaluation.modes
    assert not any("84" in i.detail for i in result.evaluation.issues)


def test_recurring_issue_on_passing_answers_triggers_learning(make_ai):
    mental_math = {"answer": "12 times 7 is 84.", "confidence": 0.95,
                   "evidence": []}
    llm = ScriptedLLM({"respond": [mental_math] * 3})
    ai = make_ai(llm)

    replies = [ai.handle("What is 12 times 7?") for _ in range(3)]
    assert all(r.passed for r in replies)
    assert replies[0].learning.outcomes == []
    assert replies[1].learning.outcomes == []
    assert [o.target for o in replies[2].learning.with_status("kept")] == [
        "routing_rules"]

    # The count starts again, so the next pass does not re-trigger.
    llm.script["respond"] = [mental_math]
    assert ai.handle("What is 12 times 7?").learning.outcomes == []
    assert ai.memory.count_since("ungrounded_math", RECURRING_MARK) == 1


def test_budget_caps_calls():
    budget = Budget(OfflineLLM(), max_calls=1, max_cost=1.0)
    call = {"purpose": "evaluate", "system": "",
            "messages": [{"role": "user", "content": "Answer:\nfine"}]}
    budget.generate(**call)
    with pytest.raises(BudgetExceeded):
        budget.generate(**call)


def test_budget_caps_estimated_cost():
    usage = SimpleNamespace(input_tokens=1000, output_tokens=500,
                            cache_creation_input_tokens=None,
                            cache_read_input_tokens=2000)
    response = SimpleNamespace(
        stop_reason="end_turn", model="claude-opus-5-5", usage=usage,
        content=[SimpleNamespace(type="text", text="{}")])
    client = SimpleNamespace(beta=SimpleNamespace(
        messages=SimpleNamespace(create=lambda **kwargs: response)))
    llm = ClaudeLLM(client=client, max_tokens=100)
    budget = Budget(llm, max_calls=10, max_cost=0.02)
    call = {"purpose": "plan", "system": "",
            "messages": [{"role": "user", "content": "hi"}]}

    budget.generate(**call)
    # $4/M input, $20/M output, $0.20/M cache reads for Claude Opus 5.5.
    assert llm.usage.cost == pytest.approx(0.0144)
    budget.generate(**call)
    with pytest.raises(BudgetExceeded):
        budget.generate(**call)


def test_budget_never_starts_a_call_that_could_cross_the_cap():
    calls = []
    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(
        create=lambda **kwargs: calls.append(kwargs))))
    # 16,000 output tokens at $20/M alone could cost $0.32.
    budget = Budget(ClaudeLLM(client=client), max_calls=10, max_cost=0.30)
    with pytest.raises(BudgetExceeded, match="could have gone over"):
        budget.generate(purpose="plan", system="",
                        messages=[{"role": "user", "content": "hi"}])
    assert calls == []


def test_proposals_only_mode_queues_every_change(make_ai):
    ai = make_ai()
    config = ai.store.current
    config["learning"]["auto_apply_max_risk"] = "none"
    version = ai.store.commit(config, "human: proposals-only")

    reply = ai.handle("What is 12 times 7?")
    assert reply.learning.outcomes
    assert {o.status for o in reply.learning.outcomes} == {
        "awaiting_approval"}
    assert ai.store.version == version


def test_budget_stop_rolls_back_and_requeues_approved_change(make_ai):
    ai = make_ai()
    ai.handle("What is 12 times 7?")
    [pending] = ai.pending_approvals()

    def out_of_budget(case, config):
        if config["planner"]["max_steps"] == 8:
            raise BudgetExceeded("cap")
        return 0.9

    ai.learner.run_case = out_of_budget
    with pytest.raises(BudgetExceeded):
        ai.approve(pending.id, approver="tester")
    assert ai.store.current["planner"]["max_steps"] == 5
    assert [p.id for p in ai.pending_approvals()] == [pending.id]

    ai.learner.run_case = lambda case, config: 0.9
    assert ai.approve(pending.id, approver="tester").status == "kept"
    assert ai.store.current["planner"]["max_steps"] == 8


def test_budget_stop_mid_retest_leaves_no_untested_change(make_ai):
    ai = make_ai()

    def out_of_budget(case, config):
        if config["prompt_addenda"]["responder"]:
            raise BudgetExceeded("cap")
        return 0.5

    ai.learner.run_case = out_of_budget
    with pytest.raises(BudgetExceeded):
        ai.learner.route(ChangeProposal("prompt_addenda.responder", "append",
                                        "Show units.", "test"))
    assert ai.store.current["prompt_addenda"]["responder"] == []
    # Not recorded as a failed change, so it can be proposed again.
    assert not any(p["mode"] == "change_rolled_back"
                   for p in ai.memory.data["patterns"]["error"])


def test_budget_stop_during_learning_still_returns_the_answer(make_ai):
    ai = make_ai(Budget(OfflineLLM(), max_calls=8, max_cost=1.0))
    reply = ai.handle("What is 12 times 7?")
    assert reply.ok
    assert reply.learning.stopped
    assert "[stopped]" in reply.text
    assert "math-words" not in [r["id"]
                                for r in ai.store.current["routing_rules"]]


@pytest.mark.parametrize("text,term,found", [
    ("It is 391.", "391", True),
    ("It is 3910.", "391", False),
    ("It is 0.391.", "391", False),
    ("1,391 apples", "391", False),
    ("There are 86,400 seconds.", "86400", True),
    ("The symbol is Au.", "Au", True),
    ("Because it is gold.", "Au", False),
    ("3/8 = 0.375", "0.375", True),
])
def test_has_term_matches_whole_words_and_numbers(text, term, found):
    assert has_term(text, term) is found


def test_practice_set_is_well_formed():
    ids = [c["id"] for c in PRACTICE_SET]
    assert len(ids) == len(set(ids)) >= 30
    for case in PRACTICE_SET:
        # An answer inside the question would pass a wrong answer that
        # repeats the question.
        for term in case["expected"]["terms"]:
            assert not has_term(case["request"], term), case["id"]


def test_cli_practice_reports_and_stops_at_the_cap(tmp_path, capsys):
    state = str(tmp_path / "state")
    assert main(["--offline", "--state-dir", state, "practice",
                 "--cases", "2"]) == 0
    out = capsys.readouterr().out
    assert "Practised 2:" in out and "API use:" in out

    assert main(["--offline", "--state-dir", state, "practice",
                 "--cases", "5", "--max-calls", "3"]) == BUDGET_STOPPED
    assert "Stopped: reached the cap of 3 API calls" in capsys.readouterr().out
