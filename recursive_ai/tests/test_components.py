import copy
import json

import pytest

from recursive_ai.config import DEFAULT_CONFIG, ConfigError, ConfigStore, apply_change
from recursive_ai.memory import MemorySchemaError, MemoryStore
from recursive_ai.tools import ToolRouter, default_tools, safe_eval


@pytest.fixture
def config():
    return copy.deepcopy(DEFAULT_CONFIG)


def test_apply_change_does_not_mutate_input(config):
    new = apply_change(config, "prompt_addenda.responder", "append", "Hi.")
    assert new["prompt_addenda"]["responder"] == ["Hi."]
    assert config["prompt_addenda"]["responder"] == []


def test_apply_change_rejects_bad_values(config):
    with pytest.raises(ConfigError):
        apply_change(config, "memory_policy.retrieval_k", "set", 99)
    with pytest.raises(ConfigError):
        apply_change(config, "routing_rules", "append",
                     {"id": "arithmetic", "pattern": "x", "tool": "calculator",
                      "hint": "dup"})
    with pytest.raises(ConfigError):
        apply_change(config, "nope.path", "set", 1)


def test_remove_routing_rule_by_id(config):
    new = apply_change(config, "routing_rules", "remove", {"id": "clock"})
    assert "clock" not in [r["id"] for r in new["routing_rules"]]


def test_config_store_history_and_rollback(tmp_path, config):
    store = ConfigStore(tmp_path)
    changed = apply_change(config, "memory_policy.retrieval_k", "set", 4)
    assert store.commit(changed, "tune") == 2
    assert store.rollback(1, "undo") == 3
    assert store.current["memory_policy"]["retrieval_k"] == 3
    assert [h["version"] for h in store.history()] == [1, 2, 3]


def test_memory_schema_mismatch_blocks_loading(tmp_path):
    path = tmp_path / "memory.json"
    MemoryStore(path, schema_version=1)
    with pytest.raises(MemorySchemaError):
        MemoryStore(path, schema_version=2)


def test_memory_retrieves_only_successes(tmp_path, config):
    memory = MemoryStore(tmp_path / "m.json", 1)
    policy = config["memory_policy"]
    memory.add_episode({"request": "capital of France", "answer": "Paris",
                        "score": 0.95}, policy)
    memory.add_episode({"request": "capital of Spain", "answer": "?",
                        "score": 0.2}, policy)
    hits = memory.retrieve("What is the capital of France?", policy)
    assert [h["answer"] for h in hits] == ["Paris"]


@pytest.mark.parametrize("expr", ["__import__('os')", "open('x')",
                                  "(1).__class__", "2 ** 10000"])
def test_calculator_only_allows_arithmetic(expr):
    with pytest.raises(ValueError):
        safe_eval(expr)


def test_calculator_math():
    assert safe_eval("17 * 23") == 391
    assert safe_eval("(2 + 3) ^ 2") == 25


def test_router_enforces_permissions(tmp_path, config):
    router = ToolRouter(default_tools(MemoryStore(tmp_path / "m.json", 1)))
    config["tool_permissions"]["calculator"] = "deny"
    result = router.execute("calculator", {"expression": "1+1"}, config)
    assert not result.ok and "permission denied" in result.output
    assert "calculator" not in [d["name"]
                                for d in router.definitions(config)]
    # Default deny: a tool missing from the permission map cannot run.
    del config["tool_permissions"]["clock"]
    assert not router.execute("clock", {}, config).ok


def test_tool_definitions_are_strict(tmp_path, config):
    router = ToolRouter(default_tools(MemoryStore(tmp_path / "m.json", 1)))
    for definition in router.definitions(config):
        assert definition["strict"] is True
        assert definition["input_schema"]["additionalProperties"] is False
        json.dumps(definition)
