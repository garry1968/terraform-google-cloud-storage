"""Tests for saving the memory to a git repository."""

import subprocess

import pytest

from recursive_ai.__main__ import MEMORY_NOT_SAVED, MEMORY_REPO_ENV, main


@pytest.fixture(autouse=True)
def plain_git(tmp_path, monkeypatch):
    # Ignore the machine's git config: no identity, signing or helpers.
    config = tmp_path / "gitconfig"
    config.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "memory.git"
    subprocess.run(["git", "init", "-q", "--bare", str(path)], check=True)
    return str(path)


def cli(state, *args):
    return main(["--offline", "--state-dir", str(state), *args])


def saves(repo):
    return int(subprocess.run(
        ["git", "--git-dir", repo, "rev-list", "--count", "main"],
        capture_output=True, text=True, check=True).stdout)


def test_existing_memory_is_uploaded_and_later_commands_save(
        tmp_path, repo, capsys):
    state = tmp_path / "state"
    cli(state, "ask", "What is 17 * 23?")
    assert cli(state, "load", "--repo", repo) == 0
    assert "Memory saved." in capsys.readouterr().out
    assert saves(repo) == 1

    cli(state, "ask", "What is 3 + 4?")
    assert "Memory saved." in capsys.readouterr().out
    assert saves(repo) == 2
    cli(state, "history")  # read-only: nothing to save
    assert saves(repo) == 2


def test_new_machine_loads_the_saved_memory_by_itself(
        tmp_path, repo, monkeypatch, capsys):
    first = tmp_path / "first"
    cli(first, "ask", "What is 17 * 23?")
    cli(first, "load", "--repo", repo)

    monkeypatch.setenv(MEMORY_REPO_ENV, repo)
    second = tmp_path / "second"
    assert cli(second, "approvals") == 0
    assert "Loaded the saved memory." in capsys.readouterr().out
    assert (second / "memory.json").read_text() == \
        (first / "memory.json").read_text()


def test_commands_pick_up_newer_saves(tmp_path, repo):
    a, b = tmp_path / "a", tmp_path / "b"
    cli(a, "load", "--repo", repo)
    cli(a, "ask", "What is 17 * 23?")
    cli(b, "load", "--repo", repo)
    cli(a, "ask", "What is 3 + 4?")

    cli(b, "history")
    assert (b / "memory.json").read_text() == (a / "memory.json").read_text()


def test_load_refuses_to_mix_two_memories(tmp_path, repo, capsys):
    saved, other = tmp_path / "saved", tmp_path / "other"
    cli(saved, "ask", "What is 17 * 23?")
    cli(saved, "load", "--repo", repo)
    cli(other, "ask", "What is 3 + 4?")
    before = (other / "memory.json").read_text()

    assert cli(other, "load", "--repo", repo) == 1
    assert "has its own memory" in capsys.readouterr().out
    assert not (other / ".git").exists()
    assert (other / "memory.json").read_text() == before


def test_failed_save_keeps_the_memory_and_save_retries(
        tmp_path, repo, capsys):
    state = tmp_path / "state"
    cli(state, "ask", "What is 17 * 23?")
    cli(state, "load", "--repo", repo)

    def set_origin(url):
        subprocess.run(["git", "-C", str(state), "remote", "set-url",
                        "origin", url], check=True)

    set_origin(str(tmp_path / "missing.git"))
    assert cli(state, "ask", "What is 3 + 4?") == MEMORY_NOT_SAVED
    assert "Memory NOT saved" in capsys.readouterr().out
    assert saves(repo) == 1

    set_origin(repo)
    assert cli(state, "save") == 0
    assert "Memory saved." in capsys.readouterr().out
    assert saves(repo) == 2
    assert cli(state, "save") == 0
    assert "Nothing new to save." in capsys.readouterr().out
