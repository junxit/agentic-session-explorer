"""Tests for moving a project's sessions to a new directory.

Everything runs against synthetic stores under ``tmp_path``; each adapter's own
``home`` is monkeypatched (adapters do ``from ..util import home``, so the name
has to be replaced in the adapter's namespace, not in :mod:`sx.util`).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from sx.adapters import claude as claude_mod
from sx.adapters import gemini as gemini_mod
from sx.adapters import opencode as opencode_mod
from sx.adapters.claude import ClaudeAdapter, encode_project_dir
from sx.adapters.codex import CodexAdapter
from sx.adapters.gemini import GeminiAdapter
from sx.adapters.opencode import OpencodeAdapter
from sx.util import repoint

OLD = Path("/old/proj")
NEW = Path("/new/home/proj")
SID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def _write(path: Path, records: list[dict]) -> Path:
    """Write JSONL records to ``path``, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _cwds(path: Path) -> list[str]:
    """Return every ``cwd`` recorded in a Claude transcript, in order."""
    return [
        json.loads(line)["cwd"]
        for line in path.read_text().splitlines()
        if line.strip() and "cwd" in json.loads(line)
    ]


def _move(adapter, old=OLD, new=NEW, **kwargs):
    """Plan and perform a move for every session the adapter finds under ``old``."""
    sessions = adapter.sessions_for_project(old)
    plan = adapter.plan_move(sessions, old, new)
    for key, value in kwargs.items():
        setattr(plan, key, value)
    return plan, adapter.move(plan)


# --- the encoding is the linchpin of every Claude move ---------------------


@pytest.mark.parametrize(
    ("directory", "folder"),
    [
        ("/Users/Jade/source/junxit/agentic-session-explorer",
         "-Users-Jade-source-junxit-agentic-session-explorer"),
        ("/Users/Jade/.extra/source/git_clones/usgpo/api",
         "-Users-Jade--extra-source-git-clones-usgpo-api"),
        ("/Users/Jade/Documents/Professional and Social Memberships/IEEE",
         "-Users-Jade-Documents-Professional-and-Social-Memberships-IEEE"),
        ("/Users/Jade/source/junxit/forks--", "-Users-Jade-source-junxit-forks--"),
    ],
)
def test_encode_project_dir_reproduces_real_folder_names(directory, folder):
    """Dots, underscores and spaces all encode to hyphens, exactly as Claude does."""
    assert encode_project_dir(directory) == folder


def test_repoint_respects_path_boundaries():
    """A sibling whose name merely starts with the project is never re-pointed."""
    assert repoint("/a/foo", "/a/foo", "/b/bar") == "/b/bar"
    assert repoint("/a/foo/sub", "/a/foo", "/b/bar") == "/b/bar/sub"
    assert repoint("/a/foobar", "/a/foo", "/b/bar") is None
    assert repoint("/elsewhere", "/a/foo", "/b/bar") is None


# --- claude ----------------------------------------------------------------


def _claude_home(tmp_path: Path, monkeypatch, cwd: str = str(OLD)) -> ClaudeAdapter:
    """Build a Claude store holding one session, a sidecar and a memory folder."""
    monkeypatch.setattr(claude_mod, "home", lambda: tmp_path)
    folder = tmp_path / ".claude" / "projects" / encode_project_dir(cwd)
    _write(
        folder / f"{SID}.jsonl",
        [
            {"type": "mode", "mode": "default"},
            {"type": "user", "cwd": cwd, "message": {"content": "hi"}},
            {"type": "assistant", "cwd": cwd + "/sub", "message": {"content": []}},
            {"type": "user", "cwd": "/elsewhere", "message": {"content": "other"}},
        ],
    )
    (folder / SID / "subagents").mkdir(parents=True)
    (folder / SID / "subagents" / "a.jsonl").write_text('{"type":"user"}\n')
    (folder / "memory").mkdir()
    (folder / "memory" / "MEMORY.md").write_text("# Memory Index\n")
    return ClaudeAdapter()


def test_claude_move_rewrites_cwd_and_relocates_the_folder(tmp_path, monkeypatch):
    """The transcript is re-pointed and lands in the folder the new path encodes to."""
    adapter = _claude_home(tmp_path, monkeypatch)
    _, result = _move(adapter)

    projects = tmp_path / ".claude" / "projects"
    new_folder = projects / encode_project_dir(NEW)
    assert not (projects / encode_project_dir(OLD)).exists()
    assert _cwds(new_folder / f"{SID}.jsonl") == [
        "/new/home/proj",
        "/new/home/proj/sub",
        "/elsewhere",
    ]
    assert result.fields_updated == 2
    assert not result.refused


def test_claude_move_carries_the_sidecar_and_memory_directories(tmp_path, monkeypatch):
    """A folder relocation takes the session sidecar and the project memory with it."""
    adapter = _claude_home(tmp_path, monkeypatch)
    _move(adapter)

    new_folder = tmp_path / ".claude" / "projects" / encode_project_dir(NEW)
    assert (new_folder / SID / "subagents" / "a.jsonl").is_file()
    assert (new_folder / "memory" / "MEMORY.md").read_text() == "# Memory Index\n"


def test_claude_move_merges_into_an_existing_destination_folder(tmp_path, monkeypatch):
    """Running Claude at the new path already created the folder; entries merge in."""
    adapter = _claude_home(tmp_path, monkeypatch)
    destination = tmp_path / ".claude" / "projects" / encode_project_dir(NEW)
    _write(destination / "existing.jsonl", [{"type": "user", "cwd": str(NEW)}])

    _move(adapter)

    assert (destination / "existing.jsonl").is_file()
    assert (destination / f"{SID}.jsonl").is_file()
    assert (destination / "memory" / "MEMORY.md").is_file()
    assert not (tmp_path / ".claude" / "projects" / encode_project_dir(OLD)).exists()


def test_claude_move_refuses_to_overwrite_a_colliding_transcript(tmp_path, monkeypatch):
    """A name already taken at the destination is refused, never overwritten."""
    adapter = _claude_home(tmp_path, monkeypatch)
    destination = tmp_path / ".claude" / "projects" / encode_project_dir(NEW)
    _write(destination / f"{SID}.jsonl", [{"type": "user", "cwd": "untouched"}])

    _, result = _move(adapter)

    assert _cwds(destination / f"{SID}.jsonl") == ["untouched"]
    assert any("already exists" in reason for reason in result.refused.values())


def test_claude_stray_session_moves_individually_with_its_sidecar(tmp_path, monkeypatch):
    """A session in a folder named for another directory is relocated on its own."""
    monkeypatch.setattr(claude_mod, "home", lambda: tmp_path)
    projects = tmp_path / ".claude" / "projects"
    stale = projects / encode_project_dir("/some/former/name")
    _write(stale / f"{SID}.jsonl", [{"type": "user", "cwd": str(OLD)}])
    (stale / SID).mkdir()
    (stale / SID / "tool-results").mkdir()

    adapter = ClaudeAdapter()
    _, result = _move(adapter)

    destination = projects / encode_project_dir(NEW)
    assert (destination / f"{SID}.jsonl").is_file()
    assert (destination / SID / "tool-results").is_dir()
    assert result.note is not None and "individually" in result.note


def test_claude_correlated_paths_include_the_session_sidecar(tmp_path, monkeypatch):
    """Deleting a session must take its ``<session-id>/`` directory with it."""
    adapter = _claude_home(tmp_path, monkeypatch)
    session = next(iter(adapter.discover()))
    sidecar = session.primary_path.parent / SID
    assert sidecar in adapter.correlated_paths(session)


# --- codex -----------------------------------------------------------------


def test_codex_move_rewrites_every_cwd_and_leaves_the_file_in_place(tmp_path, monkeypatch):
    """Both the session_meta and every turn_context are re-pointed; nothing moves."""
    monkeypatch.setattr("sx.adapters.codex.home", lambda: tmp_path)
    rollout = tmp_path / ".codex" / "sessions" / "2026" / "01" / "01" / "rollout-x.jsonl"
    _write(
        rollout,
        [
            {"type": "session_meta", "payload": {"id": "cx", "cwd": str(OLD)}},
            {"type": "response_item", "payload": {"type": "message", "role": "user"}},
            {"type": "turn_context", "payload": {"cwd": str(OLD)}},
            {"type": "turn_context", "payload": {"cwd": str(OLD) + "/deep"}},
            {"type": "turn_context", "payload": {"cwd": "/unrelated"}},
        ],
    )

    _, result = _move(CodexAdapter())

    payload_cwds = [
        json.loads(line)["payload"].get("cwd") for line in rollout.read_text().splitlines()
    ]
    assert payload_cwds == [
        "/new/home/proj", None, "/new/home/proj", "/new/home/proj/deep", "/unrelated"
    ]
    assert result.fields_updated == 3
    assert not result.moved


# --- gemini ----------------------------------------------------------------


def test_gemini_move_updates_the_marker_and_the_registry(tmp_path, monkeypatch):
    """Gemini records the path in two registries; both have to agree afterwards."""
    monkeypatch.setattr(gemini_mod, "home", lambda: tmp_path)
    chats = tmp_path / ".gemini" / "history" / "proj" / "chats"
    _write(chats / "session-1.jsonl", [{"sessionId": "g-1"}])
    marker = chats.parent / ".project_root"
    marker.write_text(str(OLD) + "\n")
    registry = tmp_path / ".gemini" / "projects.json"
    registry.write_text(json.dumps({"projects": {str(OLD): "proj"}}))

    _, result = _move(GeminiAdapter())

    assert marker.read_text() == str(NEW) + "\n"
    assert json.loads(registry.read_text())["projects"] == {str(NEW): "proj"}
    assert result.fields_updated == 2


# --- opencode --------------------------------------------------------------


def _opencode_home(tmp_path: Path, monkeypatch) -> OpencodeAdapter:
    """Build a minimal opencode database with one session at ``OLD``."""
    monkeypatch.setattr(opencode_mod, "home", lambda: tmp_path)
    store = tmp_path / ".local" / "share" / "opencode"
    (store / "storage" / "session_diff").mkdir(parents=True)
    con = sqlite3.connect(store / "opencode.db")
    con.executescript(
        """
        CREATE TABLE project (id text PRIMARY KEY, worktree text NOT NULL);
        CREATE TABLE session (id text PRIMARY KEY, project_id text, parent_id text,
            directory text NOT NULL, path text, title text,
            time_created integer, time_updated integer);
        CREATE TABLE workspace (id text PRIMARY KEY, directory text, project_id text);
        """
    )
    con.execute("INSERT INTO project VALUES ('p1', ?)", (str(OLD),))
    con.execute(
        "INSERT INTO session VALUES ('ses_1','p1',NULL,?,?,'t',1780000000000,1780000000000)",
        (str(OLD), str(OLD) + "/x"),
    )
    con.execute("INSERT INTO workspace VALUES ('w1', ?, 'p1')", (str(OLD),))
    con.commit()
    con.close()
    return OpencodeAdapter()


def test_opencode_move_updates_session_columns(tmp_path, monkeypatch):
    """The session's directory and path, and its workspace, all follow the move."""
    adapter = _opencode_home(tmp_path, monkeypatch)
    _, result = _move(adapter)

    con = sqlite3.connect(tmp_path / ".local" / "share" / "opencode" / "opencode.db")
    directory, path = con.execute("SELECT directory, path FROM session").fetchone()
    (workspace,) = con.execute("SELECT directory FROM workspace").fetchone()
    con.close()

    assert (directory, path, workspace) == (str(NEW), str(NEW) + "/x", str(NEW))
    assert result.fields_updated == 3
    assert result.note is not None


def test_opencode_move_never_touches_the_shared_project_row(tmp_path, monkeypatch):
    """A project row is shared by other sessions, so a move states it is left alone."""
    adapter = _opencode_home(tmp_path, monkeypatch)
    plan, _ = _move(adapter)

    con = sqlite3.connect(tmp_path / ".local" / "share" / "opencode" / "opencode.db")
    (worktree,) = con.execute("SELECT worktree FROM project").fetchone()
    con.close()

    assert worktree == str(OLD)
    assert "shared project row" in (plan.note or "")


# --- end to end ------------------------------------------------------------


def test_session_is_grouped_under_the_new_project_after_a_move(tmp_path, monkeypatch):
    """The whole point: the session stops pointing at a directory that moved."""
    adapter = _claude_home(tmp_path, monkeypatch)
    target = tmp_path / "real-project"
    target.mkdir()

    sessions = adapter.sessions_for_project(OLD)
    adapter.move(adapter.plan_move(sessions, OLD, target))

    moved = next(iter(adapter.discover()))
    assert moved.project_path == str(target)
    assert moved.is_orphan is False
