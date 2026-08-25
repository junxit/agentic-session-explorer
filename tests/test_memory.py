"""Tests for the project-memory browser.

Memory is what a harness keeps *between* sessions, so these cover reading it,
attributing it to the session that wrote it, archiving it, and the one path that
removes it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sx.adapters import claude as claude_mod
from sx.adapters.claude import ClaudeAdapter, encode_project_dir
from sx.memory import (
    discover_memories,
    export_memory,
    memories_for_project,
    parse_frontmatter,
)
from sx.service import DeleteService

PROJECT = "/work/proj"
SID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
GONE = "99999999-9999-9999-9999-999999999999"

DOC = """---
name: {name}
description: "On-disk session formats, verified by sampling"
metadata:
  node_type: memory
  type: reference
  originSessionId: {sid}
---

The body of the memory.
"""


def _store(tmp_path: Path, monkeypatch) -> Path:
    """Build a Claude store with one session and three memory documents."""
    monkeypatch.setattr(claude_mod, "home", lambda: tmp_path)
    folder = tmp_path / ".claude" / "projects" / encode_project_dir(PROJECT)
    (folder / "memory").mkdir(parents=True)
    (folder / f"{SID}.jsonl").write_text(
        json.dumps({"type": "user", "cwd": PROJECT, "message": {"content": "hi"}}) + "\n"
    )
    (folder / "memory" / "harness-formats.md").write_text(
        DOC.format(name="harness-formats", sid=SID)
    )
    (folder / "memory" / "stale-note.md").write_text(
        DOC.format(name="stale-note", sid=GONE)
    )
    (folder / "memory" / "MEMORY.md").write_text("# Memory Index\n\n- [a](a.md) — hook\n")
    return tmp_path / ".claude" / "projects"


# --- frontmatter -----------------------------------------------------------


def test_frontmatter_is_flattened_and_unquoted():
    """Nested metadata keys are needed by name, so nesting is ignored."""
    fields = parse_frontmatter(DOC.format(name="harness-formats", sid=SID))
    assert fields["name"] == "harness-formats"
    assert fields["description"] == "On-disk session formats, verified by sampling"
    assert fields["type"] == "reference"
    assert fields["originSessionId"] == SID


def test_a_document_without_frontmatter_still_parses():
    """An index file is plain Markdown; it must not break discovery."""
    assert parse_frontmatter("# Memory Index\n\n- [a](a.md)\n") == {}


def test_frontmatter_scan_is_bounded():
    """A document whose body contains ``key: value`` lines cannot fake fields."""
    text = "---\nname: real\n---\n" + "\n".join(f"fake{i}: x" for i in range(200))
    assert parse_frontmatter(text) == {"name": "real"}


# --- discovery -------------------------------------------------------------


def test_discovery_attributes_each_memory_to_its_session(tmp_path, monkeypatch):
    """originSessionId is present on most memories and is worth surfacing."""
    root = _store(tmp_path, monkeypatch)
    found = {m.name: m for m in discover_memories(root)}

    assert found["harness-formats"].origin_session_id == SID
    assert found["harness-formats"].origin_exists is True
    assert found["harness-formats"].kind == "reference"


def test_a_memory_whose_session_was_deleted_says_so(tmp_path, monkeypatch):
    """The whole point of the browser: memory that outlived its conversation."""
    root = _store(tmp_path, monkeypatch)
    stale = next(m for m in discover_memories(root) if m.name == "stale-note")

    assert stale.origin_exists is False
    assert stale.origin_label.endswith("(deleted)")


def test_an_index_file_lists_without_an_origin(tmp_path, monkeypatch):
    """MEMORY.md carries no frontmatter; it must still appear."""
    root = _store(tmp_path, monkeypatch)
    index = next(m for m in discover_memories(root) if m.path.name == "MEMORY.md")

    assert index.kind == "index"
    assert index.origin_session_id is None
    assert index.origin_label == "—"


def test_memories_resolve_to_the_project_that_owns_them(tmp_path, monkeypatch):
    """Grouping is by the cwd recorded in the transcript, not the lossy folder name."""
    root = _store(tmp_path, monkeypatch)
    assert {m.project_path for m in discover_memories(root)} == {PROJECT}
    assert len(memories_for_project(PROJECT, root)) == 3


def test_discovery_is_empty_when_no_project_has_memory(tmp_path, monkeypatch):
    """No memory is not an error."""
    monkeypatch.setattr(claude_mod, "home", lambda: tmp_path)
    (tmp_path / ".claude" / "projects").mkdir(parents=True)
    assert discover_memories(tmp_path / ".claude" / "projects") == []


# --- export and delete ------------------------------------------------------


def test_export_copies_the_document_verbatim(tmp_path, monkeypatch):
    """An archive that differs from the original is not an archive."""
    root = _store(tmp_path, monkeypatch)
    memory = next(m for m in discover_memories(root) if m.name == "harness-formats")

    written = export_memory(memory, tmp_path / "out")

    assert written.read_text() == memory.path.read_text()
    assert "harness-formats" in written.name


def test_export_never_overwrites(tmp_path, monkeypatch):
    """Two exports of one memory keep both, as the transcript exporter does."""
    root = _store(tmp_path, monkeypatch)
    memory = next(m for m in discover_memories(root) if m.name == "harness-formats")

    first = export_memory(memory, tmp_path / "out")
    second = export_memory(memory, tmp_path / "out")

    assert first != second
    assert first.exists() and second.exists()


def test_delete_removes_one_memory_and_logs_it(tmp_path, monkeypatch):
    """Memory removal gets its own op-log action, never folded into a session."""
    root = _store(tmp_path, monkeypatch)
    memory = next(m for m in discover_memories(root) if m.name == "stale-note")
    log = tmp_path / "ops.log"
    service = DeleteService({"claude": ClaudeAdapter()}, log_path=log)

    result = service.delete_memory(memory)

    assert not memory.path.exists()
    assert result.removed == [memory.path]
    entry = json.loads(log.read_text().strip())
    assert entry["action"] == "delete_memory"
    assert PROJECT in entry["title"]


def test_deleting_a_memory_leaves_its_siblings(tmp_path, monkeypatch):
    """One row, one file."""
    root = _store(tmp_path, monkeypatch)
    memories = discover_memories(root)
    service = DeleteService({"claude": ClaudeAdapter()}, log_path=tmp_path / "ops.log")

    service.delete_memory(next(m for m in memories if m.name == "stale-note"))

    assert len(discover_memories(root)) == len(memories) - 1


def test_memory_delete_goes_through_the_store_root_guard(tmp_path, monkeypatch):
    """A memory outside the projects store is refused, not removed."""
    _store(tmp_path, monkeypatch)
    outsider = tmp_path / "elsewhere.md"
    outsider.write_text("not ours")
    service = DeleteService({"claude": ClaudeAdapter()}, log_path=tmp_path / "ops.log")

    class _Fake:
        path = outsider
        origin_session_id = None
        name = "elsewhere"
        project_path = None

    result = service.delete_memory(_Fake())

    assert outsider.exists()
    assert result.failed is True


def test_discovery_never_falls_back_to_the_real_store(tmp_path, monkeypatch):
    """A guard against the isolation leak this module introduced once already.

    ``sx.memory`` resolves ``home`` in its own namespace, so patching an
    adapter's ``home`` does not cover it. A test that forgot would silently read
    the developer's real ``~/.claude`` and pass for the wrong reason.
    """
    monkeypatch.setattr("sx.memory.home", lambda: tmp_path)
    assert discover_memories() == []
