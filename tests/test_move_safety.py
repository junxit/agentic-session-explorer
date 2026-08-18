"""Safety invariants for the move operation.

A move is meant to be lossless and reversible. These tests hold it to that: it
must never overwrite, never truncate, never reach outside a harness's store, and
never report work it did not do.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sx.adapters import claude as claude_mod
from sx.adapters.base import HarnessAdapter
from sx.adapters.claude import ClaudeAdapter, encode_project_dir
from sx.model import Capability, MovePlan, MoveResult
from sx.service import MoveService
from sx.util import is_within, rewrite_jsonl

OLD = Path("/old/proj")
NEW = Path("/new/proj")
SID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


class _RefusingAdapter(HarnessAdapter):
    """An adapter that reports work and then refuses all of it."""

    name = "refusing"
    display = "Refusing"
    capabilities = Capability.BROWSE | Capability.MOVE

    def store_roots(self):
        return [Path("/nowhere")]

    def available(self):
        return True

    def discover(self):
        return iter(())

    def load(self, session):
        return []

    def move(self, plan, *, dry_run=False):
        return MoveResult(dry_run=dry_run, skipped={Path("/x"): "refused by policy"})


def _claude_store(tmp_path: Path, monkeypatch) -> tuple[ClaudeAdapter, Path]:
    """Build a one-session Claude store and return the adapter and transcript."""
    monkeypatch.setattr(claude_mod, "home", lambda: tmp_path)
    folder = tmp_path / ".claude" / "projects" / encode_project_dir(str(OLD))
    folder.mkdir(parents=True)
    transcript = folder / f"{SID}.jsonl"
    transcript.write_text(
        json.dumps({"type": "user", "cwd": str(OLD), "message": {"content": "hi"}}) + "\n"
    )
    return ClaudeAdapter(), transcript


# --- failure is never reported as success ----------------------------------


def test_refused_move_is_marked_failed_not_silent():
    """Nothing done plus something refused is a failure, not a quiet success."""
    result = MoveResult(skipped={Path("/x"): "outside store root (refused)"})
    assert result.failed is True
    assert "refused" in result.summary()


def test_absent_targets_are_not_counted_as_refusals():
    """A path that simply is not there is not a refusal worth alarming about."""
    result = MoveResult(skipped={Path("/x"): "does not exist"})
    assert result.refused == {}
    assert result.failed is False


def test_note_alone_counts_as_work_done():
    """Row-only harnesses report their work in ``note`` and have no paths."""
    result = MoveResult(note="re-pointed 3 database column(s)")
    assert result.failed is False
    assert "3 database" in result.summary()


def test_service_reports_a_refusing_adapter_honestly():
    """An adapter that refuses everything must surface as a failed result."""
    adapter = _RefusingAdapter()
    service = MoveService({adapter.name: adapter})
    plan = MovePlan(harness=adapter.name, old=OLD, new=NEW)
    results = service.move({adapter.name: plan}, dry_run=True)
    assert results[adapter.name].failed is True


# --- the rewrite must never lose data --------------------------------------


def test_unparseable_lines_survive_a_rewrite_verbatim(tmp_path: Path):
    """iter_jsonl drops bad lines; a rewrite built on that would delete them."""
    path = tmp_path / "s.jsonl"
    original = (
        b'{"cwd":"/old/proj"}\n'
        b'{ half-written line\n'
        b"\n"
        b'{"cwd":"/other"}\n'
    )
    path.write_bytes(original)

    def transform(obj):
        return {"cwd": "/new/proj"} if obj.get("cwd") == "/old/proj" else None

    changed, error = rewrite_jsonl(path, transform)
    body = path.read_bytes()

    assert (changed, error) == (1, None)
    assert b"{ half-written line\n" in body
    assert b'{"cwd":"/other"}\n' in body
    assert body.count(b"\n") == original.count(b"\n")


def test_rewrite_abandons_the_write_when_the_source_changes(tmp_path: Path):
    """A harness appending mid-rewrite must not have its turns clobbered."""
    path = tmp_path / "s.jsonl"
    path.write_text(json.dumps({"cwd": "/old/proj"}) + "\n")

    appended = False

    def transform(obj):
        # Simulate the harness appending a turn while the rewrite is in flight.
        nonlocal appended
        if not appended:
            appended = True
            with path.open("a") as fh:
                fh.write(json.dumps({"cwd": "/old/proj", "late": True}) + "\n")
            os.utime(path, (0, 0))
        return {"cwd": "/new/proj"}

    changed, error = rewrite_jsonl(path, transform)

    assert changed == 0
    assert error is not None and "changed" in error
    assert '"late": true' in path.read_text()
    assert "/old/proj" in path.read_text()


def test_a_rewrite_leaves_no_temporary_files_behind(tmp_path: Path):
    """The temp file is always cleaned up, including when nothing changed."""
    path = tmp_path / "s.jsonl"
    path.write_text(json.dumps({"cwd": "/unrelated"}) + "\n")
    rewrite_jsonl(path, lambda obj: None)
    assert [p.name for p in tmp_path.iterdir()] == ["s.jsonl"]


def test_round_trip_restores_the_transcript_byte_for_byte(tmp_path, monkeypatch):
    """A move is reversible: the inverse move restores the original bytes."""
    adapter, transcript = _claude_store(tmp_path, monkeypatch)
    original = transcript.read_bytes()

    sessions = adapter.sessions_for_project(OLD)
    adapter.move(adapter.plan_move(sessions, OLD, NEW))
    back = adapter.sessions_for_project(NEW)
    adapter.move(adapter.plan_move(back, NEW, OLD))

    restored = tmp_path / ".claude" / "projects" / encode_project_dir(str(OLD)) / f"{SID}.jsonl"
    assert restored.read_bytes() == original


# --- containment -----------------------------------------------------------


def test_every_planned_relocation_stays_inside_the_store_roots(tmp_path, monkeypatch):
    """The same invariant the delete guard enforces, applied to relocations."""
    adapter, _ = _claude_store(tmp_path, monkeypatch)
    plan = adapter.plan_move(adapter.sessions_for_project(OLD), OLD, NEW)
    roots = adapter.store_roots()

    assert plan.relocations
    for source, destination in plan.relocations:
        assert is_within(source, roots)
        assert is_within(destination, roots)
    for path in plan.rewrites:
        assert is_within(path, roots)


def test_claude_config_is_left_alone_unless_opted_in(tmp_path, monkeypatch):
    """~/.claude.json belongs to a running harness; it moves only on request."""
    adapter, _ = _claude_store(tmp_path, monkeypatch)
    config = tmp_path / ".claude.json"
    config.write_text(json.dumps({"projects": {str(OLD): {"allowedTools": []}}}))

    sessions = adapter.sessions_for_project(OLD)
    adapter.move(adapter.plan_move(sessions, OLD, NEW))
    assert str(OLD) in json.loads(config.read_text())["projects"]


def test_claude_config_is_re_pointed_when_requested(tmp_path, monkeypatch):
    """With the opt-in set, the project keeps its trust decision and allowlist."""
    adapter, _ = _claude_store(tmp_path, monkeypatch)
    config = tmp_path / ".claude.json"
    config.write_text(json.dumps({"projects": {str(OLD): {"allowedTools": ["Bash"]}}}))
    history = tmp_path / ".claude" / "history.jsonl"
    history.write_text(json.dumps({"display": "hi", "project": str(OLD)}) + "\n")

    sessions = adapter.sessions_for_project(OLD)
    plan = adapter.plan_move(sessions, OLD, NEW)
    plan.include_config = True
    adapter.move(plan)

    projects = json.loads(config.read_text())["projects"]
    assert projects == {str(NEW): {"allowedTools": ["Bash"]}}
    assert json.loads(history.read_text())["project"] == str(NEW)


def test_claude_config_refuses_to_merge_over_existing_settings(tmp_path, monkeypatch):
    """If the destination already has settings, they are never overwritten."""
    adapter, _ = _claude_store(tmp_path, monkeypatch)
    config = tmp_path / ".claude.json"
    config.write_text(
        json.dumps({"projects": {str(OLD): {"a": 1}, str(NEW): {"keep": True}}})
    )

    sessions = adapter.sessions_for_project(OLD)
    plan = adapter.plan_move(sessions, OLD, NEW)
    plan.include_config = True
    result = adapter.move(plan)

    assert json.loads(config.read_text())["projects"][str(NEW)] == {"keep": True}
    assert any("already has Claude settings" in r for r in result.refused.values())


# --- relocating the project directory itself -------------------------------


@pytest.fixture()
def service(tmp_path, monkeypatch):
    """A MoveService over a Claude adapter rooted in ``tmp_path``."""
    monkeypatch.setattr(claude_mod, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "fake-home"))
    adapter = ClaudeAdapter()
    return MoveService({adapter.name: adapter}, log_path=tmp_path / "op.log")


def test_relocation_refuses_a_destination_inside_the_source(tmp_path, service):
    """Moving a directory into itself would recurse; it is refused up front."""
    source = tmp_path / "project"
    source.mkdir()
    assert "inside" in (service.check_relocation(source, source / "inner") or "")


def test_relocation_refuses_a_non_empty_destination(tmp_path, service):
    """A move never merges into or overwrites an existing directory."""
    source = tmp_path / "project"
    source.mkdir()
    destination = tmp_path / "taken"
    destination.mkdir()
    (destination / "file").write_text("keep me")
    assert "not empty" in (service.check_relocation(source, destination) or "")


def test_relocation_refuses_the_home_directory(tmp_path, service, monkeypatch):
    """Moving home would drag every harness store along with it."""
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    assert service.check_relocation(fake_home, tmp_path / "elsewhere") is not None


def test_relocation_refuses_a_directory_containing_a_session_store(tmp_path, service):
    """``~/.claude/projects`` lives under tmp_path here, so tmp_path is off limits."""
    reason = service.check_relocation(tmp_path, tmp_path.parent / "moved-store")
    assert reason is not None and "session store" in reason


def test_relocation_moves_the_directory_and_logs_both_endpoints(tmp_path, service):
    """The op-log records where it came from, which is what makes it reversible."""
    source = tmp_path / "project"
    (source / "src").mkdir(parents=True)
    (source / "src" / "app.py").write_text("print('x')\n")
    destination = tmp_path / "moved"

    assert service.relocate_project(source, destination) is None
    assert (destination / "src" / "app.py").read_text() == "print('x')\n"
    assert not source.exists()

    entry = json.loads((tmp_path / "op.log").read_text().strip())
    assert entry["action"] == "move_project"
    assert (entry["old"], entry["new"]) == (str(source), str(destination))
