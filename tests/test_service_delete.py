"""Safety-critical tests for sx.service.DeleteService and the delete guard.

Ported from the service/delete portion of /tmp/verify_m4_core.py.

SAFETY: every file operated on here is created under pytest's ``tmp_path``.
The concrete adapter's ``store_roots`` points only at ``tmp_path/"store"``, so
no real ``~/.claude``, ``~/.codex`` or ``~/.gemini`` data is ever in scope.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import pytest

from sx.adapters.base import HarnessAdapter
from sx.model import Capability, Message, Role, Session
from sx.service import DeleteService


class _StoreAdapter(HarnessAdapter):
    """Concrete BROWSE|DELETE adapter rooted at a single tmp store dir."""

    name = "tmp"
    display = "Tmp"
    capabilities = Capability.BROWSE | Capability.DELETE

    def __init__(self, root: Path) -> None:
        """Store the single allowed root."""
        self._root = root

    def store_roots(self) -> list[Path]:
        """Return the one store root (everything else is refused)."""
        return [self._root]

    def discover(self):
        """Discovery is unused in these tests."""
        return iter(())

    def load(self, session: Session) -> list[Message]:
        """Return a fixed transcript (unused by delete tests)."""
        return [Message(Role.USER, text="hi")]


@pytest.fixture()
def store(tmp_path: Path) -> Path:
    """Create and return the tmp store root directory."""
    root = tmp_path / "store"
    root.mkdir()
    return root


@pytest.fixture()
def session_file(store: Path) -> Path:
    """Create a synthetic in-root session file and return its path."""
    proj = store / "proj"
    proj.mkdir()
    f = proj / "sess.jsonl"
    f.write_text('{"x": 1}\n', encoding="utf-8")
    return f


def _session(session_file: Path) -> Session:
    """Build a Session pointing at the in-root synthetic file."""
    return Session(
        harness="tmp",
        session_id="abcd1234",
        project_path="/some/proj",
        title="My Test Session",
        modified=datetime(2026, 5, 30, 12, 0),
        paths=[session_file],
    )


def _service(store: Path, tmp_path: Path) -> tuple[DeleteService, _StoreAdapter, Path]:
    """Build a DeleteService + adapter with an op-log under tmp_path."""
    adapter = _StoreAdapter(store)
    log_path = tmp_path / "ops.log"
    return DeleteService({"tmp": adapter}, log_path=log_path), adapter, log_path


# --- active detection ----------------------------------------------------


def test_is_active_fresh_file(store: Path, session_file: Path, tmp_path: Path):
    """A file with a current mtime is reported active."""
    svc, _, _ = _service(store, tmp_path)
    assert svc.is_active(_session(session_file)) is True


def test_is_active_old_file(store: Path, session_file: Path, tmp_path: Path):
    """A file last modified long ago is reported not active."""
    svc, _, _ = _service(store, tmp_path)
    old = time.time() - 10_000
    os.utime(session_file, (old, old))
    assert svc.is_active(_session(session_file)) is False


# --- dry-run preview -----------------------------------------------------


def test_preview_is_dry_run_and_keeps_file(store: Path, session_file: Path, tmp_path: Path):
    """preview reports dry_run, lists the file, frees bytes, removes nothing."""
    svc, _, _ = _service(store, tmp_path)
    result = svc.preview(_session(session_file))
    assert result.dry_run is True
    assert session_file in result.removed
    assert result.freed_bytes > 0
    assert session_file.exists()  # SAFETY: file untouched after preview


# --- allowlist refusal (safety) -----------------------------------------


def test_delete_refuses_path_outside_store_root(store: Path, tmp_path: Path):
    """A path outside store_roots is skipped with a refusal reason, untouched."""
    outside = tmp_path / "outside.txt"
    outside.write_text("do not delete me", encoding="utf-8")
    svc, _, _ = _service(store, tmp_path)
    bad = Session(harness="tmp", session_id="evil", paths=[outside])

    result = svc.delete(bad)
    assert outside in result.skipped
    assert "outside store root" in result.skipped[outside]
    assert result.removed == []
    assert outside.exists()  # SAFETY: outside file survives


# --- real delete + op-log ------------------------------------------------


def test_delete_in_root_file_and_logs(store: Path, session_file: Path, tmp_path: Path):
    """An in-root file is removed and a JSON op-log line is written."""
    svc, _, log_path = _service(store, tmp_path)
    session = _session(session_file)

    result = svc.delete(session)
    assert session_file in result.removed
    assert not session_file.exists()  # actually removed (in-root, synthetic)
    assert log_path.exists()

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    entry = json.loads(lines[-1])
    assert entry["harness"] == "tmp"
    assert entry["id"] == "abcd1234"


def test_preview_writes_no_log_line(store: Path, session_file: Path, tmp_path: Path):
    """A dry-run preview appends nothing to the op-log; a real delete does."""
    svc, _, log_path = _service(store, tmp_path)
    session = _session(session_file)

    svc.preview(session)
    assert not log_path.exists()  # preview logged nothing

    svc.delete(session)
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1  # exactly one line from the real delete
