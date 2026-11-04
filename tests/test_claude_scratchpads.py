"""Claude's scratchpads are shared by every Claude config, so cleanup never takes them.

``/private/tmp/claude-<uid>`` follows the user id, not the config home: every
Claude config on the account (``CLAUDE_CONFIG_DIR``, another ``HOME``) writes
its sessions' scratchpads there. One config therefore cannot tell another
config's scratchpads from stale ones. ``sx`` once offered every scratchpad
whose session the config it read did not know — all of them, for a config with
no sessions — and 24 real scratchpads were deleted that way during a test run
under a fake ``HOME``. Everything here is synthetic under ``tmp_path``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from sx.adapters import claude as claude_mod
from sx.adapters.claude import ClaudeAdapter, encode_project_dir
from sx.model import OrphanKind
from sx.util import scratchpad_root as real_scratchpad_root

PROJECT = "/work/proj"
MINE = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
THEIRS = "99999999-9999-9999-9999-999999999999"


def _idle(*paths: Path) -> None:
    """Backdate a tree so nothing in it counts as recently written."""
    past = time.time() - 7 * 24 * 3600
    for root in paths:
        for top, dirs, files in os.walk(root, topdown=False):
            for name in (*files, *dirs):
                os.utime(Path(top, name), (past, past), follow_symlinks=False)
        os.utime(root, (past, past), follow_symlinks=False)


def _config(tmp_path: Path, monkeypatch, *, sessions: tuple[str, ...]) -> tuple[Path, Path]:
    """Build a Claude config home with ``sessions`` and a shared scratchpad root.

    Returns:
        ``(config dir, scratchpad project folder)``.
    """
    monkeypatch.setattr(claude_mod, "home", lambda: tmp_path)
    scratch = tmp_path / "scratch"
    monkeypatch.setattr(claude_mod, "scratchpad_root", lambda: scratch)
    base = tmp_path / ".claude"
    folder = base / "projects" / encode_project_dir(PROJECT)
    folder.mkdir(parents=True)
    for sid in sessions:
        (folder / f"{sid}.jsonl").write_text(
            json.dumps({"type": "user", "cwd": PROJECT, "message": {"content": "hi"}}) + "\n",
            encoding="utf-8",
        )
    project_scratch = scratch / encode_project_dir(PROJECT)
    for sid in (*sessions, THEIRS):
        (project_scratch / sid).mkdir(parents=True)
        (project_scratch / sid / "tool-output.txt").write_text("x" * 1024, encoding="utf-8")
    _idle(tmp_path)
    return base, project_scratch


def _offered(adapter: ClaudeAdapter) -> list[Path]:
    """Every path the orphan scan offers."""
    return [path for orphan in adapter.find_orphans() for path in orphan.paths]


def test_a_config_with_no_sessions_offers_no_scratchpads(tmp_path, monkeypatch):
    """The incident: an empty config saw every scratchpad as stale."""
    _, project_scratch = _config(tmp_path, monkeypatch, sessions=())

    offered = _offered(ClaudeAdapter())

    assert not any(path.is_relative_to(tmp_path / "scratch") for path in offered), offered
    assert (project_scratch / THEIRS).is_dir()


def test_another_configs_scratchpad_is_never_offered(tmp_path, monkeypatch):
    """A scratchpad whose session this config does not know may be another config's."""
    _config(tmp_path, monkeypatch, sessions=(MINE,))

    for orphan in ClaudeAdapter().find_orphans():
        assert orphan.kind is not OrphanKind.STALE_SESSION or not any(
            path.is_relative_to(tmp_path / "scratch") for path in orphan.paths
        ), orphan


def test_project_state_cleanup_leaves_the_shared_scratchpad_folder(tmp_path, monkeypatch):
    """The last session's cleanup never takes another config's scratchpads with it."""
    _, project_scratch = _config(tmp_path, monkeypatch, sessions=(MINE,))
    adapter = ClaudeAdapter()
    session = next(s for s in adapter.discover() if s.session_id == MINE)
    adapter.delete(session)

    leftovers = adapter.project_leftovers(PROJECT)
    assert not any(path.is_relative_to(tmp_path / "scratch") for path in leftovers.paths)
    adapter.delete_project_leftovers(leftovers)

    assert not (project_scratch / MINE).exists(), "a deleted session's own scratchpad goes"
    assert (project_scratch / THEIRS / "tool-output.txt").is_file()


def test_an_unreadable_project_folder_means_no_stale_artifacts(tmp_path, monkeypatch):
    """A session hidden behind a permission error must not make its artifacts look stale."""
    base, _ = _config(tmp_path, monkeypatch, sessions=(MINE,))
    (base / "session-env" / MINE).mkdir(parents=True)
    _idle(base / "session-env")
    folder = base / "projects" / encode_project_dir(PROJECT)
    folder.chmod(0o000)
    try:
        if os.access(folder, os.R_OK):
            pytest.skip("running with privileges that read any folder")
        offered = _offered(ClaudeAdapter())
    finally:
        folder.chmod(0o755)

    assert base / "session-env" / MINE not in offered


def test_the_suite_never_sees_the_real_scratchpad_root():
    """The conftest guard points Claude's scratchpad root into a sandbox for every test."""
    real = real_scratchpad_root()

    assert claude_mod.scratchpad_root() != real
    assert not str(claude_mod.scratchpad_root()).startswith(str(real))
