"""Shared pytest helpers for the sx test suite.

Everything here operates on synthetic data under pytest's ``tmp_path`` fixture.
No test ever touches the user's real ``~/.claude``, ``~/.codex`` or
``~/.gemini`` data. ``CLAUDE_CONFIG_DIR`` is scrubbed for every test to keep
that true on a machine where the developer has relocated their own store.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_ui_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point UI state at ``tmp_path`` so tests never write the real ~/.config."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))


@pytest.fixture(autouse=True)
def _no_claude_config_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scrub ``CLAUDE_CONFIG_DIR`` so the suite never follows a real one.

    ``sx`` reads this variable the way Claude Code does. Without the scrub, a
    developer who exports it gets the Claude adapter building its store roots —
    the delete allowlist — and its ``.claude.json`` path from their live
    configuration directory while the test believes it is under ``tmp_path``,
    at which point a move test rewrites their real config. Tests that exercise
    the override set the variable themselves; ``monkeypatch`` is function
    scoped, so their ``setenv`` simply wins.
    """
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)


def write_jsonl(path: Path, records: list[dict]) -> Path:
    """Write a list of dicts as a JSONL file and return the path.

    Args:
        path: Destination file (parent dirs are created if needed).
        records: Objects to serialize, one JSON object per line.

    Returns:
        The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in records),
        encoding="utf-8",
    )
    return path
