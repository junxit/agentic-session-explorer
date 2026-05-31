"""Regression tests for the orphan-cleanup screen's delete bindings.

These guard against the "action calls an async helper but never awaits it" bug,
where pressing ``d``/``D`` silently did nothing. The screen's delete helpers
must run as Textual workers so their confirmation/await chain executes.

All file operations use synthetic orphans under ``tmp_path``; the real
``~/.claude`` / ``~/.codex`` / ``~/.gemini`` stores are never touched.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from sx.adapters.base import HarnessAdapter
from sx.model import Capability, Message, Orphan, OrphanKind, Role
from sx.service import DeleteService
from sx.tui.app import SxApp
from sx.tui.screens import ConfirmDeleteScreen, OrphanScreen
from textual.widgets import Button, DataTable, Input


class _FakeAdapter(HarnessAdapter):
    """Synthetic adapter whose store root is a temp dir."""

    name = "fake"
    display = "Fake"
    capabilities = Capability.BROWSE | Capability.ORPHANS | Capability.DELETE

    def __init__(self, root, orphan_files):
        self._root = root
        self._orphan_files = orphan_files

    def store_roots(self):
        return [self._root]

    def discover(self):
        return iter(())

    def load(self, session):
        return [Message(Role.USER, text="hi", timestamp=datetime(2026, 5, 30, 12, 0))]

    def find_orphans(self):
        return [
            Orphan(
                harness="fake",
                kind=OrphanKind.STRAY_TEMP,
                paths=[p],
                reason=f"stray {p.name}",
                size_bytes=p.stat().st_size,
            )
            for p in self._orphan_files
            if p.exists()
        ]


def _wire(app, store, orphan_files, log_path):
    """Inject a synthetic adapter into the app, replacing the real registry."""
    adapter = _FakeAdapter(store, orphan_files)
    app._adapters_by_name = {"fake": adapter}
    app._delete_service = DeleteService({"fake": adapter}, log_path=log_path)
    app._sessions_by_harness = {}
    app._load_errors = []


@pytest.mark.asyncio
async def test_orphan_delete_selected_removes_one(tmp_path):
    """Pressing ``d`` confirms and deletes exactly the selected orphan."""
    store = tmp_path / "store"
    store.mkdir()
    a = store / "a.tmp"
    a.write_text("a")
    b = store / "b.tmp"
    b.write_text("b")

    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        _wire(app, store, [a, b], tmp_path / "ops.log")
        app.action_show_orphans()
        await pilot.pause()
        await pilot.pause()
        assert isinstance(app.screen, OrphanScreen)
        assert len(app.screen._orphans) == 2

        app.screen.query_one("#orphan-table", DataTable).focus()
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        # The bug: nothing happened. The fix: a confirm modal appears.
        assert isinstance(app.screen, ConfirmDeleteScreen)
        await pilot.click("#confirm")
        await pilot.pause()
        await pilot.pause()

        assert not a.exists()
        assert b.exists()
        assert isinstance(app.screen, OrphanScreen)
        assert len(app.screen._orphans) == 1


@pytest.mark.asyncio
async def test_orphan_delete_all_requires_phrase_then_removes_all(tmp_path):
    """Pressing ``D`` requires the typed phrase, then deletes every orphan."""
    store = tmp_path / "store"
    store.mkdir()
    files = []
    for name in ("a.tmp", "b.tmp", "c.tmp"):
        p = store / name
        p.write_text(name)
        files.append(p)

    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        _wire(app, store, files, tmp_path / "ops.log")
        app.action_show_orphans()
        await pilot.pause()
        await pilot.pause()
        app.screen.query_one("#orphan-table", DataTable).focus()
        await pilot.pause()

        await pilot.press("D")
        await pilot.pause()
        await pilot.pause()
        assert isinstance(app.screen, ConfirmDeleteScreen)
        confirm = app.screen.query_one("#confirm", Button)
        assert confirm.disabled is True  # gated on the typed phrase

        app.screen.query_one("#confirm-input", Input).focus()
        await pilot.pause()
        for ch in "DELETE 3":
            await pilot.press(ch if ch != " " else "space")
        await pilot.pause()
        assert confirm.disabled is False

        await pilot.click("#confirm")
        await pilot.pause()
        await pilot.pause()
        assert all(not p.exists() for p in files)


@pytest.mark.asyncio
async def test_orphan_delete_empty_is_graceful(tmp_path):
    """``d``/``D`` with no orphans must not crash and stays on the screen."""
    store = tmp_path / "store"
    store.mkdir()
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        _wire(app, store, [], tmp_path / "ops.log")
        app.action_show_orphans()
        await pilot.pause()
        await pilot.pause()
        assert isinstance(app.screen, OrphanScreen)
        await pilot.press("d")
        await pilot.pause()
        await pilot.press("D")
        await pilot.pause()
        assert isinstance(app.screen, OrphanScreen)
