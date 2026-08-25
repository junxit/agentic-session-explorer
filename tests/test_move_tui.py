"""TUI regressions for the move bindings.

Covers the key rebinding (grouping moved off ``m`` to make room for it), that
``m`` actually reaches the move dialog, and that the dialog will not let a move
be confirmed that the user has not previewed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sx.adapters import claude as claude_mod
from sx.adapters.claude import ClaudeAdapter, encode_project_dir
from sx.grouping import GroupMode
from sx.service import DeleteService, MoveService
from sx.tui.app import SxApp
from sx.tui.screens import MoveScreen
from textual.widgets import Button, Input

SID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


@pytest.fixture(autouse=True)
def _no_real_stores(monkeypatch, tmp_path):
    """Keep the app away from the developer's real harness stores.

    ``SxApp()`` scans the registry on mount, and ``sx.memory`` resolves ``home``
    in its own namespace — patching an adapter's ``home`` does not cover it, so a
    memory test read the real ``~/.claude`` until this fixture existed.
    """
    monkeypatch.setattr("sx.tui.app.build_registry", lambda: ([], []))
    monkeypatch.setattr("sx.memory.home", lambda: tmp_path)


def _wire(app, tmp_path: Path, monkeypatch, project: Path):
    """Inject a Claude adapter over a synthetic store holding one session."""
    monkeypatch.setattr(claude_mod, "home", lambda: tmp_path)
    folder = tmp_path / ".claude" / "projects" / encode_project_dir(str(project))
    folder.mkdir(parents=True)
    transcript = folder / f"{SID}.jsonl"
    transcript.write_text(
        json.dumps({"type": "user", "cwd": str(project), "message": {"content": "hi"}})
        + "\n"
    )
    # Back-date it so the live-session guard does not demand a typed phrase;
    # that path has its own test below.
    os.utime(transcript, (1_700_000_000, 1_700_000_000))
    adapter = ClaudeAdapter()
    adapters = {adapter.name: adapter}
    app._adapters_by_name = adapters
    app._delete_service = DeleteService(adapters, log_path=tmp_path / "ops.log")
    app._move_service = MoveService(adapters, log_path=tmp_path / "ops.log")
    app._sessions_by_harness = {adapter.name: list(adapter.discover())}
    app._load_errors = []
    app._live_cache = {}
    app._populate_tree()
    return adapter


def test_both_move_keys_are_discoverable_in_the_footer():
    """``M`` is a primary action, not a vim-style alias, so the footer shows it.

    It is also the more consequential of the two — it moves a real directory on
    disk — so it must not be the one a user cannot find.
    """
    shown = {binding.key: binding.description for binding in SxApp.BINDINGS if binding.show}
    assert shown.get("m") == "Move"
    assert shown.get("M") == "Move dir"
    assert "j" not in shown  # navigation aliases stay hidden


@pytest.mark.asyncio
async def test_b_cycles_the_grouping_mode(tmp_path, monkeypatch):
    """Grouping moved from ``m`` to ``b`` so ``m``/``M`` could become move."""
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        _wire(app, tmp_path, monkeypatch, Path("/old/proj"))
        assert app._group_mode is GroupMode.PROJECT
        await pilot.press("b")
        await pilot.pause()
        assert app._group_mode is GroupMode.DATE


@pytest.mark.asyncio
async def test_keys_reach_their_bindings_at_startup(tmp_path, monkeypatch):
    """The tree holds focus on mount, not the hidden filter box.

    Textual focuses the first focusable widget, which is the filter Input. Its
    CSS hides it but it still received keys, so typing any binding at startup
    silently filled the invisible filter — ``oedrb`` set the filter to "oedrb"
    and fired none of orphans/export/delete/refresh/group.
    """
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        _wire(app, tmp_path, monkeypatch, Path("/old/proj"))
        await pilot.press("b")
        await pilot.pause()
        assert app._filter == ""
        assert app._group_mode is GroupMode.DATE


@pytest.mark.asyncio
async def test_m_opens_the_move_dialog_for_the_highlighted_session(tmp_path, monkeypatch):
    """``m`` reaches the dialog, pre-filled with the project it would move."""
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        _wire(app, tmp_path, monkeypatch, Path("/old/proj"))
        session = app._sessions_by_harness["claude"][0]
        monkeypatch.setattr(app, "_current_session", lambda: session)

        app.action_move_sessions()
        for _ in range(4):
            await pilot.pause()

        assert isinstance(app.screen, MoveScreen)
        assert app.screen.query_one("#destination", Input).value == "/old/proj"
        assert app.screen.query_one("#confirm", Button).disabled is True


@pytest.mark.asyncio
async def test_move_stays_disabled_until_the_plan_has_been_previewed(tmp_path, monkeypatch):
    """Confirming a destination nobody has seen a plan for is not possible."""
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        _wire(app, tmp_path, monkeypatch, Path("/old/proj"))
        session = app._sessions_by_harness["claude"][0]
        monkeypatch.setattr(app, "_current_session", lambda: session)

        app.action_move_sessions()
        for _ in range(4):
            await pilot.pause()
        screen = app.screen
        destination = screen.query_one("#destination", Input)
        confirm = screen.query_one("#confirm", Button)

        destination.value = str(tmp_path / "elsewhere")
        await pilot.pause()
        assert confirm.disabled is True

        screen._run_preview()
        await pilot.pause()
        assert confirm.disabled is False

        # Editing the destination invalidates the plan that was approved.
        destination.value = str(tmp_path / "somewhere-else")
        await pilot.pause()
        assert confirm.disabled is True


@pytest.mark.asyncio
async def test_moving_with_no_session_selected_says_so(tmp_path, monkeypatch):
    """The action must not open a dialog it cannot fill in."""
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        _wire(app, tmp_path, monkeypatch, Path("/old/proj"))
        monkeypatch.setattr(app, "_current_session", lambda: None)

        app.action_move_sessions()
        for _ in range(4):
            await pilot.pause()

        assert not isinstance(app.screen, MoveScreen)


@pytest.mark.asyncio
async def test_a_live_session_requires_the_typed_phrase(tmp_path, monkeypatch):
    """Rewriting a transcript the harness is writing needs a deliberate MOVE."""
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        _wire(app, tmp_path, monkeypatch, Path("/old/proj"))
        session = app._sessions_by_harness["claude"][0]
        # Mark it live, the way an in-progress conversation would be.
        os.utime(session.primary_path, None)
        monkeypatch.setattr(app, "_current_session", lambda: session)

        app.action_move_sessions()
        for _ in range(4):
            await pilot.pause()
        screen = app.screen
        screen.query_one("#destination", Input).value = str(tmp_path / "elsewhere")
        screen._run_preview()
        await pilot.pause()

        confirm = screen.query_one("#confirm", Button)
        phrase = screen.query_one("#confirm-input", Input)
        assert phrase.display is True
        assert confirm.disabled is True

        phrase.value = "MOVE"
        await pilot.pause()
        assert confirm.disabled is False


@pytest.mark.asyncio
async def test_n_opens_the_memory_browser(tmp_path, monkeypatch):
    """Memory has no home in the session tree, so it gets its own screen."""
    from sx.adapters import claude as claude_mod
    from sx.tui.screens import MemoryScreen

    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        _wire(app, tmp_path, monkeypatch, Path("/old/proj"))
        folder = tmp_path / ".claude" / "projects" / encode_project_dir("/old/proj")
        (folder / "memory").mkdir(parents=True, exist_ok=True)
        (folder / "memory" / "a-fact.md").write_text(
            "---\nname: a-fact\ndescription: something learned\n---\nbody\n"
        )

        await pilot.press("n")
        for _ in range(4):
            await pilot.pause()

        assert isinstance(app.screen, MemoryScreen)
        assert [m.name for m in app.screen._memories] == ["a-fact"]


@pytest.mark.asyncio
async def test_the_project_cleanup_box_appears_only_on_the_last_session(
    tmp_path, monkeypatch
):
    """Memory is offered when nothing else references the project, never before."""
    from sx.tui.screens import ConfirmDeleteScreen
    from textual.widgets import Checkbox

    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        _wire(app, tmp_path, monkeypatch, Path("/old/proj"))
        folder = tmp_path / ".claude" / "projects" / encode_project_dir("/old/proj")
        (folder / "memory").mkdir(parents=True, exist_ok=True)
        (folder / "memory" / "a-fact.md").write_text("---\nname: a-fact\n---\nbody\n")

        # A second session for the same project: not the last one.
        (folder / "bbbbbbbb-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl").write_text(
            json.dumps({"type": "user", "cwd": "/old/proj", "message": {"content": "x"}})
            + "\n"
        )
        adapter = app._adapters_by_name["claude"]
        app._sessions_by_harness["claude"] = list(adapter.discover())
        session = app._sessions_by_harness["claude"][0]
        monkeypatch.setattr(app, "_current_session", lambda: session)

        app.action_delete_session()
        for _ in range(4):
            await pilot.pause()
        assert isinstance(app.screen, ConfirmDeleteScreen)
        assert not app.screen.query("#bundle")
        app.screen.dismiss(None)
        for _ in range(3):
            await pilot.pause()

        # Now it is the only session left.
        (folder / "bbbbbbbb-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl").unlink()
        app._sessions_by_harness["claude"] = list(adapter.discover())
        app.action_delete_session()
        for _ in range(4):
            await pilot.pause()
        assert isinstance(app.screen, ConfirmDeleteScreen)
        box = app.screen.query_one("#bundle", Checkbox)
        assert box.value is False, "memory must never be pre-selected for deletion"
        assert "memory" in str(box.label)
