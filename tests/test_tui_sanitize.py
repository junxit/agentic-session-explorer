"""TUI regressions: text read from session files cannot drive the terminal (T10, T8).

A transcript can carry escape sequences — copied from a web page, say — and
Pi titles a session with its first user line verbatim. The tree label always
sanitized it, but the delete preview, the tree's project groups, the move
dialog and plan, the orphan and memory screens and several toasts passed such
text on as it was. A Rich ``Text`` is not markup, but its characters reach the
terminal unchanged.

Toasts are markup, so their untrusted parts are also escaped: an export into a
directory named ``[red]…`` restyled the "Exported →" toast (T8).

Fake adapters plant the text under ``tmp_path``; the registry scan on mount is
patched out and the working directory is moved to ``tmp_path``.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest
from rich.text import Text
from textual.widgets import DataTable, Input, Label, Static, Tree

from sx.adapters.base import HarnessAdapter
from sx.memory import MemoryFile
from sx.model import (
    Capability,
    DeleteResult,
    Message,
    MovePlan,
    MoveResult,
    Orphan,
    OrphanKind,
    Role,
    Session,
)
from sx.service import DeleteService, MoveService
from sx.tui.app import SxApp
from sx.tui.screens import ConfirmDeleteScreen, MemoryScreen, MoveScreen

#: An OSC title change terminated by BEL, and one terminated by ST (ESC \).
OSC_BEL = "\x1b]0;PWNED\x07"
OSC_ST = "\x1b]0;PWNED\x1b\\"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Keep the app off the real stores and its exports out of the repository."""
    monkeypatch.setattr("sx.tui.app.build_registry", lambda: ([], []))
    monkeypatch.setattr("sx.memory.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)


def _clean(text: str) -> bool:
    """Return True if ``text`` holds no ESC or BEL."""
    return "\x1b" not in text and "\x07" not in text


class _Planted(HarnessAdapter):
    """A harness whose one session, plan and orphan carry escape sequences.

    Attributes:
        session_id: The session's id.
        title: The session's title.
        project: The project path the session records.
    """

    name = "planted"
    display = "Planted"
    capabilities = Capability.BROWSE | Capability.DELETE | Capability.MOVE | Capability.ORPHANS

    def __init__(
        self,
        root: Path,
        *,
        session_id: str = "s1",
        title: str = f"{OSC_BEL}hello title",
        project: str = "/proj",
    ) -> None:
        """Write the session's back-dated file."""
        self._root = root
        self.session_id = session_id
        self.title = title
        self.project = project
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "s1.jsonl"
        self.path.write_text("{}\n", encoding="utf-8")
        os.utime(self.path, (1_700_000_000, 1_700_000_000))

    def store_roots(self) -> list[Path]:
        """The synthetic store."""
        return [self._root]

    def discover(self):
        """Yield the one session."""
        yield Session(
            harness=self.name,
            session_id=self.session_id,
            project_path=self.project,
            title=self.title,
            modified=datetime(2026, 5, 30, 12, 0),
            paths=[self.path],
        )

    def load(self, session: Session) -> list[Message]:
        """Return a one-message transcript."""
        return [Message(Role.USER, text="hello")]

    def delete(self, session: Session, *, dry_run: bool = False) -> DeleteResult:
        """Report a note and a refusal that embed an escape, as a database row might."""
        result = super().delete(session, dry_run=dry_run)
        result.note = f"{OSC_ST}1 row"
        result.skipped[self._root / f"x{OSC_BEL}.db"] = f"locked {OSC_ST}(refused)"
        return result

    def plan_move(self, sessions: list[Session], old: Path, new: Path) -> MovePlan:
        """Plan to re-point the session's file."""
        return MovePlan(
            harness=self.name, old=old, new=new, sessions=list(sessions), rewrites=[self.path]
        )

    def move(self, plan: MovePlan, *, dry_run: bool = False) -> MoveResult:
        """Report a rewrite, a note and a refusal that embed escapes."""
        return MoveResult(
            dry_run=dry_run,
            rewritten=[self._root / f"re{OSC_BEL}.jsonl"],
            fields_updated=1,
            note=f"{OSC_ST}updated rows",
            skipped={self._root / f"y{OSC_BEL}": f"busy {OSC_ST}(refused)"},
        )

    def find_orphans(self) -> list[Orphan]:
        """One orphan whose reason quotes a planted project path."""
        return [
            Orphan(
                harness=self.name,
                kind=OrphanKind.DEAD_PROJECT,
                paths=[self.path],
                reason=f"project directory no longer exists: /gone{OSC_BEL}",
                size_bytes=3,
            )
        ]


def _wire(app: SxApp, adapter: HarnessAdapter, tmp_path: Path) -> Session:
    """Inject ``adapter`` in place of the real registry; return its session."""
    adapters = {adapter.name: adapter}
    app._adapters_by_name = adapters
    app._delete_service = DeleteService(adapters, log_path=tmp_path / "ops.log")
    app._move_service = MoveService(adapters, log_path=tmp_path / "ops.log")
    app._sessions_by_harness = {adapter.name: list(adapter.discover())}
    app._load_errors = []
    app._live_cache = {}
    app._populate_tree()
    return app._sessions_by_harness[adapter.name][0]


def _notices(app: SxApp) -> list[str]:
    """Record every notification the app shows from now on."""
    seen: list[str] = []
    original = app.notify

    def spy(message, **kwargs):
        """Record ``message``, then show it as the app would.

        Args:
            message: The notification text.
            **kwargs: The rest of :meth:`~textual.app.App.notify`'s arguments.

        Returns:
            Whatever the real ``notify`` returns.
        """
        seen.append(str(message))
        return original(message, **kwargs)

    app.notify = spy  # type: ignore[method-assign]
    return seen


async def _wait_for(app: SxApp, pilot, screen_type) -> None:
    """Pause until ``screen_type`` is the active screen."""
    for _ in range(20):
        await pilot.pause()
        if isinstance(app.screen, screen_type):
            return
    raise AssertionError(f"the screen is {type(app.screen).__name__}")


def _rendered(app: SxApp) -> str:
    """Return what the active screen sends to the terminal, escape codes included."""
    return app.screen._compositor.render_full_update().render_segments(app.console)


@pytest.mark.asyncio
async def test_the_delete_preview_cannot_drive_the_terminal(tmp_path, monkeypatch):
    """Title, project, note and refusal reach the dialog neutralized (T10)."""
    adapter = _Planted(tmp_path / "store", project=f"/proj{OSC_BEL}")
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)
        monkeypatch.setattr(app, "_current_session", lambda: session)

        await pilot.press("d")
        await _wait_for(app, pilot, ConfirmDeleteScreen)
        preview = app.screen._preview.plain
        assert _clean(preview), repr(preview)
        assert "␛]0;PWNED␇hello title" in preview, "shown, as visible glyphs"
        assert "\x1b]0;PWNED" not in _rendered(app)
        app.screen.dismiss(None)
        await pilot.pause()


@pytest.mark.asyncio
async def test_tree_project_groups_cannot_drive_the_terminal(tmp_path):
    """Grouped by project, the group label is a path read from a session file (T10)."""
    adapter = _Planted(tmp_path / "store", project=f"/proj{OSC_ST}")
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        _wire(app, adapter, tmp_path)
        [harness] = [n for n in app.query_one("#tree", Tree).root.children if n.children]
        [group] = harness.children
        assert _clean(str(group.label)), repr(str(group.label))
        assert "/proj␛]0;PWNED␛\\" in str(group.label)


@pytest.mark.asyncio
async def test_the_move_dialog_shows_the_old_path_neutralized(tmp_path, monkeypatch):
    """The "from" line is sanitized, and the path is not pre-filled altered (T10)."""
    adapter = _Planted(tmp_path / "store", project=f"/proj{OSC_BEL}")
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)
        monkeypatch.setattr(app, "_current_session", lambda: session)

        app.action_move_sessions()
        await _wait_for(app, pilot, MoveScreen)
        source = str(app.screen.query_one("#from", Label).content)
        assert _clean(source) and source.startswith("from  /proj␛]0;PWNED␇")
        assert app.screen.query_one("#destination", Input).value == ""
        assert "\x1b]0;PWNED" not in _rendered(app)
        app.screen.dismiss(None)
        await pilot.pause()


@pytest.mark.asyncio
async def test_a_clean_old_path_is_still_pre_filled(tmp_path, monkeypatch):
    """Sanitizing changes nothing for an ordinary path, so it is offered for editing."""
    adapter = _Planted(tmp_path / "store")
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)
        monkeypatch.setattr(app, "_current_session", lambda: session)

        app.action_move_sessions()
        await _wait_for(app, pilot, MoveScreen)
        assert app.screen.query_one("#destination", Input).value == "/proj"
        app.screen.dismiss(None)
        await pilot.pause()


@pytest.mark.asyncio
async def test_the_move_plan_cannot_drive_the_terminal(tmp_path, monkeypatch):
    """Rewritten paths, notes and refusals in the plan are neutralized (T10)."""
    adapter = _Planted(tmp_path / "store")
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)
        monkeypatch.setattr(app, "_current_session", lambda: session)

        app.action_move_sessions()
        await _wait_for(app, pilot, MoveScreen)
        app.screen.query_one("#destination", Input).value = str(tmp_path / "new")
        app.screen._run_preview()
        await pilot.pause()
        plan = app.screen.query_one("#preview-body", Static).content
        assert isinstance(plan, Text)
        assert _clean(plan.plain), repr(plan.plain)
        assert "re␛]0;PWNED␇.jsonl" in plan.plain and "busy ␛]0;PWNED␛\\(refused)" in plan.plain
        app.screen.dismiss(None)
        await pilot.pause()


@pytest.mark.asyncio
async def test_the_orphan_confirmation_cannot_drive_the_terminal(tmp_path):
    """An orphan's reason quotes a path read from a session file (T10)."""
    adapter = _Planted(tmp_path / "store")
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        _wire(app, adapter, tmp_path)
        orphans = app._collect_orphans()
        confirm = app.run_worker(app._confirm_orphan(orphans))
        await _wait_for(app, pilot, ConfirmDeleteScreen)
        assert _clean(app.screen._preview.plain), repr(app.screen._preview.plain)
        app.screen.dismiss(None)
        await confirm.wait()


@pytest.mark.asyncio
async def test_the_memory_screen_cannot_drive_the_terminal(tmp_path):
    """A memory's name, type and description are model-written (T10)."""
    document = tmp_path / "a.md"
    document.write_text("body\n", encoding="utf-8")
    memory = MemoryFile(
        path=document,
        project_path=f"/proj{OSC_BEL}",
        name=f"{OSC_ST}name",
        description=f"{OSC_BEL}described",
        kind=f"user{OSC_BEL}",
        size_bytes=5,
    )
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        screen = MemoryScreen(
            memories=[memory], on_export=None, on_delete=None, confirm=None
        )
        app.push_screen(screen)
        await _wait_for(app, pilot, MemoryScreen)
        table = screen.query_one("#memory-table", DataTable)
        cells = [str(cell) for cell in table.get_row_at(0)]
        assert all(_clean(cell) for cell in cells), cells
        screen.action_preview()
        await pilot.pause()
        body = screen.query_one("#memory-body", Static).content
        assert _clean(body.plain), repr(body.plain)
        assert "\x1b]0;PWNED" not in _rendered(app)


@pytest.mark.asyncio
async def test_delete_toasts_cannot_drive_the_terminal(tmp_path, monkeypatch):
    """The outcome toast embeds the adapter's note, sanitized and escaped (T10)."""
    adapter = _Planted(tmp_path / "store")
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)
        monkeypatch.setattr(app, "_current_session", lambda: session)
        notices = _notices(app)

        await pilot.press("d")
        await _wait_for(app, pilot, ConfirmDeleteScreen)
        await pilot.click("#confirm")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert notices and all(_clean(n) for n in notices), notices
        # The session's file goes but the planted database file is refused, so
        # the toast is a partial delete's, which names what stayed.
        assert any(
            n.startswith("Partly deleted — 1 path(s) stayed on disk: x␛]0;PWNED␇.db")
            and "␛]0;PWNED␛\\1 row" in n
            for n in notices
        ), notices


@pytest.mark.asyncio
async def test_the_export_toast_is_not_read_as_markup(tmp_path, monkeypatch):
    """A directory named like a style tag is shown as it is, not applied (T8)."""
    work = tmp_path / "[red]work"
    work.mkdir()
    monkeypatch.chdir(work)
    adapter = _Planted(tmp_path / "store")
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)
        monkeypatch.setattr(app, "_current_session", lambda: session)
        notices = _notices(app)

        await pilot.press("e")
        await app.workers.wait_for_complete()
        await pilot.pause()

        [export] = sorted((work / "session-exports").glob("*.md"))
        assert notices == [f"Exported → {SxApp._toast_safe(export)}"]
        assert r"\[red]work" in notices[0]


@pytest.mark.asyncio
async def test_a_slash_in_the_session_id_does_not_abort_the_export(tmp_path, monkeypatch):
    """The id is slugged for the filename, so the archive lands where it should (T8)."""
    adapter = _Planted(tmp_path / "store", session_id="ab/../cd-x")
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)
        monkeypatch.setattr(app, "_current_session", lambda: session)
        notices = _notices(app)

        await pilot.press("e")
        await app.workers.wait_for_complete()
        await pilot.pause()

        [export] = sorted((tmp_path / "session-exports").glob("*.md"))
        assert "-ab-cd-x-" in export.name
        assert notices[0].startswith("Exported → ")
