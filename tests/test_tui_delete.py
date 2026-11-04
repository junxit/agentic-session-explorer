"""TUI regressions for deleting a session: liveness, project state, caveats.

* D10 — liveness was checked only when the dialog opened, so a session that
  became active while it was open was deleted with a plain confirm.
* D11 — the "last session for this project" offer said only "Pi — 1 path(s)",
  and called a Pi session folder "memory and settings".
* The project state ticked for removal is checked again before it goes: a
  session started in the project since the tree was scanned lives in the same
  folder.
* An adapter's open-session caveat is shown in the dialog.
* The dialog fits the screen, and when it cannot, opens on its heading: at
  80x24 it opened scrolled to the bottom, the live warning out of view.
* A delete that refused part of a session warns, naming what stayed, instead
  of toasting success.
* A session deleted elsewhere while the dialog is open is reported gone, not
  as one that became active.

A fake file-per-session harness keeps its transcripts in one project folder
under ``tmp_path``, the way Pi does; the registry scan on mount is patched out
and the working directory is moved to ``tmp_path``.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest
from rich.text import Text
from textual.widgets import Checkbox, Input, Label

from sx.adapters.base import HarnessAdapter
from sx.model import (
    Branch,
    Capability,
    DeleteResult,
    Message,
    ProjectLeftovers,
    Role,
    Session,
    Transcript,
)
from sx.service import DeleteService
from sx.tui.app import SxApp
from sx.tui.screens import ConfirmDeleteScreen
from sx.util import dir_size

PROJECT = "/proj"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Keep the app off the real stores and its exports out of the repository."""
    monkeypatch.setattr("sx.tui.app.build_registry", lambda: ([], []))
    monkeypatch.setattr("sx.memory.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)


class _Folder(HarnessAdapter):
    """A harness keeping each project's transcripts in one folder, as Pi does.

    Attributes:
        folder: The project's folder, which ``project_leftovers`` offers.
        memory: Memory documents reported with the folder.
        notes: Settings edits reported with the folder.
        cleaned: Every ``delete_project_leftovers`` call that was not a dry run.
    """

    name = "fake"
    display = "Fake"
    capabilities = Capability.BROWSE | Capability.DELETE

    def __init__(self, root: Path, *session_ids: str) -> None:
        """Write one back-dated transcript per id into the project's folder."""
        self._root = root
        self.folder = root / "proj-folder"
        self.memory: list[Path] = []
        self.notes: list[str] = []
        self.cleaned: list[ProjectLeftovers] = []
        for session_id in session_ids:
            self.add(session_id)

    def add(self, session_id: str) -> Path:
        """Write a back-dated transcript for ``session_id``, as the harness would.

        Returns:
            The transcript's path.
        """
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / f"{session_id}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        os.utime(path, (1_700_000_000, 1_700_000_000))
        return path

    def store_roots(self) -> list[Path]:
        """The synthetic store."""
        return [self._root]

    def discover(self):
        """Yield one session per transcript in the folder."""
        for path in sorted(self.folder.glob("*.jsonl")):
            yield Session(
                harness=self.name,
                session_id=path.stem,
                project_path=PROJECT,
                title=path.stem,
                modified=datetime(2026, 5, 30, 12, 0),
                paths=[path],
            )

    def load(self, session: Session) -> list[Message]:
        """Return a one-message transcript."""
        return [Message(Role.USER, text="hello")]

    def project_leftovers(self, project: str) -> ProjectLeftovers | None:
        """Offer the folder, with any memory and settings configured."""
        if not self.folder.is_dir():
            return None
        return ProjectLeftovers(
            project_path=project,
            memory_files=list(self.memory),
            paths=[self.folder],
            config_notes=list(self.notes),
            size_bytes=dir_size(self.folder),
        )

    def delete_project_leftovers(
        self, leftovers: ProjectLeftovers, *, dry_run: bool = False
    ) -> DeleteResult:
        """Record the call, then remove the folder through the store guard."""
        if not dry_run:
            self.cleaned.append(leftovers)
        return self._delete_paths(leftovers.paths, dry_run=dry_run)


class _Careful(_Folder):
    """A harness that warns about deleting a session it has open, as Pi does."""

    open_session_caveat = "close the session in Fake first"


def _wire(app: SxApp, adapter: HarnessAdapter, tmp_path: Path) -> dict[str, Session]:
    """Inject ``adapter`` in place of the real registry; return its sessions by id."""
    app._adapters_by_name = {adapter.name: adapter}
    app._delete_service = DeleteService(app._adapters_by_name, log_path=tmp_path / "ops.log")
    app._sessions_by_harness = {adapter.name: list(adapter.discover())}
    app._load_errors = []
    app._live_cache = {}
    app._populate_tree()
    return {s.session_id: s for s in app._sessions_by_harness[adapter.name]}


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


async def _open_delete_dialog(app: SxApp, pilot) -> ConfirmDeleteScreen:
    """Press ``d`` and wait for the delete dialog.

    Returns:
        The dialog.
    """
    await pilot.press("d")
    for _ in range(10):
        await pilot.pause()
        if isinstance(app.screen, ConfirmDeleteScreen):
            break
    assert isinstance(app.screen, ConfirmDeleteScreen)
    return app.screen


async def _settle(app: SxApp, pilot) -> None:
    """Let every running worker finish."""
    await app.workers.wait_for_complete()
    await pilot.pause()


# --- D10: liveness at the moment of the delete --------------------------------


@pytest.mark.asyncio
async def test_a_session_that_turns_live_in_the_dialog_is_not_deleted(tmp_path, monkeypatch):
    """Confirmed without the live-session phrase, a now-live session is refused (D10)."""
    adapter = _Folder(tmp_path / "store", "one")
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["one"]
        monkeypatch.setattr(app, "_current_session", lambda: session)
        notices = _notices(app)

        screen = await _open_delete_dialog(app, pilot)
        assert not screen.query("#confirm-input"), "not live, so no phrase is asked"
        os.utime(session.primary_path, None)  # the harness writes to it now
        await pilot.click("#confirm")
        await _settle(app, pilot)

        assert session.primary_path.exists()
        assert any("became active while the dialog was open" in n for n in notices), notices


@pytest.mark.asyncio
async def test_a_live_session_confirmed_with_the_phrase_is_deleted(tmp_path, monkeypatch):
    """The re-check only refuses what the phrase never covered."""
    adapter = _Folder(tmp_path / "store", "one")
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["one"]
        monkeypatch.setattr(app, "_current_session", lambda: session)
        os.utime(session.primary_path, None)

        screen = await _open_delete_dialog(app, pilot)
        screen.query_one("#confirm-input", Input).value = "DELETE"
        await pilot.pause()
        await pilot.click("#confirm")
        await _settle(app, pilot)

        assert not session.primary_path.exists()


# --- D11: what the last-session offer would remove ------------------------------


@pytest.mark.asyncio
async def test_the_last_session_preview_lists_each_path_it_would_remove(tmp_path, monkeypatch):
    """Each leftover path is named with what it holds, and labeled as what it is (D11)."""
    adapter = _Folder(tmp_path / "store", "one")
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["one"]
        monkeypatch.setattr(app, "_current_session", lambda: session)

        screen = await _open_delete_dialog(app, pilot)
        preview = screen._preview.plain
        assert f"folder {adapter.folder}" in preview
        assert "contains 1 file(s), 1 transcript(s) — this session's files among them" in preview
        assert "this is the LAST session for this project" in preview
        label = str(screen.query_one("#bundle", Checkbox).label)
        assert "memory and settings" not in label
        assert label.startswith("Also remove the Fake project state listed above (1 path(s), ")


@pytest.mark.asyncio
async def test_the_last_session_preview_names_memory_and_settings(tmp_path, monkeypatch):
    """Memory documents and settings edits are listed, and counted in the label."""
    adapter = _Folder(tmp_path / "store", "one")
    memory = adapter.folder / "memory"
    memory.mkdir()
    for name in ("a-fact.md", "MEMORY.md"):
        (memory / name).write_text("remember\n", encoding="utf-8")
        adapter.memory.append(memory / name)
    adapter.notes.append("Fake settings entry (trust)")
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["one"]
        monkeypatch.setattr(app, "_current_session", lambda: session)

        screen = await _open_delete_dialog(app, pilot)
        preview = screen._preview.plain
        assert "contains 3 file(s), 1 transcript(s), 2 memory file(s)" in preview
        assert "• 2 memory file(s): a-fact.md, MEMORY.md" in preview
        assert "• Fake settings entry (trust)" in preview
        label = str(screen.query_one("#bundle", Checkbox).label)
        assert "2 memory file(s), 1 settings or history entry(s)" in label


@pytest.mark.asyncio
async def test_project_state_stays_when_a_session_appeared_since_the_scan(
    tmp_path, monkeypatch
):
    """The offer came from the tree's scan; the stores are counted again before it goes."""
    adapter = _Folder(tmp_path / "store", "one")
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["one"]
        monkeypatch.setattr(app, "_current_session", lambda: session)
        notices = _notices(app)
        newer = adapter.add("two")  # started after the tree was scanned

        screen = await _open_delete_dialog(app, pilot)
        screen.query_one("#bundle", Checkbox).value = True
        await pilot.pause()
        await pilot.click("#confirm")
        await _settle(app, pilot)

        assert not session.primary_path.exists()
        assert newer.exists(), "the newer session must survive"
        assert adapter.cleaned == []
        assert any(n.startswith("Project state kept — 1 other session(s)") for n in notices)


@pytest.mark.asyncio
async def test_project_state_goes_when_it_is_still_left_over(tmp_path, monkeypatch):
    """With nothing new in the project, the ticked state is removed as before."""
    adapter = _Folder(tmp_path / "store", "one")
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["one"]
        monkeypatch.setattr(app, "_current_session", lambda: session)

        screen = await _open_delete_dialog(app, pilot)
        screen.query_one("#bundle", Checkbox).value = True
        await pilot.pause()
        await pilot.click("#confirm")
        await _settle(app, pilot)

        assert len(adapter.cleaned) == 1
        assert not adapter.folder.exists()


# --- the open-session caveat ----------------------------------------------------


@pytest.mark.asyncio
async def test_the_delete_dialog_shows_the_open_session_caveat(tmp_path, monkeypatch):
    """An adapter's warning about open sessions is in the confirmation."""
    adapter = _Careful(tmp_path / "store", "one")
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["one"]
        monkeypatch.setattr(app, "_current_session", lambda: session)

        screen = await _open_delete_dialog(app, pilot)
        caveat = screen.query_one("#caveat", Label)
        assert str(caveat.content) == "⚠ Close the session in Fake first"


@pytest.mark.asyncio
async def test_no_caveat_without_one_from_the_adapter(tmp_path, monkeypatch):
    """Adapters that set nothing — or predate the attribute — show no caveat."""
    adapter = _Folder(tmp_path / "store", "one")
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["one"]
        monkeypatch.setattr(app, "_current_session", lambda: session)

        screen = await _open_delete_dialog(app, pilot)
        assert not screen.query("#caveat")


# --- the dialog fits the screen -------------------------------------------------


class _Branchy(_Careful):
    """A Pi-like harness: branched sessions, project state, and a caveat."""

    supports_branches = True

    def load_transcript(self, session: Session) -> Transcript:
        """A prompt answered twice, the first answer abandoned."""
        ask = Message(Role.USER, text="the prompt")
        abandoned = Message(Role.ASSISTANT, text="abandoned answer")
        kept = Message(Role.ASSISTANT, text="kept answer")
        return Transcript(
            current=[ask, kept],
            branches=[Branch(label="retried", messages=[abandoned])],
            tree=[ask, abandoned, Message(Role.OTHER, text="Branch 2 of 2", marker=True), kept],
            hidden=1,
        )


def _inside(outer, inner) -> bool:
    """Return True if region ``inner`` lies wholly within region ``outer``."""
    return (
        outer.x <= inner.x
        and outer.y <= inner.y
        and inner.right <= outer.right
        and inner.bottom <= outer.bottom
    )


@pytest.mark.asyncio
async def test_the_fullest_delete_dialog_keeps_its_controls_on_screen(tmp_path, monkeypatch):
    """The preview gives up rows so the choice, the phrase and the buttons stay in view.

    With a live, branched last session the dialog outgrew the screen, and the
    Delete button was cut off below its border.
    """
    adapter = _Branchy(tmp_path / "store", "one")
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["one"]
        monkeypatch.setattr(app, "_current_session", lambda: session)
        os.utime(session.primary_path, None)  # live: the typed phrase is asked for

        screen = await _open_delete_dialog(app, pilot)
        for _ in range(3):
            await pilot.pause()
        dialog = screen.query_one("#dialog")
        assert dialog.max_scroll_y == 0, "nothing should need scrolling at this size"
        for selector in ("#caveat", "#export", "#export-mode", "#bundle", "#confirm-input",
                         "#confirm"):
            assert _inside(dialog.content_region, screen.query_one(selector).region), selector
        shrunk = screen.query_one("#preview").outer_size.height

        await pilot.resize_terminal(160, 70)
        for _ in range(3):
            await pilot.pause()
        assert screen.query_one("#preview").outer_size.height > shrunk, "room again, so it grows"
        screen.dismiss(None)
        await _settle(app, pilot)


@pytest.mark.asyncio
async def test_the_fullest_delete_dialog_opens_on_its_heading_at_80x24(tmp_path, monkeypatch):
    """Too tall to fit, the dialog opens at its top, focused on the typed phrase.

    It opened scrolled to the bottom instead: the heading, the live warning and
    the preview were all out of view. The focused field still takes the
    phrase, and Tab brings the buttons into view.
    """
    adapter = _Branchy(tmp_path / "store", "one")
    app = SxApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["one"]
        monkeypatch.setattr(app, "_current_session", lambda: session)
        os.utime(session.primary_path, None)  # live: the typed phrase is asked for

        screen = await _open_delete_dialog(app, pilot)
        for _ in range(10):  # long enough for an animated scroll to have landed
            await pilot.pause(0.05)
        dialog = screen.query_one("#dialog")
        assert dialog.max_scroll_y > 0, "the fullest dialog cannot fit at 80x24"
        assert dialog.scroll_y == 0
        for selector in ("#heading", "#live-warning", "#caveat"):
            assert _inside(dialog.content_region, screen.query_one(selector).region), selector
        preview = screen.query_one("#preview").region
        assert dialog.content_region.y <= preview.y < dialog.content_region.bottom
        assert app.focused is screen.query_one("#confirm-input", Input)

        await pilot.press(*"DELETE", "tab", "tab")
        confirm = screen.query_one("#confirm")
        assert app.focused is confirm and not confirm.disabled
        assert _inside(dialog.content_region, confirm.region), "focus scrolls it into view"
        screen.dismiss(None)
        await _settle(app, pilot)


#: As long as the caveat Pi's adapter gives.
LONG_CAVEAT = (
    "Fake leaves no sign of which sessions it has open, so close a session in Fake "
    "before deleting or moving it: otherwise its next turn fails if the session's "
    "folder is gone, or is written to a header-less file it never lists."
)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(160, 45), (80, 24)])
async def test_the_warnings_wrap_inside_the_dialog(size):
    """A label as wide as its text ran past the dialog's edge, and was cut off there."""
    app = SxApp()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        screen = ConfirmDeleteScreen(
            heading="Delete?", preview=Text("preview"), is_live=True, caveat=LONG_CAVEAT
        )
        app.push_screen(screen)
        for _ in range(3):
            await pilot.pause()
        dialog = screen.query_one("#dialog")
        for selector in ("#live-warning", "#caveat"):
            region = screen.query_one(selector).region
            assert region.right <= dialog.content_region.right, (selector, region)
        assert screen.query_one("#caveat").region.height > 1, "the caveat wraps"
        screen.dismiss(None)
        await pilot.pause()


# --- the outcome toast ----------------------------------------------------------


def _toasts(app: SxApp) -> list[tuple[str, str]]:
    """Record every notification from now on, with its severity."""
    seen: list[tuple[str, str]] = []
    original = app.notify

    def spy(message, **kwargs):
        """Record ``message`` and its severity, then show it as the app would.

        Args:
            message: The notification text.
            **kwargs: The rest of :meth:`~textual.app.App.notify`'s arguments.

        Returns:
            Whatever the real ``notify`` returns.
        """
        seen.append((str(message), kwargs.get("severity", "information")))
        return original(message, **kwargs)

    app.notify = spy  # type: ignore[method-assign]
    return seen


class _StuckBackup(_Folder):
    """Removes a session's backup but keeps its transcript, as Oh My Pi's delete can.

    omp deletes a session's backups first and stops at the first it cannot
    remove, keeping the transcript: a backup outliving it would bring the
    session back. Here the second backup is the one that cannot go.
    """

    def add(self, session_id: str) -> Path:
        """Write the transcript and two backups next to it."""
        path = super().add(session_id)
        for n in (1, 2):
            (self.folder / f"{session_id}.jsonl.{n}.bak").write_text("{}\n", encoding="utf-8")
        return path

    def delete(self, session: Session, *, dry_run: bool = False) -> DeleteResult:
        """Remove the first backup; refuse the second, and so the transcript."""
        path = session.primary_path
        first, second = (path.with_name(f"{path.name}.{n}.bak") for n in (1, 2))
        result = self._delete_paths([first], dry_run=dry_run)
        result.skipped[second] = "Permission denied (refused)"
        result.skipped[path] = (
            f"not deleted: {second.name} could not be removed, and Oh My Pi would "
            "restore the session from it if the rest went (refused)"
        )
        return result


@pytest.mark.asyncio
async def test_a_partly_refused_delete_warns_what_stayed(tmp_path, monkeypatch):
    """Something removed is not success when the transcript stayed: no "Deleted:" toast.

    The ticked project state stays too: the project still has the session, and
    removing its folder would take the transcript the delete just kept.
    """
    adapter = _StuckBackup(tmp_path / "store", "one")
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["one"]
        monkeypatch.setattr(app, "_current_session", lambda: session)
        toasts = _toasts(app)

        screen = await _open_delete_dialog(app, pilot)
        screen.query_one("#bundle", Checkbox).value = True
        await pilot.pause()
        await pilot.click("#confirm")
        await _settle(app, pilot)

        assert session.primary_path.exists()
        assert not (adapter.folder / "one.jsonl.1.bak").exists()
        [(outcome, severity), *rest] = toasts
        assert severity == "warning", toasts
        assert outcome.startswith("Session NOT deleted — 2 path(s) stayed on disk"), outcome
        assert "one.jsonl.2.bak, one.jsonl (Permission denied (refused))" in outcome
        assert "Removed only 1 path(s)" in outcome
        assert not any(text.startswith("Deleted:") for text, _ in toasts), toasts
        assert adapter.cleaned == [] and adapter.folder.is_dir()
        assert ("Project state kept — the session itself was not deleted.", "warning") in rest


class _StuckSidecar(_Folder):
    """Removes a session's transcript but not its sidecar."""

    def delete(self, session: Session, *, dry_run: bool = False) -> DeleteResult:
        """Remove the transcript; refuse a sidecar next to it."""
        result = self._delete_paths(list(session.paths), dry_run=dry_run)
        result.skipped[self.folder / "one.meta"] = "outside store root (refused)"
        return result


@pytest.mark.asyncio
async def test_a_delete_that_kept_only_a_sidecar_warns_it_was_partial(tmp_path, monkeypatch):
    """The transcript went, so the toast says partly deleted, and names what stayed."""
    adapter = _StuckSidecar(tmp_path / "store", "one")
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["one"]
        monkeypatch.setattr(app, "_current_session", lambda: session)
        toasts = _toasts(app)

        await _open_delete_dialog(app, pilot)
        await pilot.click("#confirm")
        await _settle(app, pilot)

        assert not session.primary_path.exists()
        [(outcome, severity)] = toasts
        assert severity == "warning"
        assert outcome.startswith(
            "Partly deleted — 1 path(s) stayed on disk: one.meta (outside store root (refused)). "
            "Removed 1 path(s) · "
        ), outcome


# --- a session deleted elsewhere --------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("live", [False, True], ids=["idle", "live"])
@pytest.mark.parametrize("export", [False, True], ids=["delete", "export-first"])
async def test_a_session_deleted_while_the_dialog_is_open_is_reported_gone(
    tmp_path, monkeypatch, live, export
):
    """The toast says the session no longer exists and nothing was deleted.

    It said the session had become active, and to press d again: its file could
    not be read, which the liveness check takes for activity. Confirmed with
    the live phrase, the delete toasted "Deleted: nothing to remove"; with
    the export ticked, an archive of the missing session was written first.
    """
    adapter = _Folder(tmp_path / "store", "one")
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["one"]
        monkeypatch.setattr(app, "_current_session", lambda: session)
        if live:
            os.utime(session.primary_path, None)
        toasts = _toasts(app)

        screen = await _open_delete_dialog(app, pilot)
        session.primary_path.unlink()  # deleted elsewhere, by the harness or by hand
        if live:
            screen.query_one("#confirm-input", Input).value = "DELETE"
        screen.query_one("#export", Checkbox).value = export
        await pilot.pause()
        await pilot.click("#confirm")
        await _settle(app, pilot)

        gone = (
            "Nothing was deleted — the session no longer exists: "
            f"{session.primary_path} is gone. Press r to refresh the list."
        )
        assert toasts == [(gone, "warning")]
        assert not (tmp_path / "session-exports").exists()
        assert adapter.folder.is_dir(), "nothing else of the project was touched"


@pytest.mark.asyncio
async def test_a_session_already_gone_opens_no_dialog(tmp_path, monkeypatch):
    """Pressing d on a session deleted since the tree was built says so at once."""
    adapter = _Folder(tmp_path / "store", "one")
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["one"]
        monkeypatch.setattr(app, "_current_session", lambda: session)
        notices = _notices(app)
        session.primary_path.unlink()

        await pilot.press("d")
        for _ in range(10):
            await pilot.pause()
        opened = isinstance(app.screen, ConfirmDeleteScreen)
        if opened:
            app.screen.dismiss(None)  # so the delete's worker can finish
        await _settle(app, pilot)

        assert not opened, "a dialog about a session that is not there"
        assert len(notices) == 1 and "the session no longer exists" in notices[0]
