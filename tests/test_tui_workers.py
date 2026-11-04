"""TUI regressions for the background work behind opening, exporting and deleting.

Opening a session parses it on a thread. Two things went wrong with that:

* the load ran as an exclusive worker in the default group, so opening a
  session canceled an ``e`` or ``d`` still reading its own session (T6);
* a thread worker keeps running after it is canceled, and a superseded load
  still filled the pane and the open-session state, so ``t`` then cycled the
  wrong session (T9).

A fake adapter holds its parse of one session at a gate the test opens, which
makes each interleaving deterministic. Its store lives under ``tmp_path``; the
registry scan on mount is patched out, and the working directory is moved to
``tmp_path`` so exports never land in the repository.
"""

from __future__ import annotations

import asyncio
import os
import threading
from datetime import datetime
from pathlib import Path

import pytest
from textual.widgets import RichLog

from sx.adapters.base import HarnessAdapter
from sx.model import Branch, Capability, Message, Role, Session, Transcript
from sx.service import DeleteService
from sx.tui.app import SxApp
from sx.tui.screens import BranchModeScreen, ConfirmDeleteScreen

#: Seconds a gated parse waits before giving up, so a failing test cannot hang.
GATE_TIMEOUT = 10


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Keep the app off the real stores and its exports out of the repository."""
    monkeypatch.setattr("sx.tui.app.build_registry", lambda: ([], []))
    monkeypatch.setattr("sx.memory.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)


def _transcript(word: str) -> Transcript:
    """A session that branched, every message naming ``word``."""
    ask = Message(Role.USER, text=f"{word} prompt")
    kept = Message(Role.ASSISTANT, text=f"{word} kept answer")
    abandoned = Message(Role.ASSISTANT, text=f"{word} abandoned answer")
    return Transcript(
        current=[ask, kept],
        branches=[Branch(label="retried", messages=[abandoned])],
        tree=[ask, abandoned, Message(Role.OTHER, text="Branch 2 of 2", marker=True), kept],
        hidden=1,
    )


class _Gated(HarnessAdapter):
    """A Pi-like harness whose parse of the session ``slow`` waits at a gate.

    Attributes:
        entered: Set once a parse of ``slow`` is waiting at the gate.
        gate: Set by the test to let that parse finish.
        finished: Set once a parse of ``slow`` has returned.
    """

    name = "gated"
    display = "Gated"
    capabilities = Capability.BROWSE | Capability.DELETE
    supports_branches = True

    def __init__(self, root: Path) -> None:
        """Write a back-dated file for ``slow`` and for ``fast``."""
        self._root = root
        self.entered = threading.Event()
        self.gate = threading.Event()
        self.finished = threading.Event()
        root.mkdir(parents=True, exist_ok=True)
        for session_id in ("slow", "fast"):
            path = root / f"{session_id}.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            os.utime(path, (1_700_000_000, 1_700_000_000))

    def store_roots(self) -> list[Path]:
        """The synthetic store."""
        return [self._root]

    def discover(self):
        """Yield ``slow`` and ``fast``."""
        for session_id in ("slow", "fast"):
            yield Session(
                harness=self.name,
                session_id=session_id,
                project_path="/proj",
                title=session_id,
                modified=datetime(2026, 5, 30, 12, 0),
                paths=[self._root / f"{session_id}.jsonl"],
            )

    def load(self, session: Session) -> list[Message]:
        """Return the session's current branch."""
        return list(self.load_transcript(session).current)

    def load_transcript(self, session: Session) -> Transcript:
        """Return the session's transcript; ``slow`` first waits for the gate."""
        if session.session_id == "slow":
            self.entered.set()
            try:
                self.gate.wait(GATE_TIMEOUT)
            finally:
                self.finished.set()
        return _transcript(session.session_id.upper())


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


def _pane(app: SxApp) -> str:
    """Return the transcript pane's text with all whitespace collapsed."""
    lines = app.query_one("#transcript", RichLog).lines
    return " ".join(" ".join(line.text for line in lines).split())


async def _until(pilot, condition, tries: int = 100) -> None:
    """Pause until ``condition()`` holds, failing the test if it never does."""
    for _ in range(tries):
        if condition():
            return
        await pilot.pause(0.02)
    raise AssertionError("condition never held")


async def _gated(pilot, adapter: _Gated) -> None:
    """Wait until a parse of ``slow`` is held at the gate."""
    await _until(pilot, adapter.entered.is_set)


@pytest.mark.asyncio
async def test_a_superseded_load_never_reaches_the_pane(tmp_path):
    """The session opened last owns the pane and ``t``, whichever parse ends last (T9)."""
    adapter = _Gated(tmp_path / "store")
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        sessions = _wire(app, adapter, tmp_path)
        try:
            app.open_session(sessions["slow"])
            await _gated(pilot, adapter)
            await app.open_session(sessions["fast"]).wait()
            await pilot.pause()
            assert "FAST prompt" in _pane(app)
        finally:
            adapter.gate.set()
        await _until(pilot, adapter.finished.is_set)
        for _ in range(10):  # room for a late result to land, if it were going to
            await pilot.pause(0.02)

        assert "FAST prompt" in _pane(app) and "SLOW" not in _pane(app)
        assert app._open_session is sessions["fast"]
        assert app._open_transcript is not None
        assert app._open_transcript.current[0].text == "FAST prompt"


@pytest.mark.asyncio
async def test_opening_a_session_does_not_cancel_an_export(tmp_path, monkeypatch):
    """``e`` still asks, and exports, after another session is opened meanwhile (T6)."""
    adapter = _Gated(tmp_path / "store")
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        sessions = _wire(app, adapter, tmp_path)
        monkeypatch.setattr(app, "_current_session", lambda: sessions["slow"])
        try:
            app.action_export_session()
            await _gated(pilot, adapter)
            await app.open_session(sessions["fast"]).wait()
        finally:
            adapter.gate.set()
        await _until(pilot, lambda: isinstance(app.screen, BranchModeScreen))

        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        [export] = sorted((tmp_path / "session-exports").glob("*.md"))
        assert "SLOW kept answer" in export.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_opening_a_session_does_not_cancel_a_delete(tmp_path, monkeypatch):
    """``d`` still reaches its dialog after another session is opened meanwhile (T6)."""
    adapter = _Gated(tmp_path / "store")
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        sessions = _wire(app, adapter, tmp_path)
        monkeypatch.setattr(app, "_current_session", lambda: sessions["slow"])
        try:
            app.action_delete_session()
            await _gated(pilot, adapter)
            await app.open_session(sessions["fast"]).wait()
        finally:
            adapter.gate.set()
        await _until(pilot, lambda: isinstance(app.screen, ConfirmDeleteScreen))

        await pilot.click("#confirm")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert not sessions["slow"].primary_path.exists()
        assert sessions["fast"].primary_path.exists()


@pytest.mark.asyncio
async def test_a_canceled_export_says_so(tmp_path, monkeypatch):
    """An export that is canceled anyway tells the user it did not finish (T6)."""
    adapter = _Gated(tmp_path / "store")
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        sessions = _wire(app, adapter, tmp_path)
        monkeypatch.setattr(app, "_current_session", lambda: sessions["slow"])
        notices = _notices(app)
        try:
            worker = app.action_export_session()
            await _gated(pilot, adapter)
            worker.cancel()
            await _until(pilot, lambda: bool(notices))
        finally:
            adapter.gate.set()
        await _until(pilot, adapter.finished.is_set)
        await asyncio.sleep(0.05)
        await pilot.pause()

        assert notices == ["Export canceled before it finished."]
        assert not (tmp_path / "session-exports").exists()


@pytest.mark.asyncio
async def test_a_canceled_delete_says_nothing_was_deleted(tmp_path, monkeypatch):
    """A delete canceled before its dialog says so, and deletes nothing (T6)."""
    adapter = _Gated(tmp_path / "store")
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        sessions = _wire(app, adapter, tmp_path)
        monkeypatch.setattr(app, "_current_session", lambda: sessions["slow"])
        notices = _notices(app)
        try:
            worker = app.action_delete_session()
            await _gated(pilot, adapter)
            worker.cancel()
            await _until(pilot, lambda: bool(notices))
        finally:
            adapter.gate.set()
        await _until(pilot, adapter.finished.is_set)
        await pilot.pause()

        assert notices == ["Delete canceled — nothing was deleted."]
        assert not isinstance(app.screen, ConfirmDeleteScreen)
        assert sessions["slow"].primary_path.exists()
