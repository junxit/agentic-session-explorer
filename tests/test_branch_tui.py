"""TUI tests for viewing and exporting sessions that branched.

Covers ``t`` (cycle the view without re-parsing), ``e`` (ask which view to
export, only when the session branched, from a fresh read), the delete
dialog's matching choice for its export step (the whole tree unless told
otherwise, and the highlighted row is the choice), dialogs that ignore a
second dismissal, an export refused for a session whose file is gone, and
what the tree says about each harness's store.

Fake adapters serve hand-built transcripts from a store under ``tmp_path``; the
registry scan on mount is patched out, and the working directory is moved to
``tmp_path`` so exports never land in the repository.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest
from rich.text import Text
from textual import events
from textual.widgets import Button, Checkbox, OptionList, RichLog, Tree

from sx.adapters.base import HarnessAdapter
from sx.export import messages_to_markdown
from sx.model import Branch, BranchMode, Capability, Message, Role, Session, Transcript
from sx.service import DeleteService
from sx.tui.app import SxApp
from sx.tui.screens import BranchModeScreen, ConfirmDeleteScreen, MoveScreen

CURRENT_HEADER = "⑂ current branch only — 1 message(s) on 1 other branch(es) not shown"
APPENDIX_HEADER = "⑂ current branch, then 1 other branch(es)"
TREE_HEADER = "⑂ whole tree in the order it was written — 2 branches"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Keep the app off the real stores and its exports out of the repository."""
    monkeypatch.setattr("sx.tui.app.build_registry", lambda: ([], []))
    monkeypatch.setattr("sx.memory.home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)


def _branched(prompt: str = "the prompt") -> Transcript:
    """A prompt answered twice: the first answer was abandoned by a retry."""
    ask = Message(Role.USER, text=prompt)
    abandoned = Message(Role.ASSISTANT, text="abandoned answer")
    kept = Message(Role.ASSISTANT, text="kept answer")
    return Transcript(
        current=[ask, kept],
        branches=[Branch(label=f"retried after: {prompt}", messages=[abandoned])],
        tree=[
            ask,
            abandoned,
            Message(Role.OTHER, text=f"Branch 2 of 2 — retried after: {prompt}", marker=True),
            kept,
        ],
        hidden=1,
    )


def _linear() -> Transcript:
    """A session that never branched."""
    return Transcript(
        current=[Message(Role.USER, text="only prompt"), Message(Role.ASSISTANT, text="only answer")]
    )


def _edited(original: Transcript) -> Transcript:
    """``original`` after its first prompt was edited and resent, as Pi records it.

    The edit starts a new root, so everything ``original`` held moves to a side
    branch and the current branch holds only the edited exchange.
    """
    edited = [Message(Role.USER, text="edited prompt"), Message(Role.ASSISTANT, text="edited answer")]
    return Transcript(
        current=edited,
        branches=[Branch(label="a separate root", messages=list(original.current))],
        tree=[
            *original.current,
            Message(Role.OTHER, text="Branch 2 of 2 — a separate root", marker=True),
            *edited,
        ],
        hidden=len(original.current),
    )


def _emptied() -> Transcript:
    """A session whose last step went back to the start: its current branch is empty."""
    ask = Message(Role.USER, text="the prompt")
    answer = Message(Role.ASSISTANT, text="the only answer")
    return Transcript(
        current=[],
        branches=[Branch(label="before going back", messages=[ask, answer])],
        tree=[ask, answer],
        hidden=2,
    )


class _Linear(HarnessAdapter):
    """A harness that stores sessions linearly; records every parse."""

    name = "linear"
    display = "Linear"
    capabilities = Capability.BROWSE | Capability.DELETE

    def __init__(self, root: Path, transcripts: dict[str, Transcript]) -> None:
        """Write one back-dated file per transcript, so none counts as live."""
        self._root = root
        self._transcripts = transcripts
        self.parsed: list[str] = []
        root.mkdir(parents=True, exist_ok=True)
        for session_id in transcripts:
            path = root / f"{session_id}.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            os.utime(path, (1_700_000_000, 1_700_000_000))

    def store_roots(self) -> list[Path]:
        """The synthetic store."""
        return [self._root]

    def discover(self):
        """Yield one session per transcript."""
        for session_id in self._transcripts:
            yield Session(
                harness=self.name,
                session_id=session_id,
                project_path="/proj",
                title=session_id,
                modified=datetime(2026, 5, 30, 12, 0),
                paths=[self._root / f"{session_id}.jsonl"],
            )

    def load(self, session: Session) -> list[Message]:
        """Return the session's messages (its current branch)."""
        self.parsed.append(session.session_id)
        return list(self._transcripts[session.session_id].current)


class _Branching(_Linear):
    """A tree-shaped harness, like Pi: ``load_transcript`` returns branches."""

    name = "branchy"
    display = "Branchy"
    supports_branches = True

    def load_transcript(self, session: Session) -> Transcript:
        """Return the session's full transcript."""
        self.parsed.append(session.session_id)
        return self._transcripts[session.session_id]


def _wire(app: SxApp, adapter: HarnessAdapter, tmp_path: Path) -> dict[str, Session]:
    """Inject ``adapter`` in place of the real registry; return its sessions by id."""
    app._adapters_by_name = {adapter.name: adapter}
    app._delete_service = DeleteService(app._adapters_by_name, log_path=tmp_path / "ops.log")
    app._sessions_by_harness = {adapter.name: list(adapter.discover())}
    app._load_errors = []
    app._live_cache = {}
    app._populate_tree()
    return {s.session_id: s for s in app._sessions_by_harness[adapter.name]}


async def _open(app: SxApp, pilot, session: Session) -> None:
    """Open ``session`` in the transcript pane and wait for it to render."""
    await app.open_session(session).wait()
    await pilot.pause()


async def _settle(app: SxApp, pilot) -> None:
    """Let every running worker (an export, a delete) finish."""
    await app.workers.wait_for_complete()
    await pilot.pause()


def _pane(app: SxApp) -> str:
    """Return the transcript pane's text with all whitespace collapsed.

    ``RichLog`` wraps to the app console's width (80 columns in a headless
    test), so line breaks depend on the terminal, not on what was written.
    """
    lines = app.query_one("#transcript", RichLog).lines
    return " ".join(" ".join(line.text for line in lines).split())


def _flat(text: str) -> str:
    """Collapse whitespace the way :func:`_pane` does."""
    return " ".join(text.split())


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


def _pushed(app: SxApp) -> list:
    """Record every screen the app waits on from now on."""
    seen: list = []
    original = app.push_screen_wait

    async def spy(screen):
        """Record ``screen``, then push it and wait for it as the app would.

        Args:
            screen: The screen being pushed.

        Returns:
            What the screen was dismissed with.
        """
        seen.append(screen)
        return await original(screen)

    app.push_screen_wait = spy  # type: ignore[method-assign]
    return seen


def _exports(tmp_path: Path) -> list[Path]:
    """Return the Markdown files exported into the default directory."""
    return sorted((tmp_path / "session-exports").glob("*.md"))


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


def _prompts(screen, selector: str) -> list[str]:
    """Return the plain text of every row in the option list at ``selector``."""
    return [str(option.prompt) for option in screen.query_one(selector, OptionList).options]


# --- t: cycle the view -----------------------------------------------------


def test_t_is_shown_in_the_footer():
    """Changing the view is a primary action, so the footer names it."""
    shown = {binding.key: binding.description for binding in SxApp.BINDINGS if binding.show}
    assert shown.get("t") == "Branches"


@pytest.mark.asyncio
async def test_t_cycles_the_view_without_parsing_again(tmp_path):
    """Each ``t`` re-renders the loaded transcript in the next mode."""
    adapter = _Branching(tmp_path / "store", {"one": _branched()})
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        sessions = _wire(app, adapter, tmp_path)
        await _open(app, pilot, sessions["one"])
        assert _pane(app).startswith(_flat(f"{CURRENT_HEADER}  t: change view ▌ user the prompt"))
        assert "abandoned answer" not in _pane(app)

        await pilot.press("t")
        await pilot.pause()
        assert app._branch_mode is BranchMode.APPENDIX
        assert _pane(app).startswith(_flat(APPENDIX_HEADER))
        assert _flat("── Other branch 1 of 1 — retried after: the prompt ──") in _pane(app)

        await pilot.press("t")
        await pilot.pause()
        assert app._branch_mode is BranchMode.TREE
        assert _pane(app).startswith(_flat(TREE_HEADER))
        assert _flat("── Branch 2 of 2 — retried after: the prompt ──") in _pane(app)

        await pilot.press("t")
        await pilot.pause()
        assert app._branch_mode is BranchMode.CURRENT
        assert _pane(app).startswith(_flat(CURRENT_HEADER))

        assert adapter.parsed == ["one"], "the session must be parsed only once"


@pytest.mark.asyncio
async def test_the_view_carries_over_to_the_next_session(tmp_path):
    """The chosen mode is the app's, not the session's."""
    adapter = _Branching(tmp_path / "store", {"one": _branched(), "two": _branched("other")})
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        sessions = _wire(app, adapter, tmp_path)
        await _open(app, pilot, sessions["one"])
        await pilot.press("t")
        await pilot.pause()

        await _open(app, pilot, sessions["two"])
        assert _pane(app).startswith(_flat(APPENDIX_HEADER))
        assert _flat("── Other branch 1 of 1 — retried after: other ──") in _pane(app)


@pytest.mark.asyncio
async def test_t_on_an_unbranched_session_only_says_so(tmp_path):
    """Nothing to cycle: the mode stays put and the user is told why."""
    adapter = _Branching(tmp_path / "store", {"flat": _linear()})
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        sessions = _wire(app, adapter, tmp_path)
        notices = _notices(app)

        await pilot.press("t")  # nothing open yet
        await pilot.pause()
        assert notices == ["No transcript is open."]

        await _open(app, pilot, sessions["flat"])
        before = _pane(app)
        assert "⑂" not in before
        await pilot.press("t")
        await pilot.pause()
        assert app._branch_mode is BranchMode.CURRENT
        assert notices[-1] == "This session has no other branches."
        assert _pane(app) == before


# --- e: export -------------------------------------------------------------


@pytest.mark.asyncio
async def test_export_of_a_branched_session_asks_which_view(tmp_path, monkeypatch):
    """The dialog starts on the viewer's mode and the export uses the choice."""
    adapter = _Branching(tmp_path / "store", {"one": _branched()})
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["one"]
        monkeypatch.setattr(app, "_current_session", lambda: session)
        await _open(app, pilot, session)

        await pilot.press("e")
        for _ in range(3):
            await pilot.pause()
        assert isinstance(app.screen, BranchModeScreen)
        assert app.screen.query_one("#modes", OptionList).highlighted == 0

        await pilot.press("down", "down", "enter")
        await _settle(app, pilot)

        [export] = _exports(tmp_path)
        md = export.read_text(encoding="utf-8")
        assert f"- **Branches:** {TREE_HEADER.removeprefix('⑂ ')}" in md
        assert "abandoned answer" in md and "kept answer" in md
        # Opened, then read again to decide whether to ask, and once more as it
        # exports: never the pane's copy.
        assert adapter.parsed == ["one", "one", "one"]


@pytest.mark.asyncio
async def test_export_dialog_defaults_to_the_mode_being_viewed(tmp_path, monkeypatch):
    """Enter alone exports what the viewer shows."""
    adapter = _Branching(tmp_path / "store", {"one": _branched()})
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["one"]
        monkeypatch.setattr(app, "_current_session", lambda: session)
        app._branch_mode = BranchMode.APPENDIX

        await pilot.press("e")
        for _ in range(3):
            await pilot.pause()
        assert isinstance(app.screen, BranchModeScreen)
        assert app.screen.query_one("#modes", OptionList).highlighted == 1

        await pilot.press("enter")
        await _settle(app, pilot)

        [export] = _exports(tmp_path)
        assert f"- **Branches:** {APPENDIX_HEADER.removeprefix('⑂ ')}" in export.read_text(
            encoding="utf-8"
        )
        assert adapter.parsed == ["one", "one"], "read to ask, and again as it exports"


@pytest.mark.asyncio
async def test_export_reads_the_session_rather_than_the_pane(tmp_path, monkeypatch):
    """A session that branched since it was opened is exported as it is now (T3).

    The pane's copy predates the branch: exporting it offered no choice and
    left the new branch out.
    """
    adapter = _Branching(tmp_path / "store", {"one": _linear()})
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["one"]
        monkeypatch.setattr(app, "_current_session", lambda: session)
        await _open(app, pilot, session)
        assert "⑂" not in _pane(app)

        adapter._transcripts["one"] = _edited(_linear())  # the harness retries meanwhile
        await pilot.press("e")
        for _ in range(3):
            await pilot.pause()
        assert isinstance(app.screen, BranchModeScreen), "the fresh read has branches"
        await pilot.press("down", "down", "enter")  # the whole tree
        await _settle(app, pilot)

        [export] = _exports(tmp_path)
        md = export.read_text(encoding="utf-8")
        assert "only answer" in md and "edited answer" in md


@pytest.mark.asyncio
async def test_export_holds_what_was_written_while_the_dialog_was_open(tmp_path, monkeypatch):
    """The session is read again after the choice, so the file is not stale (T3)."""
    adapter = _Branching(tmp_path / "store", {"one": _branched()})
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["one"]
        monkeypatch.setattr(app, "_current_session", lambda: session)

        await pilot.press("e")
        for _ in range(3):
            await pilot.pause()
        assert isinstance(app.screen, BranchModeScreen)
        later = _branched()
        later.current.append(Message(Role.USER, text="written while the dialog was open"))
        adapter._transcripts["one"] = later
        await pilot.press("enter")
        await _settle(app, pilot)

        [export] = _exports(tmp_path)
        assert "written while the dialog was open" in export.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_the_export_dialog_marks_the_highlighted_row_as_the_choice(tmp_path, monkeypatch):
    """The mark follows the highlight, and the Export button exports that row (T2)."""
    adapter = _Branching(tmp_path / "store", {"one": _branched()})
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["one"]
        monkeypatch.setattr(app, "_current_session", lambda: session)

        await pilot.press("e")
        for _ in range(3):
            await pilot.pause()
        screen = app.screen
        assert isinstance(screen, BranchModeScreen)
        assert [row[0] for row in _prompts(screen, "#modes")] == ["●", "○", "○"]

        await pilot.press("down")
        screen.query_one("#confirm", Button).focus()  # the list loses focus
        await pilot.pause()
        assert [row[0] for row in _prompts(screen, "#modes")] == ["○", "●", "○"]

        await pilot.press("enter")
        await _settle(app, pilot)
        [export] = _exports(tmp_path)
        assert f"- **Branches:** {APPENDIX_HEADER.removeprefix('⑂ ')}" in export.read_text(
            encoding="utf-8"
        )


@pytest.mark.asyncio
async def test_escape_cancels_the_export(tmp_path, monkeypatch):
    """Backing out of the dialog writes nothing."""
    adapter = _Branching(tmp_path / "store", {"one": _branched()})
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["one"]
        monkeypatch.setattr(app, "_current_session", lambda: session)

        await pilot.press("e")
        for _ in range(3):
            await pilot.pause()
        assert isinstance(app.screen, BranchModeScreen)
        await pilot.press("escape")
        await _settle(app, pilot)

        assert not isinstance(app.screen, BranchModeScreen)
        assert _exports(tmp_path) == []


@pytest.mark.asyncio
async def test_two_quick_enters_on_the_export_dialog_export_once(tmp_path, monkeypatch):
    """The second Enter must not dismiss the dialog again.

    A terminal delivers a burst of keys back to back, so both Enters reached
    the list before the first dismissal landed. The second dismissal popped
    the screen below as well, and with nothing below that, the app crashed with
    ``ScreenStackError``.
    """
    adapter = _Branching(tmp_path / "store", {"one": _branched()})
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["one"]
        monkeypatch.setattr(app, "_current_session", lambda: session)

        await pilot.press("e")
        for _ in range(3):
            await pilot.pause()
        assert isinstance(app.screen, BranchModeScreen)
        for _ in range(2):  # queued together, before either is handled
            key = events.Key("enter", "\r")
            key.set_sender(app)
            app.post_message(key)
        await _settle(app, pilot)

        assert len(app.screen_stack) == 1 and not isinstance(app.screen, BranchModeScreen)
        [export] = _exports(tmp_path)
        assert f"- **Branches:** {CURRENT_HEADER.removeprefix('⑂ ')}" in export.read_text(
            encoding="utf-8"
        )


@pytest.mark.asyncio
async def test_a_dialog_dismissed_twice_only_closes_itself(tmp_path):
    """Every dialog ignores a second dismissal: the first result stands."""
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        results: list = []
        screens = [
            BranchModeScreen(
                descriptions={mode: mode.label() for mode in BranchMode}, default=BranchMode.TREE
            ),
            ConfirmDeleteScreen(heading="Delete?", preview=Text("preview")),
            MoveScreen(heading="Move?", old="/old/proj", preview=lambda *_: (Text(""), False, False)),
        ]
        for screen in screens:
            app.push_screen(screen, results.append)
            await pilot.pause()
            screen.dismiss("first")
            screen.dismiss("second")
            await pilot.pause()
            assert len(app.screen_stack) == 1

        assert results == ["first"] * len(screens)


@pytest.mark.asyncio
async def test_an_unbranched_session_exports_without_asking(tmp_path, monkeypatch):
    """A branching harness's linear session goes straight to the file."""
    adapter = _Branching(tmp_path / "store", {"flat": _linear()})
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["flat"]
        monkeypatch.setattr(app, "_current_session", lambda: session)
        pushed = _pushed(app)

        await pilot.press("e")
        await _settle(app, pilot)

        assert pushed == []
        [export] = _exports(tmp_path)
        md = export.read_text(encoding="utf-8")
        assert "Branches" not in md and "only answer" in md
        assert adapter.parsed == ["flat"]


@pytest.mark.asyncio
async def test_a_harness_without_branches_exports_exactly_as_before(tmp_path, monkeypatch):
    """No dialog, no extra parse, and the same document as before branches."""
    adapter = _Linear(tmp_path / "store", {"flat": _linear()})
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["flat"]
        monkeypatch.setattr(app, "_current_session", lambda: session)
        pushed = _pushed(app)

        await pilot.press("e")
        await _settle(app, pilot)

        assert pushed == []
        assert adapter.parsed == ["flat"], "only the export itself may parse the session"
        [export] = _exports(tmp_path)

        def _undated(markdown: str) -> list[str]:
            """Return the document's lines without the export timestamp.

            Args:
                markdown: An exported document.

            Returns:
                Its lines, minus the ``Exported`` line, which records the clock.
            """
            return [ln for ln in markdown.splitlines() if not ln.startswith("- **Exported:**")]

        expected = messages_to_markdown(session, _linear().current)
        assert _undated(export.read_text(encoding="utf-8")) == _undated(expected)


class _ReadsLinear(_Linear):
    """A linear harness that reads nothing once a session's file is gone, as real ones do."""

    def load(self, session: Session) -> list[Message]:
        """Return the session's messages while its file exists, else none."""
        if not session.primary_path.exists():
            self.parsed.append(session.session_id)
            return []
        return super().load(session)


class _ReadsBranches(_Branching):
    """A branching harness that reads nothing once a session's file is gone."""

    def load_transcript(self, session: Session) -> Transcript:
        """Return the session's transcript while its file exists, else an empty one."""
        if not session.primary_path.exists():
            self.parsed.append(session.session_id)
            return Transcript()
        return super().load_transcript(session)


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_type", [_ReadsLinear, _ReadsBranches], ids=["linear", "branching"])
async def test_exporting_a_session_whose_file_is_gone_writes_nothing(
    tmp_path, monkeypatch, adapter_type
):
    """The export is refused with a toast; it wrote an empty archive and toasted success."""
    adapter = adapter_type(tmp_path / "store", {"one": _branched()})
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["one"]
        monkeypatch.setattr(app, "_current_session", lambda: session)
        notices = _notices(app)
        session.primary_path.unlink()  # deleted elsewhere since the tree was built

        await pilot.press("e")
        await _settle(app, pilot)

        assert not (tmp_path / "session-exports").exists()
        refused = (
            "Nothing was exported — the session's file no longer exists: "
            f"{session.primary_path}. Press r to refresh the list."
        )
        assert notices == [refused]


# --- d: delete, exporting first ---------------------------------------------


@pytest.mark.asyncio
async def test_delete_offers_the_view_for_a_branched_session(tmp_path, monkeypatch):
    """The choice is off until export is ticked, and starts on the whole tree."""
    adapter = _Branching(tmp_path / "store", {"one": _branched()})
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["one"]
        monkeypatch.setattr(app, "_current_session", lambda: session)
        app._branch_mode = BranchMode.APPENDIX  # not what an archive starts on

        app.action_delete_session()
        for _ in range(4):
            await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ConfirmDeleteScreen)
        modes = screen.query_one("#export-mode", OptionList)
        assert modes.disabled is True, "only relevant once the export is ticked"
        assert modes.highlighted == 2, "starts on the whole tree"
        assert [row[0] for row in _prompts(screen, "#export-mode")] == ["○", "○", "●"]

        screen.query_one("#export", Checkbox).value = True
        await pilot.pause()
        assert modes.disabled is False
        await pilot.click("#confirm")
        await _settle(app, pilot)

        [export] = _exports(tmp_path)
        md = export.read_text(encoding="utf-8")
        assert f"- **Branches:** {TREE_HEADER.removeprefix('⑂ ')}" in md
        assert "abandoned answer" in md
        assert not session.primary_path.exists()
        # Read once to decide whether to offer the choice, and again at the
        # moment of export, so the archive holds everything up to the delete.
        assert adapter.parsed == ["one", "one"]


@pytest.mark.asyncio
async def test_export_before_delete_keeps_every_branch_by_default(tmp_path, monkeypatch):
    """Ticking export alone archives the whole tree, whatever the viewer shows (T1).

    It used to start on the viewer's mode — the current branch by default — so
    the delete then destroyed every other branch for good.
    """
    adapter = _Branching(tmp_path / "store", {"one": _branched()})
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["one"]
        monkeypatch.setattr(app, "_current_session", lambda: session)
        assert app._branch_mode is BranchMode.CURRENT

        screen = await _open_delete_dialog(app, pilot)
        screen.query_one("#export", Checkbox).value = True
        await pilot.pause()
        await pilot.click("#confirm")
        await _settle(app, pilot)

        [export] = _exports(tmp_path)
        md = export.read_text(encoding="utf-8")
        assert "abandoned answer" in md and "kept answer" in md
        assert not session.primary_path.exists()


@pytest.mark.asyncio
async def test_an_empty_current_branch_is_called_out_and_not_archived_alone(
    tmp_path, monkeypatch
):
    """The current-branch option says it is empty; the default archive is not (T1)."""
    adapter = _Branching(tmp_path / "store", {"one": _emptied()})
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["one"]
        monkeypatch.setattr(app, "_current_session", lambda: session)

        screen = await _open_delete_dialog(app, pilot)
        assert "EMPTY" in screen._export_modes[BranchMode.CURRENT]
        screen.query_one("#export", Checkbox).value = True
        await pilot.pause()
        await pilot.click("#confirm")
        await _settle(app, pilot)

        [export] = _exports(tmp_path)
        md = export.read_text(encoding="utf-8")
        assert "- **Messages:** 2" in md and "the only answer" in md


@pytest.mark.asyncio
async def test_the_empty_current_branch_is_named_in_the_rows_shown(tmp_path, monkeypatch):
    """The warning is in the row the user reads, in both dialogs."""
    adapter = _Branching(tmp_path / "store", {"one": _emptied()})
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["one"]
        monkeypatch.setattr(app, "_current_session", lambda: session)

        screen = await _open_delete_dialog(app, pilot)
        current, appendix, tree = _prompts(screen, "#export-mode")
        assert "EMPTY" in current and "EMPTY" not in appendix + tree
        screen.dismiss(None)
        await _settle(app, pilot)

        await pilot.press("e")
        for _ in range(3):
            await pilot.pause()
        assert isinstance(app.screen, BranchModeScreen)
        assert "EMPTY" in _prompts(app.screen, "#modes")[0]
        await pilot.press("escape")
        await _settle(app, pilot)


@pytest.mark.asyncio
async def test_a_branch_made_after_the_dialog_opened_is_archived(tmp_path, monkeypatch):
    """No choice was shown, so the archive is the whole tree as it is at export (T1).

    The session was linear when the dialog opened; Pi then recorded an edit of
    the first prompt, which moves the original exchange off the current
    branch. Archiving the current branch would silently drop it.
    """
    adapter = _Branching(tmp_path / "store", {"one": _linear()})
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["one"]
        monkeypatch.setattr(app, "_current_session", lambda: session)

        screen = await _open_delete_dialog(app, pilot)
        assert not screen.query("#export-mode"), "linear when the dialog opened"
        adapter._transcripts["one"] = _edited(_linear())
        screen.query_one("#export", Checkbox).value = True
        await pilot.pause()
        await pilot.click("#confirm")
        await _settle(app, pilot)

        [export] = _exports(tmp_path)
        md = export.read_text(encoding="utf-8")
        assert "only answer" in md, "the original exchange must be in the archive"
        assert "edited answer" in md
        assert not session.primary_path.exists()


@pytest.mark.asyncio
async def test_the_highlighted_row_is_the_export_choice(tmp_path, monkeypatch):
    """Arrowing to a view chooses it; there is no separate press to forget (T2).

    The dialog used a radio set: the arrows moved a highlight that every row
    drew alike, while the export used the pressed row, shown only by color.
    """
    adapter = _Branching(tmp_path / "store", {"one": _branched()})
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["one"]
        monkeypatch.setattr(app, "_current_session", lambda: session)
        app._branch_mode = BranchMode.TREE  # both designs start on the whole tree

        screen = await _open_delete_dialog(app, pilot)
        screen.query_one("#export", Checkbox).value = True
        await pilot.pause()
        screen.query_one("#export-mode").focus()
        await pilot.pause()
        await pilot.press("up")  # the whole tree → current branch + other branches
        screen.query_one("#confirm", Button).focus()
        await pilot.pause()
        await pilot.press("enter")
        await _settle(app, pilot)

        [export] = _exports(tmp_path)
        md = export.read_text(encoding="utf-8")
        assert f"- **Branches:** {APPENDIX_HEADER.removeprefix('⑂ ')}" in md
        assert not session.primary_path.exists()


@pytest.mark.asyncio
async def test_the_chosen_row_is_marked_without_color(tmp_path, monkeypatch):
    """The ● follows the highlight, focused or not, and Enter in the list deletes nothing."""
    adapter = _Branching(tmp_path / "store", {"one": _branched()})
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["one"]
        monkeypatch.setattr(app, "_current_session", lambda: session)

        screen = await _open_delete_dialog(app, pilot)
        screen.query_one("#export", Checkbox).value = True
        await pilot.pause()
        screen.query_one("#export-mode", OptionList).focus()
        await pilot.pause()
        await pilot.press("up", "up")
        assert [row[0] for row in _prompts(screen, "#export-mode")] == ["●", "○", "○"]
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen is screen, "Enter on a row only chooses it"
        assert session.primary_path.exists()

        screen.query_one("#cancel", Button).focus()
        await pilot.pause()
        assert [row[0] for row in _prompts(screen, "#export-mode")] == ["●", "○", "○"]
        assert screen._chosen_mode() is BranchMode.CURRENT


@pytest.mark.asyncio
async def test_delete_dialog_is_unchanged_for_an_unbranched_session(tmp_path, monkeypatch):
    """No branches, no choice: the export step works exactly as it did."""
    adapter = _Branching(tmp_path / "store", {"flat": _linear()})
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["flat"]
        monkeypatch.setattr(app, "_current_session", lambda: session)

        app.action_delete_session()
        for _ in range(4):
            await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ConfirmDeleteScreen)
        assert not screen.query("#export-mode")

        screen.query_one("#export", Checkbox).value = True
        await pilot.pause()
        await pilot.click("#confirm")
        await _settle(app, pilot)

        [export] = _exports(tmp_path)
        assert "Branches" not in export.read_text(encoding="utf-8")
        assert not session.primary_path.exists()


@pytest.mark.asyncio
async def test_delete_does_not_parse_a_session_from_a_harness_without_branches(
    tmp_path, monkeypatch
):
    """Opening the dialog for a linear harness reads no transcript at all."""
    adapter = _Linear(tmp_path / "store", {"flat": _linear()})
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        session = _wire(app, adapter, tmp_path)["flat"]
        monkeypatch.setattr(app, "_current_session", lambda: session)

        app.action_delete_session()
        for _ in range(4):
            await pilot.pause()
        assert isinstance(app.screen, ConfirmDeleteScreen)
        assert not app.screen.query("#export-mode")
        assert adapter.parsed == []


@pytest.mark.asyncio
async def test_the_delete_dialog_returns_the_mode_only_when_offered():
    """The dismissal dict gains ``mode`` with the choice, and nothing without it."""
    transcript = _branched()
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        results: list = []

        app.push_screen(
            ConfirmDeleteScreen(
                heading="Delete?",
                preview=Text("preview"),
                can_export=True,
                export_modes={mode: transcript.describe(mode) for mode in BranchMode},
                export_mode=BranchMode.TREE,
            ),
            results.append,
        )
        await pilot.pause()
        app.screen.query_one("#export", Checkbox).value = True
        await pilot.pause()
        await pilot.click("#confirm")
        await pilot.pause()

        app.push_screen(
            ConfirmDeleteScreen(heading="Delete?", preview=Text("preview"), can_export=True),
            results.append,
        )
        await pilot.pause()
        await pilot.click("#confirm")
        await pilot.pause()

        assert results == [
            {"export": True, "bundle": False, "mode": BranchMode.TREE},
            {"export": False, "bundle": False},
        ]


# --- store warnings ---------------------------------------------------------


class _Warns(_Linear):
    """An adapter with something to say about where its store is."""

    name = "warns"
    display = "Warns"

    def store_warnings(self) -> list[str]:
        """One warning, with an escape in it."""
        return ["sessions may also be written to /elsewhere\x1b[2J"]


class _BrokenCheck(_Linear):
    """An adapter whose store check raises."""

    name = "broken"
    display = "Broken"

    def store_warnings(self) -> list[str]:
        """Fail the way a bug in an adapter would."""
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_store_warnings_join_the_tree_error_rows(tmp_path, monkeypatch):
    """Warnings show where load errors do, sanitized, and a failing check too."""
    warns = _Warns(tmp_path / "a", {})
    broken = _BrokenCheck(tmp_path / "b", {})
    monkeypatch.setattr("sx.tui.app.build_registry", lambda: ([warns, broken], []))
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        rows = [str(node.label) for node in app.query_one("#tree", Tree).root.children]
        assert "⚠ Warns: sessions may also be written to /elsewhere␛[2J" in rows
        assert any(row.startswith("⚠ Broken: store check failed: RuntimeError") for row in rows)


@pytest.mark.asyncio
async def test_a_warning_already_shown_for_claude_is_not_repeated(tmp_path, monkeypatch):
    """If the Claude adapter reports the config warning itself, it shows once."""
    from sx.util import claude_config_warning

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "relative/dir")
    warning = claude_config_warning()
    assert warning

    class _Claude(_Linear):
        """A stand-in for the Claude adapter that reports the config warning itself."""

        name = "claude"
        display = "Claude Code"

        def store_warnings(self) -> list[str]:
            """Report the same warning the app derives from ``CLAUDE_CONFIG_DIR``.

            Returns:
                That one warning.
            """
            return [warning]

    adapter = _Claude(tmp_path / "store", {})
    monkeypatch.setattr("sx.tui.app.build_registry", lambda: ([adapter], []))
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        assert app._load_errors.count(("Claude Code", warning)) == 1


class _Unavailable(_Linear):
    """A harness with no store of its own."""

    name = "missing"
    display = "Missing"

    def available(self) -> bool:
        """Nothing to browse."""
        return False


class _HandedOver(_Unavailable):
    """A harness whose store another harness has taken over."""

    name = "handed"
    display = "Handed"

    def store_owner(self) -> str | None:
        """Name the harness the store belongs to."""
        return "Pi"


class _OwnerFails(_Unavailable):
    """A harness whose ownership check raises."""

    name = "owner-fails"
    display = "Owner fails"

    def store_owner(self) -> str | None:
        """Fail the way a bug in an adapter would."""
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_a_store_handed_over_names_its_owner_in_the_tree(tmp_path, monkeypatch):
    """"store belongs to …" replaces "not installed", as ``sx harnesses`` says."""
    adapters = [
        _HandedOver(tmp_path / "a", {}),
        _OwnerFails(tmp_path / "b", {}),
        _Unavailable(tmp_path / "c", {}),
    ]
    monkeypatch.setattr("sx.tui.app.build_registry", lambda: (adapters, []))
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        rows = [str(node.label) for node in app.query_one("#tree", Tree).root.children]
        assert "· Handed  (store belongs to Pi)" in rows
        assert "· Owner fails  (not installed)" in rows
        assert "· Missing  (not installed)" in rows
