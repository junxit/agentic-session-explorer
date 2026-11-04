"""CLI tests: what ``sx`` prints about stores, sessions and moves.

Adapters' store notes and warnings reach the terminal, and a harness that
handed its store over names its owner; text read from session files does not
drive the terminal; a move plan says what moving an open session does, with
each harness's open-session caveat; and a dry run that predicts a refusal
exits non-zero, as the real move would.

``sx.cli.build_registry`` is patched to fake adapters, so no command here ever
instantiates a real harness adapter or reads a real session store. Stores, the
op-log and the working directory all live under ``tmp_path``.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest

from sx.adapters.base import HarnessAdapter
from sx.cli import LIVE_MOVE_WARNING, main
from sx.model import Capability, Message, MovePlan, MoveResult, Session

NOTE = "sessions read from /custom/store (FAKE_DIR)"
WARNING = "FAKE_DIR is set but empty; reading the default store"
#: An OSC title change, terminated by BEL.
OSC = "\x1b]0;PWNED\x07"


class _Fake(HarnessAdapter):
    """An adapter with one note and one warning about its store."""

    name = "fake"
    display = "Fake"
    capabilities = Capability.BROWSE

    def store_roots(self) -> list[Path]:
        """No store: the harness reads as not installed."""
        return []

    def discover(self):
        """No sessions."""
        return iter(())

    def load(self, session: Session) -> list[Message]:
        """Nothing to load."""
        return []

    def store_notes(self) -> list[str]:
        """Where the store was found."""
        return [NOTE]

    def store_warnings(self) -> list[str]:
        """Something the user should act on."""
        return [WARNING]


class _Broken(_Fake):
    """An adapter whose store check raises."""

    name = "broken"
    display = "Broken"

    def store_warnings(self) -> list[str]:
        """Fail the way a bug in an adapter would."""
        raise RuntimeError("boom")


@pytest.fixture
def registry(monkeypatch) -> list[int]:
    """Serve the fakes as the registry; return a list counting the builds."""
    builds: list[int] = []

    def build():
        """Count the build, then return the fakes as the real registry would.

        Returns:
            ``(adapters, errors)``, as from :func:`sx.registry.build_registry`.
        """
        builds.append(1)
        return [_Fake(), _Broken()], []

    monkeypatch.setattr("sx.cli.build_registry", build)
    monkeypatch.setenv("SX_NO_UPDATE_CHECK", "1")
    return builds


def test_harnesses_lists_store_notes_and_warnings(registry, capsys):
    """Notes and warnings follow the table, named, with warnings marked."""
    assert main(["harnesses"]) == 0
    out, err = capsys.readouterr()
    assert f"\n  Fake: {NOTE}\n" in out
    assert f"\n  ⚠ Fake: {WARNING}\n" in out
    assert out.index("Known harnesses") < out.index(NOTE) < out.index(WARNING)
    assert "  ⚠ Broken: store check failed: RuntimeError('boom')" in out
    assert WARNING not in err, "listed once, not again on stderr"
    assert len(registry) == 1


def test_other_commands_print_store_warnings_to_stderr(registry, capsys):
    """Commands that print Claude's config warning print adapters' warnings too."""
    assert main(["list"]) == 0
    out, err = capsys.readouterr()
    assert f"⚠ Fake: {WARNING}\n" in err
    assert "⚠ Broken: store check failed: RuntimeError('boom')" in err
    assert NOTE not in out + err, "notes belong to `sx harnesses` only"
    assert "No sessions found." in out
    assert len(registry) == 1, "the registry is built once per run"


# --- sessions and moves ---------------------------------------------------------


class _Movable(HarnessAdapter):
    """A harness with sessions under ``tmp_path`` that can be re-pointed.

    Attributes:
        sessions: What :meth:`discover` yields.
        note: A note every move result carries, as a database harness's would.
        moved: The session ids of every plan carried out for real.
    """

    name = "movable"
    display = "Movable"
    capabilities = Capability.BROWSE | Capability.MOVE

    def __init__(self, root: Path, sessions: list[Session], note: str | None = None) -> None:
        """Serve ``sessions`` from a store at ``root``."""
        self._root = root
        self.sessions = sessions
        self.note = note
        self.moved: list[list[str]] = []
        root.mkdir(parents=True, exist_ok=True)

    def store_roots(self) -> list[Path]:
        """The synthetic store."""
        return [self._root]

    def discover(self):
        """Yield the sessions."""
        return iter(list(self.sessions))

    def load(self, session: Session) -> list[Message]:
        """Nothing to load."""
        return []

    def plan_move(self, sessions: list[Session], old: Path, new: Path) -> MovePlan:
        """Plan to rewrite each session's file."""
        return MovePlan(
            harness=self.name,
            old=old,
            new=new,
            sessions=list(sessions),
            rewrites=[s.primary_path for s in sessions if s.primary_path],
        )

    def move(self, plan: MovePlan, *, dry_run: bool = False) -> MoveResult:
        """Report the planned rewrites; record the sessions when not a dry run."""
        if not dry_run:
            self.moved.append([s.session_id for s in plan.sessions])
        return MoveResult(
            dry_run=dry_run,
            rewritten=list(plan.rewrites),
            fields_updated=len(plan.rewrites),
            note=self.note,
        )


class _Careful(_Movable):
    """A harness that warns about moving a session it has open, as Pi does."""

    open_session_caveat = "close the session in Movable first"


class _PartlyRefusing(_Movable):
    """A harness that moves every session but the last, which it refuses."""

    def move(self, plan: MovePlan, *, dry_run: bool = False) -> MoveResult:
        """Rewrite all but the last planned file; refuse that one."""
        result = super().move(plan, dry_run=dry_run)
        *done, last = plan.rewrites
        result.rewritten = done
        result.fields_updated = len(done)
        result.skipped = {last: "the destination was taken after planning (refused)"}
        return result


def _session(
    root: Path,
    session_id: str,
    *,
    title: str = "a title",
    project: str = "/old/proj",
    live: bool = False,
) -> Session:
    """Write a session file under ``root`` and describe it.

    The file is named by its position in the store, since an id may hold
    anything a path cannot.

    Args:
        root: The store.
        session_id: The session's id.
        title: Its title.
        project: The project it records.
        live: Leave the file freshly written instead of back-dating it.

    Returns:
        The session.
    """
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{len(list(root.iterdir()))}.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    if not live:
        os.utime(path, (1_700_000_000, 1_700_000_000))
    return Session(
        harness="movable",
        session_id=session_id,
        project_path=project,
        title=title,
        modified=datetime(2026, 5, 30, 12, 0),
        paths=[path],
    )


@pytest.fixture
def serve(monkeypatch, tmp_path):
    """Return a function that makes the registry serve one adapter.

    The op-log and the working directory move to ``tmp_path`` as well.
    """
    monkeypatch.setenv("SX_NO_UPDATE_CHECK", "1")
    monkeypatch.setenv("SX_LOG_FILE", str(tmp_path / "ops.log"))
    monkeypatch.chdir(tmp_path)

    def install(adapter: HarnessAdapter) -> HarnessAdapter:
        """Serve ``adapter`` as the whole registry.

        Returns:
            The adapter.
        """
        monkeypatch.setattr("sx.cli.build_registry", lambda: ([adapter], []))
        return adapter

    return install


def test_list_neutralizes_escapes_read_from_session_files(serve, tmp_path, capsys):
    """Ids, projects and titles come from the files, so none may drive the terminal (T10)."""
    store = tmp_path / "store"
    session = _session(store, f"{OSC}id", title=f"{OSC}hello title", project=f"/proj{OSC}")
    serve(_Movable(store, [session]))

    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "\x1b" not in out and "\x07" not in out, repr(out)
    assert "␛]0;PWNED␇hello title" in out and "/proj␛]0;PWNED␇" in out


def test_move_plan_lines_are_neutralized(serve, tmp_path, capsys):
    """Paths and notes in ``sx move``'s plan cannot drive the terminal (T10)."""
    store = tmp_path / "store"
    serve(_Movable(store, [_session(store, "s1")], note=f"{OSC}rows updated"))

    assert main(["move", "--from", "/old/proj", "--to", "/new/proj", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "\x1b" not in out and "\x07" not in out, repr(out)
    assert "• ␛]0;PWNED␇rows updated" in out


def test_move_with_yes_still_warns_that_moving_an_open_session_breaks_it(
    serve, tmp_path, capsys
):
    """The live warning says what really happens, and ``--yes`` does not hide it (M9)."""
    store = tmp_path / "store"
    adapter = serve(_Movable(store, [_session(store, "s1", live=True)]))

    assert main(["move", "--from", "/old/proj", "--to", "/new/proj", "--yes"]) == 0
    out = capsys.readouterr().out
    assert f"    ● 1 live session(s) — {LIVE_MOVE_WARNING}\n" in out
    assert "breaks it" in LIVE_MOVE_WARNING and "header-less fragment" in LIVE_MOVE_WARNING
    assert "risks losing those turns" not in out
    assert adapter.moved == [["s1"]]


def test_move_output_shows_the_open_session_caveat(serve, tmp_path, capsys):
    """A harness's warning about open sessions is part of the plan printed."""
    store = tmp_path / "store"
    serve(_Careful(store, [_session(store, "s1")]))

    assert main(["move", "--from", "/old/proj", "--to", "/new/proj", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "\n  Movable (1 session(s))\n    ⚠ Close the session in Movable first\n" in out


def test_a_move_that_refused_anything_exits_nonzero(serve, tmp_path, capsys):
    """A partly done move says so in its exit status, not only in its summary."""
    store = tmp_path / "store"
    serve(_PartlyRefusing(store, [_session(store, "s1"), _session(store, "s2")]))

    assert main(["move", "--from", "/old/proj", "--to", "/new/proj", "--yes"]) == 1
    assert "1 refused" in capsys.readouterr().out


def test_no_caveat_for_a_harness_without_one(serve, tmp_path, capsys):
    """Adapters that set nothing — or predate the attribute — print no caveat."""
    store = tmp_path / "store"
    serve(_Movable(store, [_session(store, "s1")]))

    assert main(["move", "--from", "/old/proj", "--to", "/new/proj", "--dry-run"]) == 0
    assert "⚠" not in capsys.readouterr().out


def test_move_refuses_a_session_that_went_live_at_the_prompt(
    serve, tmp_path, capsys, monkeypatch
):
    """A plain yes covered no live session, so one that went live meanwhile stops the move."""
    store = tmp_path / "store"
    session = _session(store, "s1")
    adapter = serve(_Movable(store, [session]))

    class _Terminal:
        """Stands in for an interactive stdin."""

        @staticmethod
        def isatty() -> bool:
            """Report a terminal."""
            return True

    def answer(prompt: str) -> str:
        """Agree, while the harness writes to the session.

        Args:
            prompt: The question asked.

        Returns:
            ``"y"``.
        """
        assert "MOVE" not in prompt, "nothing was live, so a plain yes is asked for"
        os.utime(session.primary_path, None)
        return "y"

    monkeypatch.setattr("sx.cli.sys.stdin", _Terminal())
    monkeypatch.setattr("builtins.input", answer)

    assert main(["move", "--from", "/old/proj", "--to", "/new/proj"]) == 1
    captured = capsys.readouterr()
    assert "a session became active while waiting for confirmation" in captured.err
    assert adapter.moved == []


class _Blocked(_Movable):
    """A harness whose plan has work that cannot proceed, as a name collision makes."""

    def plan_move(self, sessions: list[Session], old: Path, new: Path) -> MovePlan:
        """Plan the rewrites, and block the last one."""
        plan = super().plan_move(sessions, old, new)
        last = plan.rewrites.pop()
        plan.blocked[last] = "already exists at destination (refused)"
        return plan

    def move(self, plan: MovePlan, *, dry_run: bool = False) -> MoveResult:
        """Report what was blocked as skipped, as every real adapter does."""
        result = super().move(plan, dry_run=dry_run)
        result.skipped.update(plan.blocked)
        return result


@pytest.mark.parametrize("adapter_type", [_PartlyRefusing, _Blocked], ids=["refused", "blocked"])
def test_a_dry_run_that_predicts_a_refusal_exits_nonzero(serve, tmp_path, capsys, adapter_type):
    """A preview listing refusals exits 1, as the real run would; nothing is moved."""
    store = tmp_path / "store"
    adapter = serve(adapter_type(store, [_session(store, "s1"), _session(store, "s2")]))

    assert main(["move", "--from", "/old/proj", "--to", "/new/proj", "--dry-run"]) == 1
    out = capsys.readouterr().out
    assert "    ✗ " in out
    assert out.endswith("\nDry run — nothing was changed; 1 target(s) would be refused.\n")
    assert adapter.moved == []


def test_a_clean_dry_run_exits_zero(serve, tmp_path, capsys):
    """With nothing refused, the preview exits 0 as it always did."""
    store = tmp_path / "store"
    serve(_Movable(store, [_session(store, "s1"), _session(store, "s2")]))

    assert main(["move", "--from", "/old/proj", "--to", "/new/proj", "--dry-run"]) == 0
    assert capsys.readouterr().out.endswith("\nDry run — nothing was changed.\n")


# --- a store handed over to another harness ---------------------------------------


class _Handed(_Fake):
    """A harness that is not installed because another harness took its store over."""

    name = "handed"
    display = "Handed"

    def store_owner(self) -> str | None:
        """Name the harness the store belongs to."""
        return "Pi"


class _OwnerFails(_Fake):
    """A harness whose ownership check raises."""

    name = "owner-fails"
    display = "Owner fails"

    def store_owner(self) -> str | None:
        """Fail the way a bug in an adapter would."""
        raise RuntimeError("boom")


def test_harnesses_names_the_owner_of_a_store_handed_over(monkeypatch, capsys):
    """"store belongs to …" replaces "not installed"; without an owner nothing changes."""
    monkeypatch.setattr(
        "sx.cli.build_registry", lambda: ([_Handed(), _OwnerFails(), _Fake()], [])
    )
    monkeypatch.setenv("SX_NO_UPDATE_CHECK", "1")

    assert main(["harnesses"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "  · Handed         store belongs to Pi" in lines
    assert "  · Owner fails    not installed" in lines
    assert "  · Fake           not installed" in lines
