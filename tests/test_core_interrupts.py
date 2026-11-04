"""Moves cut short by Ctrl-C, or by a crash: the disk and the record must agree.

A unit that is interrupted is put back, and only then is its outcome written:
if the put-back is itself interrupted, what it had not undone yet is reported
as done, because it is. A rename interrupted between its link and its unlink
takes the new link back, so a session is never in two places. A harness whose
move stops part-way keeps what its finished units did, and what its adapter
did around them on the way out, in the result it returns and in the final
op-log record. What a harness relocates on its own, such as memory, moves
ahead of its sessions, and a re-run finds it even with no session left.

Claude Code is the harness under test, as in ``test_core_move``, with a
subclass that keeps memory apart from its sessions as Oh My Pi does; the
service's side of an adapter's own account is tested with a fake adapter.
Everything is synthetic under ``tmp_path``.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from sx.adapters import base as base_mod
from sx.adapters import claude as claude_mod
from sx.adapters.base import HarnessAdapter
from sx.adapters.claude import ClaudeAdapter, encode_project_dir
from sx.adapters.relocation import rename_no_clobber
from sx.model import Capability, MovePlan, MoveResult, MoveUnit
from sx.service import MoveService
from tests.conftest import write_compact_jsonl

TS = "2026-05-01T10:00:00.000Z"
A = "aaaaaaaa-0000-0000-0000-000000000001"
B = "aaaaaaaa-0000-0000-0000-000000000002"
C = "aaaaaaaa-0000-0000-0000-000000000003"
OLD, NEW = Path("/x/old"), Path("/x/new")


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point Claude at a sandbox and return its projects folder."""
    monkeypatch.setattr(claude_mod, "home", lambda: tmp_path)
    monkeypatch.setattr(claude_mod, "scratchpad_root", lambda: tmp_path / "scratch")
    projects = tmp_path / ".claude" / "projects"
    projects.mkdir(parents=True)
    return projects


def _session(projects: Path, cwd: str, sid: str, *, folder: str | None = None) -> Path:
    """Write an aged two-turn Claude transcript recording ``cwd``."""
    path = write_compact_jsonl(
        projects / (folder or encode_project_dir(cwd)) / f"{sid}.jsonl",
        [
            {"type": "user", "cwd": cwd, "sessionId": sid, "uuid": "u1",
             "message": {"role": "user", "content": "hi"}, "timestamp": TS},
            {"type": "assistant", "cwd": cwd, "sessionId": sid, "uuid": "a1",
             "message": {"role": "assistant", "content": []}, "timestamp": TS},
        ],
    )
    past = time.time() - 3600
    os.utime(path, (past, past))
    return path


def _cwds(path: Path) -> list[str]:
    """Return every ``cwd`` a transcript records, in order."""
    return [json.loads(line)["cwd"] for line in path.read_text().splitlines() if line.strip()]


def _layout(projects: Path) -> dict[str, list[str]]:
    """Map each transcript, as ``folder/name``, to the directories it records."""
    return {
        f"{path.parent.name}/{path.name}": _cwds(path)
        for path in sorted(projects.glob("*/*.jsonl"))
    }


def _service(tmp_path: Path) -> MoveService:
    """Return a move service over a fresh Claude adapter, logging under ``tmp_path``."""
    return MoveService({"claude": ClaudeAdapter()}, log_path=tmp_path / "ops.log")


def _move(tmp_path: Path):
    """Plan and run ``OLD -> NEW`` through the service; return the Claude result."""
    service = _service(tmp_path)
    return service.move(service.plan(OLD, NEW))["claude"]


def _log(tmp_path: Path) -> list[dict]:
    """Return every op-log record."""
    lines = (tmp_path / "ops.log").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def _interrupted(tmp_path: Path) -> list[dict]:
    """Run ``OLD -> NEW``, expect Ctrl-C to come through, and return the op-log."""
    service = _service(tmp_path)
    plans = service.plan(OLD, NEW)
    with pytest.raises(KeyboardInterrupt):
        service.move(plans)
    return _log(tmp_path)


# --- Ctrl-C between the link and the unlink of a rename ---------------------------------


def test_rename_no_clobber_takes_its_link_back_when_ctrl_c_lands_before_the_unlink(
    tmp_path, monkeypatch
):
    """The source stays, the new link goes, and the interruption comes through."""
    src = tmp_path / "a" / "s.jsonl"
    src.parent.mkdir()
    src.write_text("only copy\n", encoding="utf-8")
    dst = tmp_path / "b" / "s.jsonl"
    dst.parent.mkdir()
    real_unlink = os.unlink

    def interrupting(path, *args, **kwargs):
        """Raise ``KeyboardInterrupt`` instead of unlinking the source."""
        if Path(path) == src:
            raise KeyboardInterrupt
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", interrupting)
    with pytest.raises(KeyboardInterrupt):
        rename_no_clobber(src, dst)
    monkeypatch.setattr(os, "unlink", real_unlink)

    assert src.read_text(encoding="utf-8") == "only copy\n"
    assert not dst.exists()


def test_ctrl_c_between_link_and_unlink_leaves_no_second_copy(tmp_path, store, monkeypatch):
    """A folder merged file by file: the interrupted session is back, once; a re-run finishes."""
    first = _session(store, str(OLD), A)
    second = _session(store, str(OLD), B)
    _session(store, str(NEW), C)  # the destination exists, so the folder merges
    real_unlink = os.unlink
    state = {"done": False}

    def interrupting(path, *args, **kwargs):
        """Raise ``KeyboardInterrupt`` once, instead of unlinking the second transcript."""
        if not state["done"] and Path(path) == second:
            state["done"] = True
            raise KeyboardInterrupt
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", interrupting)
    log = _interrupted(tmp_path)
    monkeypatch.setattr(os, "unlink", real_unlink)

    assert _cwds(second) == ["/x/old", "/x/old"]
    assert not (store / "-x-new" / second.name).exists()
    assert _cwds(store / "-x-new" / first.name) == ["/x/new", "/x/new"]
    units = [record for record in log if record["action"] == "move_unit"]
    assert units[-1]["status"] == "refused"
    assert "put back" in units[-1]["skipped"][str(second)]

    assert _move(tmp_path).refused == {}
    assert _layout(store) == {
        f"-x-new/{A}.jsonl": ["/x/new", "/x/new"],
        f"-x-new/{B}.jsonl": ["/x/new", "/x/new"],
        f"-x-new/{C}.jsonl": ["/x/new", "/x/new"],
    }


def test_ctrl_c_right_after_the_link_returns_leaves_no_second_copy(tmp_path, store, monkeypatch):
    """Not even inside the rename: the unit's put-back finds the half-done link and takes it back."""
    path = _session(store, str(OLD), A, folder="-a-elsewhere")
    before = path.read_bytes()
    real_link = os.link
    state = {"done": False}

    def linking(src, dst, *args, **kwargs):
        """Link, then Ctrl-C once, before the source is unlinked."""
        real_link(src, dst, *args, **kwargs)
        if not state["done"] and Path(src) == path:
            state["done"] = True
            raise KeyboardInterrupt

    monkeypatch.setattr(os, "link", linking)
    _interrupted(tmp_path)
    monkeypatch.setattr(os, "link", real_link)

    assert path.read_bytes() == before
    assert not (store / "-x-new" / path.name).exists()


# --- the outcome is written once the put-back has run ------------------------------------


def test_a_second_ctrl_c_during_the_put_back_is_recorded_as_it_is(tmp_path, store, monkeypatch):
    """The first transcript's undo never ran: the record says it is still re-pointed."""
    first = _session(store, str(OLD), A)
    second = _session(store, str(OLD), B)
    real = base_mod.rewrite_jsonl
    calls = {"real": 0}

    def interrupting(path, transform, *, dry_run=False):
        """Ctrl-C on the second rewrite, and again on the undo of the first."""
        if not dry_run:
            calls["real"] += 1
            if calls["real"] in (2, 3):
                raise KeyboardInterrupt
        return real(path, transform, dry_run=dry_run)

    monkeypatch.setattr(base_mod, "rewrite_jsonl", interrupting)
    log = _interrupted(tmp_path)
    monkeypatch.setattr(base_mod, "rewrite_jsonl", real)

    assert _cwds(first) == ["/x/new", "/x/new"] and _cwds(second) == ["/x/old", "/x/old"]
    (unit,) = [record for record in log if record["action"] == "move_unit"]
    assert unit["status"] == "partial" and unit["rewritten"] == [str(first)]
    assert "interrupted" in unit["skipped"][str(first)]
    assert "interrupted again" in unit["skipped"][str(store / "-x-old")]
    final = log[-1]
    assert final["status"] == "interrupted"
    assert final["harnesses"]["claude"]["rewritten"] == [str(first)]

    _move(tmp_path)
    assert _layout(store) == {
        f"-x-new/{A}.jsonl": ["/x/new", "/x/new"],
        f"-x-new/{B}.jsonl": ["/x/new", "/x/new"],
    }


def test_ctrl_c_just_after_a_rewrite_is_swapped_in_still_puts_it_back(
    tmp_path, store, monkeypatch
):
    """The rewrite reached the disk but never returned; it is undone with the rest."""
    first = _session(store, str(OLD), A)
    second = _session(store, str(OLD), B)
    originals = first.read_bytes(), second.read_bytes()
    real_replace = os.replace
    state = {"done": False}

    def replacing(src, dst, *args, **kwargs):
        """Swap the copy in, then Ctrl-C once, for the second transcript."""
        result = real_replace(src, dst, *args, **kwargs)
        if not state["done"] and Path(dst) == second:
            state["done"] = True
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(os, "replace", replacing)
    log = _interrupted(tmp_path)
    monkeypatch.setattr(os, "replace", real_replace)

    assert (first.read_bytes(), second.read_bytes()) == originals
    (unit,) = [record for record in log if record["action"] == "move_unit"]
    assert unit["rewritten"] == [] and "put back" in unit["skipped"][str(store / "-x-old")]


def test_ctrl_c_just_after_a_rename_finished_still_puts_it_back(tmp_path, store, monkeypatch):
    """The rename ran but was never recorded: the put-back moves it back all the same."""
    path = _session(store, str(OLD), A, folder="-a-elsewhere")
    before = path.read_bytes()
    real = base_mod.rename_no_clobber
    state = {"done": False}

    def renaming(src, dst):
        """Rename the transcript, then Ctrl-C before the unit hears of it."""
        error = real(src, dst)
        if not state["done"] and src == path:
            state["done"] = True
            raise KeyboardInterrupt
        return error

    monkeypatch.setattr(base_mod, "rename_no_clobber", renaming)
    _interrupted(tmp_path)
    monkeypatch.setattr(base_mod, "rename_no_clobber", real)

    assert path.read_bytes() == before
    assert not (store / "-x-new" / path.name).exists()


# --- what the put-back could not undo, or had already undone --------------------------------


def _renames(monkeypatch: pytest.MonkeyPatch, *steps: str) -> None:
    """Script the move's renames, one step per call, then rename as usual.

    Steps: ``"ctrl-c-after"`` renames, then raises ``KeyboardInterrupt`` before
    the unit hears of it; ``"ctrl-c-before"`` raises it without renaming;
    ``"fail"`` renames nothing and returns an error, as a failed rename does.
    """
    real = base_mod.rename_no_clobber
    script = list(steps)

    def renaming(src, dst):
        """Play the next step of the script."""
        step = script.pop(0) if script else None
        if step == "ctrl-c-before":
            raise KeyboardInterrupt
        if step == "fail":
            return "move failed: [Errno 5] simulated I/O error"
        error = real(src, dst)
        if step == "ctrl-c-after":
            raise KeyboardInterrupt
        return error

    monkeypatch.setattr(base_mod, "rename_no_clobber", renaming)


def _unit_record(tmp_path: Path) -> dict:
    """Return the one per-unit record of the move."""
    (unit,) = [record for record in _log(tmp_path) if record["action"] == "move_unit"]
    return unit


def test_a_put_back_that_cannot_undo_a_rename_says_what_stayed(tmp_path, store, monkeypatch):
    """The folder moved, Ctrl-C came, and moving it back failed: "except as noted"."""
    path = _session(store, str(OLD), A)
    _renames(monkeypatch, "ctrl-c-after", "fail")

    _interrupted(tmp_path)

    moved = store / "-x-new" / path.name
    assert not path.exists() and _cwds(moved) == ["/x/new", "/x/new"], "consistent where it is"
    unit = _unit_record(tmp_path)
    assert "put back except as noted" in unit["skipped"][str(store / "-x-old")]
    assert "could not be put back" in unit["skipped"][str(store / "-x-new")]
    assert unit["moved"] == [[str(store / "-x-old"), str(store / "-x-new")]]
    assert unit["rewritten"] == [str(path)]


def test_a_put_back_cut_short_right_after_its_last_undo_reports_nothing_still_done(
    tmp_path, store, monkeypatch
):
    """Both rewrites were undone, the second Ctrl-C a moment too late to matter."""
    first = _session(store, str(OLD), A)
    second = _session(store, str(OLD), B)
    originals = first.read_bytes(), second.read_bytes()
    _renames(monkeypatch, "ctrl-c-before")
    real = base_mod.rewrite_jsonl
    calls = {"real": 0}

    def rewriting(path, transform, *, dry_run=False):
        """Rewrite; on the fourth real call (the first transcript's undo), Ctrl-C after it."""
        result = real(path, transform, dry_run=dry_run)
        if not dry_run:
            calls["real"] += 1
            if calls["real"] == 4:
                raise KeyboardInterrupt
        return result

    monkeypatch.setattr(base_mod, "rewrite_jsonl", rewriting)
    _interrupted(tmp_path)
    monkeypatch.setattr(base_mod, "rewrite_jsonl", real)

    assert (first.read_bytes(), second.read_bytes()) == originals
    unit = _unit_record(tmp_path)
    assert "interrupted again" in unit["skipped"][str(store / "-x-old")]
    assert unit["rewritten"] == [] and unit["moved"] == []

    _move(tmp_path)
    assert _layout(store) == {
        f"-x-new/{A}.jsonl": ["/x/new", "/x/new"],
        f"-x-new/{B}.jsonl": ["/x/new", "/x/new"],
    }


def test_a_put_back_cut_short_right_after_a_rename_back_does_not_report_it_moved(
    tmp_path, store, monkeypatch
):
    """The folder is back; its transcript's undo never ran, and only that is reported."""
    path = _session(store, str(OLD), A)
    _renames(monkeypatch, "ctrl-c-after", "ctrl-c-after")

    _interrupted(tmp_path)

    assert _cwds(path) == ["/x/new", "/x/new"] and not (store / "-x-new").exists()
    unit = _unit_record(tmp_path)
    assert unit["moved"] == [], "the rename was undone before the second Ctrl-C"
    assert unit["rewritten"] == [str(path)]
    assert "was not put back" in unit["skipped"][str(path)]

    _move(tmp_path)
    assert _layout(store) == {f"-x-new/{A}.jsonl": ["/x/new", "/x/new"]}


# --- a harness stopped part-way keeps what its units did -------------------------------


def _stop_second_rename(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> None:
    """Make the second rename of the move raise ``error``."""
    real = base_mod.rename_no_clobber
    calls = {"n": 0}

    def renaming(src, dst):
        """Raise on the second call; rename otherwise."""
        calls["n"] += 1
        if calls["n"] == 2:
            raise error
        return real(src, dst)

    monkeypatch.setattr(base_mod, "rename_no_clobber", renaming)


def test_an_interrupted_move_keeps_its_finished_units_in_the_final_record(
    tmp_path, store, monkeypatch
):
    """The first folder moved, the second was put back: the last record says both."""
    first = _session(store, str(OLD), A)
    _session(store, f"{OLD}/sub", B)
    _stop_second_rename(monkeypatch, KeyboardInterrupt())

    log = _interrupted(tmp_path)

    assert _layout(store) == {
        f"-x-new/{A}.jsonl": ["/x/new", "/x/new"],
        f"-x-old-sub/{B}.jsonl": ["/x/old/sub", "/x/old/sub"],
    }
    final = log[-1]
    assert final["action"] == "move_sessions" and final["status"] == "interrupted"
    record = final["harnesses"]["claude"]
    assert record["moved"] == [[str(store / "-x-old"), str(store / "-x-new")]]
    assert record["rewritten"] == [str(first)]
    assert record["relocated"] == {str(first): str(store / "-x-new" / first.name)}
    assert "put back" in record["skipped"][str(store / "-x-old-sub")]


def test_a_move_that_crashes_part_way_returns_what_its_finished_units_did(
    tmp_path, store, monkeypatch
):
    """The catch-all used to replace the whole result with "move failed"."""
    first = _session(store, str(OLD), A)
    _session(store, f"{OLD}/sub", B)
    _stop_second_rename(monkeypatch, RuntimeError("disk on fire"))

    result = _move(tmp_path)

    assert result.moved == [(store / "-x-old", store / "-x-new")]
    assert result.rewritten == [first] and not result.failed
    assert "disk on fire" in result.skipped[Path("<claude>")]
    assert "put back" in result.skipped[store / "-x-old-sub"]
    assert _log(tmp_path)[-1]["harnesses"]["claude"]["moved"] == [
        [str(store / "-x-old"), str(store / "-x-new")]
    ]


# --- what a harness relocates on its own runs ahead of its sessions ----------------------


class _WithMemory(ClaudeAdapter):
    """Claude, with a memory folder per project kept apart from its sessions.

    The memory folder is planned the way Oh My Pi plans its own: added to the
    plan's relocations once the session units are cut, outside every unit.
    """

    def memories(self) -> Path:
        """Return the directory holding one memory folder per project."""
        return self._config_dir() / "memories"

    def store_roots(self) -> list[Path]:
        """Return Claude's roots and the memory directory."""
        return [*super().store_roots(), self.memories()]

    def plan_move(self, sessions, old, new):
        """Plan the sessions as Claude does, then the project's memory folder."""
        plan = super().plan_move(sessions, old, new)
        source = self.memories() / encode_project_dir(str(old))
        if source.is_dir():
            destination = self.memories() / encode_project_dir(str(new))
            self._plan_relocation(source, destination, plan, self.store_roots())
        return plan


def _memory(tmp_path: Path, project: Path) -> Path:
    """Give ``project`` a memory folder holding one note, and return the folder."""
    folder = tmp_path / ".claude" / "memories" / encode_project_dir(str(project))
    folder.mkdir(parents=True)
    (folder / "MEMORY.md").write_text("# remember this\n", encoding="utf-8")
    return folder


def _memory_service(tmp_path: Path) -> MoveService:
    """Return a move service over the memory-keeping adapter."""
    return MoveService({"claude": _WithMemory()}, log_path=tmp_path / "ops.log")


def _interrupt_first_rename_of(monkeypatch: pytest.MonkeyPatch, source: Path) -> None:
    """Make the first rename of ``source`` raise ``KeyboardInterrupt`` before it starts."""
    real = base_mod.rename_no_clobber
    state = {"done": False}

    def renaming(src, dst):
        """Ctrl-C once, instead of renaming ``source``."""
        if not state["done"] and src == source:
            state["done"] = True
            raise KeyboardInterrupt
        return real(src, dst)

    monkeypatch.setattr(base_mod, "rename_no_clobber", renaming)


def test_memory_moves_before_the_sessions(tmp_path, store, monkeypatch):
    """Ctrl-C at the session folder: the memory has moved, the session waits for a re-run."""
    transcript = _session(store, str(OLD), A)
    memory = _memory(tmp_path, OLD)
    moved = memory.parent / encode_project_dir(str(NEW))
    _interrupt_first_rename_of(monkeypatch, store / "-x-old")
    service = _memory_service(tmp_path)

    with pytest.raises(KeyboardInterrupt):
        service.move(service.plan(OLD, NEW))

    assert (moved / "MEMORY.md").is_file() and not memory.exists()
    assert _cwds(transcript) == ["/x/old", "/x/old"], "put back, for a re-run to find"
    units = [record for record in _log(tmp_path) if record["action"] == "move_unit"]
    assert units[0]["planned"] == [[str(memory), str(moved)]]

    service.move(service.plan(OLD, NEW))
    assert _layout(store) == {f"-x-new/{A}.jsonl": ["/x/new", "/x/new"]}


def test_an_interrupted_memory_move_is_finished_by_a_rerun(tmp_path, store, monkeypatch):
    """Ctrl-C at the memory folder: the same move run again moves everything."""
    _session(store, str(OLD), A)
    memory = _memory(tmp_path, OLD)
    _interrupt_first_rename_of(monkeypatch, memory)
    service = _memory_service(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        service.move(service.plan(OLD, NEW))

    service.move(service.plan(OLD, NEW))

    assert (memory.parent / encode_project_dir(str(NEW)) / "MEMORY.md").is_file()
    assert not memory.exists()
    assert _layout(store) == {f"-x-new/{A}.jsonl": ["/x/new", "/x/new"]}


def test_memory_left_behind_by_its_sessions_is_moved_by_a_rerun(tmp_path, store):
    """Every session already moved, the memory did not: no session selects the harness now."""
    _session(store, str(NEW), A)
    memory = _memory(tmp_path, OLD)
    service = _memory_service(tmp_path)

    plans = service.plan(OLD, NEW)
    assert "claude" in plans and plans["claude"].sessions == []
    results = service.move(plans)

    moved = memory.parent / encode_project_dir(str(NEW))
    assert (moved / "MEMORY.md").is_file() and not memory.exists()
    assert results["claude"].moved == [(memory, moved)]
    assert service.plan(OLD, NEW) == {}, "nothing is left, so nothing is planned"


class _Sessionless(HarnessAdapter):
    """A movable harness with no session, whose plan for one anyway says something."""

    name = "sessionless"
    display = "Sessionless"
    capabilities = Capability.BROWSE | Capability.MOVE

    def __init__(self, root: Path, plan: Callable[[Path, Path], MovePlan]) -> None:
        """Keep ``root`` as the store and ``plan`` as what planning returns."""
        self._root = root
        self._plan = plan

    def store_roots(self) -> list[Path]:
        """Return the one root."""
        return [self._root]

    def discover(self):
        """No sessions."""
        return iter(())

    def load(self, session):
        """No transcript."""
        return []

    def plan_move(self, sessions, old, new):
        """Return what the test asked for."""
        return self._plan(old, new)


def _noting(old: Path, new: Path) -> MovePlan:
    """A plan with a note and a refusal, and nothing to relocate."""
    return MovePlan(
        harness="sessionless",
        old=old,
        new=new,
        note="2 shared rows left untouched",
        blocked={Path("/elsewhere"): "not this harness's (refused)"},
    )


def _failing(old: Path, new: Path) -> MovePlan:
    """Planning that fails."""
    raise RuntimeError("store unreadable")


@pytest.mark.parametrize("planner", [_noting, _failing], ids=["note-and-refusal", "failure"])
def test_a_harness_with_no_session_adds_nothing_it_would_not_relocate(tmp_path, planner):
    """Only a plan that moves something speaks for a harness no session selected."""
    root = tmp_path / "store"
    root.mkdir()
    service = MoveService({"sessionless": _Sessionless(root, planner)}, log_path=tmp_path / "ops.log")

    assert service.plan(OLD, NEW) == {}


# --- an adapter's own account of a move cut short ---------------------------------------


class _Relinking(HarnessAdapter):
    """A fake harness that does work of its own around its units, as Pi does.

    Its move runs two units, reporting each through the plan's observer, then
    — as Pi re-points the fork links of the units that finished on its way out
    — rewrites one more file, and lets ``error`` through, with that work, which
    no unit reported, attached as ``partial_result`` when ``attach``.
    """

    name = "relinking"
    display = "Relinking"
    capabilities = Capability.BROWSE | Capability.MOVE

    def __init__(self, root: Path, error: BaseException, *, attach: bool) -> None:
        """Keep the store, what stops the move, and whether to attach the account."""
        self._root = root
        self.error = error
        self.attach = attach
        self.fork = root / "fork.jsonl"
        self.pairs = [(root / "old" / f"{n}.jsonl", root / "new" / f"{n}.jsonl") for n in "ab"]

    def store_roots(self) -> list[Path]:
        """Return the one root."""
        return [self._root]

    def discover(self):
        """No sessions."""
        return iter(())

    def load(self, session):
        """No transcript."""
        return []

    def move(self, plan, *, dry_run=False):
        """Run two units, re-point the fork on the way out, then stop."""
        for src, dst in self.pairs:
            unit = MoveUnit(MoveUnit.SESSION, relocations=[(src, dst)], rewrites=[src])
            outcome = MoveResult(
                dry_run=dry_run,
                moved=[(src, dst)],
                rewritten=[src],
                fields_updated=2,
                relocated={src: dst},
            )
            plan.on_unit(unit, outcome)
        on_the_way_out = MoveResult(dry_run=dry_run, rewritten=[self.fork], fields_updated=1)
        on_the_way_out.skipped[self._root / "busy.jsonl"] = (
            "its fork link was not re-pointed (refused)"
        )
        if self.attach:
            self.error.partial_result = on_the_way_out
        raise self.error


def _relink(tmp_path: Path, how: str, *, attach: bool) -> tuple[_Relinking, dict]:
    """Run the fake harness's move through the service; return it and the final record.

    Args:
        tmp_path: The test's sandbox.
        how: ``"interrupted"`` (Ctrl-C) or ``"failed"`` (an error).
        attach: Whether the adapter attaches its account to the exception.
    """
    error = KeyboardInterrupt() if how == "interrupted" else RuntimeError("disk on fire")
    adapter = _Relinking(tmp_path / "store", error, attach=attach)
    service = MoveService({adapter.name: adapter}, log_path=tmp_path / "ops.log")
    plan = MovePlan(
        harness=adapter.name,
        old=OLD,
        new=NEW,
        relocations=list(adapter.pairs),
        blocked={tmp_path / "store" / "pinned.jsonl": "planned refusal (refused)"},
    )
    if isinstance(error, Exception):
        result = service.move({adapter.name: plan})[adapter.name]
        record = _log(tmp_path)[-1]["harnesses"][adapter.name]
        assert record["rewritten"] == [str(p) for p in result.rewritten], "result and log agree"
    else:
        with pytest.raises(type(error)):
            service.move({adapter.name: plan})
        record = _log(tmp_path)[-1]["harnesses"][adapter.name]
    return adapter, record


@pytest.mark.parametrize("how", ["interrupted", "failed"])
def test_what_a_move_did_on_its_way_out_is_in_the_final_record(tmp_path, how):
    """The fork re-pointed after the units is recorded, and the units' work only once."""
    adapter, record = _relink(tmp_path, how, attach=True)
    (a_src, a_dst), (b_src, b_dst) = adapter.pairs

    assert record["rewritten"] == [str(a_src), str(b_src), str(adapter.fork)]
    assert record["moved"] == [[str(a_src), str(a_dst)], [str(b_src), str(b_dst)]]
    assert record["relocated"] == {str(a_src): str(a_dst), str(b_src): str(b_dst)}
    assert record["fields_updated"] == 5
    skipped = record["skipped"]
    assert skipped[str(tmp_path / "store" / "busy.jsonl")].startswith("its fork link")
    assert skipped[str(tmp_path / "store" / "pinned.jsonl")] == "planned refusal (refused)"
    assert "<relinking>" in skipped


@pytest.mark.parametrize("how", ["interrupted", "failed"])
def test_without_an_attached_account_the_units_are_still_recorded(tmp_path, how):
    """An adapter that attaches nothing keeps what its units reported, as before."""
    adapter, record = _relink(tmp_path, how, attach=False)
    (a_src, _), (b_src, _) = adapter.pairs

    assert record["rewritten"] == [str(a_src), str(b_src)]
    assert record["fields_updated"] == 4
    assert "<relinking>" in record["skipped"]
