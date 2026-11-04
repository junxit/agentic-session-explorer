"""Moves in the shared encoded-folder core: atomic units, races, and resuming.

A transcript re-pointed at the new directory but left in the folder named for
the old one is invisible to its harness, and re-running the move used to find
"nothing to move". These tests hold each piece of a move — a folder renamed
whole, or one session's transcript with its sidecars — to all or nothing:
refused at planning, refused at run time, failed half way or interrupted, it
is put back, reported truthfully, and finished by the next run.

Claude Code is the harness under test because it is the encoded-folder harness
the core owns; Pi and Oh My Pi inherit the same machinery. Behavior is tested
through APIs that predate these fixes wherever possible, so the tests can also
be pointed at older code. Everything is synthetic under ``tmp_path``.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import time
from pathlib import Path

import pytest

from sx.adapters import base as base_mod
from sx.adapters import claude as claude_mod
from sx.adapters.claude import ClaudeAdapter, encode_project_dir
from sx.service import MoveService
from tests.conftest import write_compact_jsonl

TS = "2026-05-01T10:00:00.000Z"
A = "aaaaaaaa-0000-0000-0000-000000000001"
B = "aaaaaaaa-0000-0000-0000-000000000002"
C = "aaaaaaaa-0000-0000-0000-000000000003"
S = "aaaaaaaa-0000-0000-0000-000000000009"
OLD, NEW = Path("/x/old"), Path("/x/new")

needs_permissions = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root renames directories whatever their mode",
)


# --- builders --------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point Claude at a sandbox and return its projects folder."""
    monkeypatch.setattr(claude_mod, "home", lambda: tmp_path)
    monkeypatch.setattr(claude_mod, "scratchpad_root", lambda: tmp_path / "scratch")
    projects = tmp_path / ".claude" / "projects"
    projects.mkdir(parents=True)
    return projects


def _age(path: Path, seconds: float = 3600) -> None:
    """Backdate ``path`` so it is not live."""
    past = time.time() - seconds
    os.utime(path, (past, past), follow_symlinks=False)


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
    _age(path)
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


def _move(tmp_path: Path, old: Path = OLD, new: Path = NEW, *, dry_run: bool = False):
    """Plan and run a move through the service; return the Claude result."""
    service = _service(tmp_path)
    return service.move(service.plan(old, new), dry_run=dry_run)["claude"]


def _log(tmp_path: Path) -> list[dict]:
    """Return every op-log record."""
    path = tmp_path / "ops.log"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- M1: a relocation refused at planning is not rewritten either ----------------------


def test_a_session_whose_name_is_taken_at_the_destination_stays_whole_and_unrewritten(
    tmp_path, store
):
    """Its transcript and sidecar stay, old directory and all; the rest moves."""
    taken = _session(store, str(OLD), A)
    (taken.parent / A / "subagents").mkdir(parents=True)
    other = _session(store, str(OLD), B)
    squatter = write_compact_jsonl(store / "-x-new" / f"{A}.jsonl", [{"type": "user", "cwd": "keep"}])
    before = taken.read_bytes()

    result = _move(tmp_path)

    assert taken.read_bytes() == before and (taken.parent / A / "subagents").is_dir()
    assert _cwds(squatter) == ["keep"]
    assert _cwds(store / "-x-new" / other.name) == ["/x/new", "/x/new"]
    assert "already exists" in result.refused[taken]
    assert taken not in result.rewritten
    again = _service(tmp_path).plan(OLD, NEW)
    assert [s.session_id for s in again["claude"].sessions] == [A]


# --- M1: destinations created earlier in the same move become merges --------------------


def test_a_stray_sorting_first_no_longer_strands_the_projects_folder(tmp_path, store):
    """The stray used to create the destination, and the folder's rename was refused."""
    _session(store, str(OLD), A)
    _session(store, str(OLD), B)
    (store / "-x-old" / "memory").mkdir()
    (store / "-x-old" / "memory" / "MEMORY.md").write_text("# memory\n", encoding="utf-8")
    _session(store, str(OLD), S, folder="-a-elsewhere")

    preview = _move(tmp_path, dry_run=True)
    result = _move(tmp_path)

    assert _layout(store) == {
        f"-x-new/{A}.jsonl": ["/x/new", "/x/new"],
        f"-x-new/{B}.jsonl": ["/x/new", "/x/new"],
        f"-x-new/{S}.jsonl": ["/x/new", "/x/new"],
    }
    assert (store / "-x-new" / "memory" / "MEMORY.md").is_file()
    assert not (store / "-x-old").exists() and not (store / "-a-elsewhere").exists()
    assert result.refused == {} and not result.failed
    assert sorted(preview.moved) == sorted(result.moved)
    assert _service(tmp_path).plan(OLD, NEW) == {}


# --- M2: a failed rewrite cancels the relocation, and rolls back its unit ---------------


def test_a_rewrite_abandoned_mid_move_relocates_nothing_and_restores_the_rest(
    tmp_path, store, monkeypatch
):
    """The harness appends to one transcript while it is rewritten."""
    first = _session(store, str(OLD), A)
    racing = _session(store, str(OLD), B)
    original = first.read_bytes()
    real = base_mod.rewrite_jsonl

    def racing_rewrite(path, transform, *, dry_run=False):
        """Append a turn to ``racing`` from inside its rewrite."""

        def wrapped(obj):
            """Append once, mid-rewrite, then transform as asked."""
            if not dry_run and path == racing and not getattr(wrapped, "done", False):
                wrapped.done = True
                with open(path, "ab") as fh:
                    fh.write(b'{"type":"user","cwd":"/x/old","uuid":"u9"}\n')
            return transform(obj)

        return real(path, wrapped, dry_run=dry_run)

    monkeypatch.setattr(base_mod, "rewrite_jsonl", racing_rewrite)

    result = _move(tmp_path)

    assert first.read_bytes() == original
    assert _cwds(racing) == ["/x/old", "/x/old", "/x/old"]
    assert not (store / "-x-new").exists()
    assert result.moved == [] and result.rewritten == []
    assert result.failed and "changed while being rewritten" in result.refused[racing]


@needs_permissions
def test_a_sidecar_that_cannot_move_keeps_its_transcript_unrewritten(tmp_path, store):
    """A read-only sidecar cannot change parent; its session is put back whole."""
    path = _session(store, str(OLD), A, folder="-a-elsewhere")
    sidecar = path.parent / A
    (sidecar / "tool-results").mkdir(parents=True)
    _session(store, str(NEW), B)  # the destination folder exists
    before = path.read_bytes()
    os.chmod(sidecar, 0o555)
    try:
        result = _move(tmp_path)
    finally:
        os.chmod(sidecar, 0o755)

    assert path.read_bytes() == before and sidecar.is_dir()
    assert not (store / "-x-new" / path.name).exists()
    assert not (store / "-x-new" / A).exists()
    assert result.failed and sidecar in result.refused


def test_a_failed_rename_puts_back_what_moved_before_it(tmp_path, store, monkeypatch):
    """The transcript's own rename fails after its sidecar moved: both return."""
    path = _session(store, str(OLD), A, folder="-a-elsewhere")
    sidecar = path.parent / A
    (sidecar / "tool-results").mkdir(parents=True)
    before = path.read_bytes()
    real_link = os.link

    def failing_link(src, dst, *args, **kwargs):
        """Fail to link the transcript into place; link anything else."""
        if Path(src) == path:
            raise OSError(errno.EIO, "simulated I/O error", str(dst))
        return real_link(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "link", failing_link)

    result = _move(tmp_path)

    monkeypatch.setattr(os, "link", real_link)
    assert path.read_bytes() == before
    assert (sidecar / "tool-results").is_dir()
    assert not (store / "-x-new").exists()
    assert result.failed and "simulated I/O error" in result.refused[path]


# --- interrupted moves leave a trail, and the next run finishes them ---------------------


def test_ctrl_c_mid_move_puts_the_unit_back_logs_it_and_a_rerun_finishes(
    tmp_path, store, monkeypatch
):
    """Interrupted between two rewrites of one folder: nothing half done survives."""
    first = _session(store, str(OLD), A)
    second = _session(store, str(OLD), B)
    original = first.read_bytes(), second.read_bytes()
    real = base_mod.rewrite_jsonl
    calls = {"real": 0}

    def interrupting(path, transform, *, dry_run=False):
        """Raise ``KeyboardInterrupt`` on the second real rewrite."""
        if not dry_run:
            calls["real"] += 1
            if calls["real"] == 2:
                raise KeyboardInterrupt
        return real(path, transform, dry_run=dry_run)

    monkeypatch.setattr(base_mod, "rewrite_jsonl", interrupting)
    service = _service(tmp_path)
    plans = service.plan(OLD, NEW)
    with pytest.raises(KeyboardInterrupt):
        service.move(plans)

    assert (first.read_bytes(), second.read_bytes()) == original
    assert not (store / "-x-new").exists()
    actions = [(entry["action"], entry.get("status")) for entry in _log(tmp_path)]
    assert actions[0] == ("move_started", None)
    assert actions[-1] == ("move_sessions", "interrupted")

    monkeypatch.setattr(base_mod, "rewrite_jsonl", real)
    _move(tmp_path)
    assert _layout(store) == {
        f"-x-new/{A}.jsonl": ["/x/new", "/x/new"],
        f"-x-new/{B}.jsonl": ["/x/new", "/x/new"],
    }


def test_a_rerun_relocates_sessions_already_re_pointed_but_left_behind(tmp_path, store):
    """What an older, non-atomic move left: headers under the new path, files in the old folder."""
    stranded = _session(store, str(OLD), A)
    stranded.write_text(stranded.read_text().replace('"/x/old"', '"/x/new"'), encoding="utf-8")
    _age(stranded)
    _session(store, str(OLD), B)

    result = _move(tmp_path)

    assert _layout(store) == {
        f"-x-new/{A}.jsonl": ["/x/new", "/x/new"],
        f"-x-new/{B}.jsonl": ["/x/new", "/x/new"],
    }
    assert result.refused == {}


def test_a_rerun_finishes_a_folder_holding_only_stranded_sessions(tmp_path, store):
    """Every header was re-pointed, nothing relocated: the folder still moves, memory too."""
    stranded = _session(store, str(OLD), A)
    stranded.write_text(stranded.read_text().replace('"/x/old"', '"/x/new"'), encoding="utf-8")
    _age(stranded)
    (store / "-x-old" / "memory").mkdir()
    (store / "-x-old" / "memory" / "MEMORY.md").write_text("m", encoding="utf-8")

    plans = _service(tmp_path).plan(OLD, NEW)
    _move(tmp_path)

    assert [s.session_id for s in plans["claude"].sessions] == [A]
    assert _layout(store) == {f"-x-new/{A}.jsonl": ["/x/new", "/x/new"]}
    assert (store / "-x-new" / "memory" / "MEMORY.md").is_file()


# --- M8: the destination is checked at the moment of the rename -------------------------


def _appearing(monkeypatch, destination: Path, *, with_session: bool) -> None:
    """Make ``destination`` appear just before the first rename onto it."""
    real_rename, real_move = os.rename, shutil.move
    state = {"made": False}

    def appear() -> None:
        """Create the destination once, with a session in it if asked."""
        if not state["made"]:
            state["made"] = True
            destination.mkdir(parents=True, exist_ok=True)
            if with_session:
                (destination / "newcomer.jsonl").write_text("{}\n", encoding="utf-8")

    def renaming(src, dst, *args, **kwargs):
        """Create the destination right before renaming onto it."""
        if Path(dst) == destination:
            appear()
        return real_rename(src, dst, *args, **kwargs)

    def moving(src, dst, *args, **kwargs):
        """Create the destination right before moving onto it (older code)."""
        if Path(dst) == destination:
            appear()
        return real_move(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "rename", renaming)
    monkeypatch.setattr(base_mod.shutil, "move", moving)


def test_an_empty_destination_appearing_at_the_last_moment_is_replaced(
    tmp_path, store, monkeypatch
):
    """Pi makes its bucket at startup; an empty one is renamed over, never nested into."""
    _session(store, str(OLD), A)
    service = _service(tmp_path)
    plans = service.plan(OLD, NEW)
    _appearing(monkeypatch, store / "-x-new", with_session=False)

    service.move(plans)

    assert _layout(store) == {f"-x-new/{A}.jsonl": ["/x/new", "/x/new"]}
    assert not (store / "-x-new" / "-x-old").exists()


def test_a_full_destination_appearing_at_the_last_moment_refuses_and_restores(
    tmp_path, store, monkeypatch
):
    """Renaming onto a non-empty folder fails; the unit is put back, nothing nested."""
    path = _session(store, str(OLD), A)
    before = path.read_bytes()
    service = _service(tmp_path)
    plans = service.plan(OLD, NEW)
    _appearing(monkeypatch, store / "-x-new", with_session=True)

    result = service.move(plans)["claude"]

    assert path.read_bytes() == before
    assert sorted(p.name for p in (store / "-x-new").iterdir()) == ["newcomer.jsonl"]
    assert result.failed and "already exists" in result.refused[store / "-x-old"]


def test_a_dangling_link_at_the_destination_is_never_replaced(tmp_path, store):
    """``exists()`` is False for a dangling link; the move must still see it."""
    path = _session(store, str(OLD), A)
    before = path.read_bytes()
    link = store / "-x-new"
    link.symlink_to(tmp_path / "nowhere")

    result = _move(tmp_path)

    assert link.is_symlink() and path.read_bytes() == before
    assert result.failed and result.rewritten == []


def test_a_move_across_file_systems_is_refused_and_put_back(tmp_path, store, monkeypatch):
    """``EXDEV``: sx renames, it never copies."""
    path = _session(store, str(OLD), A)
    before = path.read_bytes()

    def cross_device(src, dst, *args, **kwargs):
        """Fail every rename as a move across file systems would."""
        raise OSError(errno.EXDEV, "Invalid cross-device link", str(dst))

    monkeypatch.setattr(os, "rename", cross_device)

    result = _move(tmp_path)

    assert path.read_bytes() == before and not (store / "-x-new").exists()
    assert "different file systems" in result.refused[store / "-x-old"]


# --- G2: what appeared since planning is respected -------------------------------------


def test_a_destination_folder_created_after_planning_is_merged_into(tmp_path, store):
    """The harness started at the new path between the plan and the move."""
    _session(store, str(OLD), A)
    service = _service(tmp_path)
    plans = service.plan(OLD, NEW)
    _session(store, str(NEW), B)

    result = service.move(plans)["claude"]

    assert _layout(store) == {
        f"-x-new/{A}.jsonl": ["/x/new", "/x/new"],
        f"-x-new/{B}.jsonl": ["/x/new", "/x/new"],
    }
    assert result.refused == {}


def test_a_name_taken_after_planning_is_refused_and_left_alone(tmp_path, store):
    """A file that appeared at the destination is never overwritten."""
    path = _session(store, str(OLD), A, folder="-a-elsewhere")
    before = path.read_bytes()
    service = _service(tmp_path)
    plans = service.plan(OLD, NEW)
    squatter = write_compact_jsonl(store / "-x-new" / path.name, [{"type": "user", "cwd": "keep"}])

    result = service.move(plans)["claude"]

    assert path.read_bytes() == before and _cwds(squatter) == ["keep"]
    assert result.failed and "already exists" in result.refused[path]


# --- M7 (generic part): liveness is asked again just before each unit ------------------


class _OpensLater(ClaudeAdapter):
    """A Claude adapter whose harness opens a session between planning and moving."""

    opened = False

    def destructive_refusal(self, session, *, rewriting):
        """Refuse every session once ``opened`` is set."""
        return "the harness has this session open (refused)" if self.opened else None


def test_a_session_opened_after_planning_is_refused_at_the_move(tmp_path, store):
    """The plan was made while it was free; the move asks again."""
    path = _session(store, str(OLD), A)
    before = path.read_bytes()
    adapter = _OpensLater()
    service = MoveService({"claude": adapter}, log_path=tmp_path / "ops.log")
    plans = service.plan(OLD, NEW)
    adapter.opened = True

    result = service.move(plans)["claude"]

    assert path.read_bytes() == before and not (store / "-x-new").exists()
    assert "has this session open" in result.refused[path]


class _BusyLater(ClaudeAdapter):
    """A Claude adapter whose harness starts using a carried transcript after planning."""

    busy: frozenset[str] = frozenset()

    def busy_transcript(self, path):
        """Report transcripts named in ``busy`` as in use."""
        return path.name in self.busy


def test_a_transcript_busy_after_planning_keeps_its_folder(tmp_path, store):
    """Every transcript a unit carries is checked, selected or not."""
    path = _session(store, str(OLD), A)
    adapter = _BusyLater()
    service = MoveService({"claude": adapter}, log_path=tmp_path / "ops.log")
    plans = service.plan(OLD, NEW)
    adapter.busy = frozenset({path.name})

    result = service.move(plans)["claude"]

    assert path.exists() and not (store / "-x-new").exists()
    assert result.failed


# --- M5: a chain of folders lands where each belongs ----------------------------------


def test_a_folder_moving_into_a_folder_that_moves_itself_lands_in_the_right_place(
    tmp_path, store
):
    """``/x/app -> /x/app-v1`` while ``/x/app/v1`` has sessions: their folders chain."""
    first = _session(store, "/x/app", A)
    second = _session(store, "/x/app/v1", B)
    plans = _service(tmp_path).plan(Path("/x/app"), Path("/x/app-v1"))

    result = _move(tmp_path, Path("/x/app"), Path("/x/app-v1"))

    # Vacated first, the destination is free: renamed whole, not merged.
    assert (store / "-x-app", store / "-x-app-v1") in result.moved
    assert "already at the destination" not in (plans["claude"].note or "")
    assert _layout(store) == {
        f"-x-app-v1/{A}.jsonl": ["/x/app-v1", "/x/app-v1"],
        f"-x-app-v1-v1/{B}.jsonl": ["/x/app-v1/v1", "/x/app-v1/v1"],
    }
    final = {str(a): str(b) for a, b in result.relocated.items()}
    assert final == {
        str(first): str(store / "-x-app-v1" / first.name),
        str(second): str(store / "-x-app-v1-v1" / second.name),
    }
    record = _log(tmp_path)[-1]["harnesses"]["claude"]
    assert record["relocated"] == final
    for _source, destination in record["moved"]:
        assert Path(destination).exists()


def test_the_inverse_of_a_chain_restores_the_layout(tmp_path, store):
    """There and back: every transcript in its original folder, byte for byte."""
    _session(store, "/x/app", A)
    _session(store, "/x/app/v1", B)
    before = {p.relative_to(store): p.read_bytes() for p in store.glob("*/*.jsonl")}

    _move(tmp_path, Path("/x/app"), Path("/x/app-v1"))
    _move(tmp_path, Path("/x/app-v1"), Path("/x/app"))

    after = {p.relative_to(store): p.read_bytes() for p in store.glob("*/*.jsonl")}
    assert after == before


# --- M11: a transcript that merely starts elsewhere travels with its folder --------------


def test_a_transcript_whose_first_directory_encodes_elsewhere_travels_with_the_folder(
    tmp_path, store
):
    """Not a collision: it cannot belong to the folder's other project, so it is no reason to stay."""
    _session(store, str(OLD), A)
    outlier = _session(store, str(OLD), B)
    outlier.write_bytes(b'{"type":"user","cwd":"/y/started-here"}\n' + outlier.read_bytes())
    _age(outlier)
    (store / "-x-old" / "memory").mkdir()
    (store / "-x-old" / "memory" / "MEMORY.md").write_text("m", encoding="utf-8")

    _move(tmp_path)

    assert not (store / "-x-old").exists()
    assert (store / "-x-new" / outlier.name).is_file()
    assert (store / "-x-new" / "memory" / "MEMORY.md").is_file()


# --- M15: the inverse of a merge is not an undo, and the plan says so ------------------


def test_a_merge_into_a_folder_with_sessions_warns_about_the_inverse(tmp_path, store):
    """The sessions already there would travel back with a later inverse move."""
    _session(store, str(OLD), A)
    _session(store, str(NEW), B)
    _session(store, str(NEW), C)

    plans = _service(tmp_path).plan(OLD, NEW)

    note = plans["claude"].note or ""
    assert "2 session(s) already at the destination" in note
    assert "inverse move will move them too" in note


def test_a_move_into_an_empty_destination_has_no_such_warning(tmp_path, store):
    """Nothing is waiting at the destination, so the inverse is an undo."""
    _session(store, str(OLD), A)
    plans = _service(tmp_path).plan(OLD, NEW)
    assert "already at the destination" not in (plans["claude"].note or "")


# --- the op-log is written as the move happens -----------------------------------------


def test_the_op_log_records_the_plan_each_unit_and_the_end(tmp_path, store):
    """Started, one record per unit, finished — older fields kept as they were."""
    _session(store, str(OLD), A)
    _session(store, str(OLD), S, folder="-a-elsewhere")

    _move(tmp_path)

    records = _log(tmp_path)
    assert [r["action"] for r in records] == [
        "move_started",
        "move_unit",
        "move_unit",
        "move_sessions",
    ]
    started, *units, finished = records
    assert started["old"] == str(OLD) and started["new"] == str(NEW)
    assert len(started["harnesses"]["claude"]["relocations"]) == 2
    assert [unit["status"] for unit in units] == ["done", "done"]
    assert finished["status"] == "finished"
    for key in ("rewritten", "moved", "fields_updated", "skipped", "note"):
        assert key in finished["harnesses"]["claude"]


def test_a_dry_run_writes_nothing_to_the_op_log(tmp_path, store):
    """A preview is not an operation."""
    _session(store, str(OLD), A)
    _move(tmp_path, dry_run=True)
    assert _log(tmp_path) == []


# --- G6: symbolic links never carry a move outside the store -----------------------------


def test_a_destination_folder_linked_outside_the_store_is_refused(tmp_path, store):
    """Merging into ``-x-new -> /elsewhere`` would write outside every store root."""
    path = _session(store, str(OLD), A)
    before = path.read_bytes()
    outside = tmp_path / "outside"
    outside.mkdir()
    (store / "-x-new").symlink_to(outside)

    result = _move(tmp_path)

    assert list(outside.iterdir()) == []
    assert path.read_bytes() == before and result.rewritten == []
    assert "outside store root" in next(iter(result.refused.values()))


def test_a_transcript_linked_outside_the_store_is_never_rewritten(tmp_path, store):
    """The rewrite would follow the link and edit a file the store does not own."""
    target = tmp_path / "outside" / "real.jsonl"
    write_compact_jsonl(target, [{"type": "user", "cwd": str(OLD), "sessionId": A}])
    before = target.read_bytes()
    link = store / "-x-old" / f"{A}.jsonl"
    link.parent.mkdir()
    link.symlink_to(target)

    plans = _service(tmp_path).plan(OLD, NEW)
    result = _service(tmp_path).move(plans)["claude"]

    assert target.read_bytes() == before
    assert "outside store root" in plans["claude"].blocked[link]
    assert link not in result.rewritten


def test_a_destination_linked_outside_after_planning_is_refused_at_the_move(tmp_path, store):
    """The plan-time check passed; the run-time check must catch the link."""
    path = _session(store, str(OLD), A, folder="-a-elsewhere")
    before = path.read_bytes()
    service = _service(tmp_path)
    plans = service.plan(OLD, NEW)
    outside = tmp_path / "outside"
    outside.mkdir()
    (store / "-x-new").symlink_to(outside)

    result = service.move(plans)["claude"]

    assert list(outside.iterdir()) == []
    assert path.read_bytes() == before
    assert "outside store root" in result.refused[path]


# --- a subclass that still rewrites through ``_rewrite_all`` is rolled back too ----------


class _LegacyRewrites(ClaudeAdapter):
    """A Claude adapter that customizes ``_rewrite_all``, as Pi's does today."""

    def _rewrite_all(self, plan, result, *, dry_run):
        """Delegate to the inherited rewrite, unchanged."""
        super()._rewrite_all(plan, result, dry_run=dry_run)


def test_a_legacy_rewrite_is_undone_when_its_unit_fails(tmp_path, store, monkeypatch):
    """The snapshot taken before the rewrite restores the transcript."""
    path = _session(store, str(OLD), A, folder="-a-elsewhere")
    (path.parent / A).mkdir()
    before = path.read_bytes()
    real_link = os.link

    def failing_link(src, dst, *args, **kwargs):
        """Fail to link the transcript into place; link anything else."""
        if Path(src) == path:
            raise OSError(errno.EIO, "simulated I/O error", str(dst))
        return real_link(src, dst, *args, **kwargs)

    adapter = _LegacyRewrites()
    service = MoveService({"claude": adapter}, log_path=tmp_path / "ops.log")
    plans = service.plan(OLD, NEW)
    monkeypatch.setattr(os, "link", failing_link)

    result = service.move(plans)["claude"]

    monkeypatch.setattr(os, "link", real_link)
    assert path.read_bytes() == before and (path.parent / A).is_dir()
    assert result.failed
    assert _move(tmp_path).refused == {}
    assert _layout(store) == {f"-x-new/{A}.jsonl": ["/x/new", "/x/new"]}


def test_what_cannot_be_put_back_is_reported_as_it_is_on_disk(tmp_path, store, monkeypatch):
    """The transcript's rename fails, then moving its sidecar back fails too."""
    path = _session(store, str(OLD), A, folder="-a-elsewhere")
    sidecar = path.parent / A
    (sidecar / "tool-results").mkdir(parents=True)
    before = path.read_bytes()
    real_link, real_rename = os.link, os.rename

    def failing_link(src, dst, *args, **kwargs):
        """Fail to link the transcript into place; link anything else."""
        if Path(src) == path:
            raise OSError(errno.EIO, "simulated I/O error", str(dst))
        return real_link(src, dst, *args, **kwargs)

    def failing_return(src, dst, *args, **kwargs):
        """Fail to move the sidecar back; rename anything else."""
        if Path(dst) == sidecar:
            raise OSError(errno.EIO, "simulated I/O error on the way back", str(dst))
        return real_rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "link", failing_link)
    monkeypatch.setattr(os, "rename", failing_return)

    result = _move(tmp_path)

    monkeypatch.setattr(os, "link", real_link)
    monkeypatch.setattr(os, "rename", real_rename)
    stranded = store / "-x-new" / A
    assert path.read_bytes() == before  # never moved, so its rewrite was undone
    assert stranded.is_dir() and not sidecar.exists()
    assert (sidecar, stranded) in result.moved
    assert "re-run the same move" in result.refused[stranded]


def test_an_interrupted_unit_says_so_in_its_op_log_record(tmp_path, store, monkeypatch):
    """The per-unit record of the unit Ctrl-C cut short names the interruption."""
    _session(store, str(OLD), A)
    real = base_mod.rewrite_jsonl

    def interrupting(path, transform, *, dry_run=False):
        """Raise ``KeyboardInterrupt`` on the first real rewrite."""
        if not dry_run:
            raise KeyboardInterrupt
        return real(path, transform, dry_run=dry_run)

    monkeypatch.setattr(base_mod, "rewrite_jsonl", interrupting)
    service = _service(tmp_path)
    plans = service.plan(OLD, NEW)
    with pytest.raises(KeyboardInterrupt):
        service.move(plans)

    (unit,) = [record for record in _log(tmp_path) if record["action"] == "move_unit"]
    assert unit["status"] == "refused"
    assert any("interrupted" in reason for reason in unit["skipped"].values())


def test_a_stray_headed_for_a_folder_linked_outside_is_refused_at_planning(tmp_path, store):
    """A single session's relocation is held to the same containment rule as a folder's."""
    path = _session(store, str(OLD), A, folder="-a-elsewhere")
    before = path.read_bytes()
    outside = tmp_path / "outside"
    outside.mkdir()
    (store / "-x-new").symlink_to(outside)

    plans = _service(tmp_path).plan(OLD, NEW)
    result = _service(tmp_path).move(plans)["claude"]

    assert "outside store root" in plans["claude"].blocked[path]
    assert plans["claude"].relocations == [] and plans["claude"].rewrites == []
    assert list(outside.iterdir()) == [] and path.read_bytes() == before
    assert result.failed


def test_a_store_kept_behind_a_link_moves_like_any_other(tmp_path, monkeypatch):
    """``projects`` links to where the sessions really are: the link is never judged.

    A destination folder that is a link is refused, so nothing is carried
    through it. The store root itself is exempt — a user may keep it on
    another volume — or every session in such a store would be refused.
    """
    monkeypatch.setattr(claude_mod, "home", lambda: tmp_path)
    monkeypatch.setattr(claude_mod, "scratchpad_root", lambda: tmp_path / "scratch")
    real = tmp_path / "volume" / "projects"
    real.mkdir(parents=True)
    (tmp_path / ".claude").mkdir()
    store = tmp_path / ".claude" / "projects"
    store.symlink_to(real, target_is_directory=True)
    _session(store, str(OLD), A)
    _session(store, str(OLD), B, folder="-a-elsewhere")

    result = _move(tmp_path)

    assert result.refused == {}
    assert store.is_symlink()
    assert _layout(real) == {
        f"-x-new/{A}.jsonl": ["/x/new", "/x/new"],
        f"-x-new/{B}.jsonl": ["/x/new", "/x/new"],
    }


class _AppendsRelocations(ClaudeAdapter):
    """A Claude adapter that adds relocations of its own after planning, out of order.

    Oh My Pi does this for its memory folders: one folder merged into a folder
    that the next relocation moves away.
    """

    def plan_move(self, sessions, old, new):
        """Append ``-mem-a/note -> -mem-b/note`` and then ``-mem-b -> -mem-c``."""
        plan = super().plan_move(sessions, old, new)
        root = self.sessions_root()
        plan.relocations.append((root / "-mem-a" / "note.md", root / "-mem-b" / "note.md"))
        plan.relocations.append((root / "-mem-b", root / "-mem-c"))
        return plan


def test_relocations_a_subclass_appends_still_vacate_before_entering(tmp_path, store):
    """Run in the order given, ``-mem-a``'s note would ride along into ``-mem-c``."""
    _session(store, str(OLD), A)
    (store / "-mem-a").mkdir()
    (store / "-mem-a" / "note.md").write_text("a", encoding="utf-8")
    (store / "-mem-b").mkdir()
    (store / "-mem-b" / "kept.md").write_text("b", encoding="utf-8")
    adapter = _AppendsRelocations()
    service = MoveService({"claude": adapter}, log_path=tmp_path / "ops.log")

    result = service.move(service.plan(OLD, NEW))["claude"]

    assert (store / "-mem-b" / "note.md").read_text(encoding="utf-8") == "a"
    assert sorted(p.name for p in (store / "-mem-c").iterdir()) == ["kept.md"]
    assert result.refused == {}


# --- G6: a transcript swapped for a link after planning is never rewritten ----------------


def test_a_transcript_swapped_for_a_link_after_planning_is_not_copied_over_it(tmp_path, store):
    """Rewriting through the link would put a re-pointed copy in its place: two copies."""
    path = _session(store, str(OLD), A)
    service = _service(tmp_path)
    plans = service.plan(OLD, NEW)
    copy = store / "-z-backup" / path.name
    copy.parent.mkdir()
    shutil.copy2(path, copy)
    before = copy.read_bytes()
    path.unlink()
    path.symlink_to(copy)

    result = service.move(plans)["claude"]

    assert path.is_symlink() and copy.read_bytes() == before
    assert not (store / "-x-new").exists()
    assert "no longer a regular file" in result.refused[path]


def test_a_transcript_swapped_for_a_link_outside_after_planning_is_not_copied_over_it(
    tmp_path, store
):
    """The same for a link out of the store: nothing outside changes, no copy appears."""
    path = _session(store, str(OLD), A)
    service = _service(tmp_path)
    plans = service.plan(OLD, NEW)
    outside = tmp_path / "outside" / path.name
    outside.parent.mkdir()
    shutil.copy2(path, outside)
    before = outside.read_bytes()
    path.unlink()
    path.symlink_to(outside)

    result = service.move(plans)["claude"]

    assert outside.read_bytes() == before and path.is_symlink()
    assert result.failed and result.rewritten == []
    assert [p for p in store.rglob("*.jsonl") if not p.is_symlink()] == []


def test_a_folder_swapped_for_a_link_outside_is_not_rewritten_through(tmp_path, store):
    """Old and new encode alike, so the transcript is re-pointed in place — never through a link."""
    old, new = Path("/x/a.b"), Path("/x/a_b")  # both encode to -x-a-b
    path = _session(store, str(old), A)
    service = _service(tmp_path)
    plans = service.plan(old, new)
    assert [unit.kind for unit in plans["claude"].units] == ["rewrite"]
    outside = tmp_path / "outside"
    shutil.move(str(path.parent), str(outside))
    path.parent.symlink_to(outside, target_is_directory=True)
    before = (outside / path.name).read_bytes()

    result = service.move(plans)["claude"]

    assert (outside / path.name).read_bytes() == before
    assert "outside store root" in result.refused[path]


# --- a destination folder that is a link carries nothing into another project -----------


def test_a_stray_is_never_moved_through_a_linked_folder_into_another_project(tmp_path, store):
    """``-x-new`` links to ``-y-other``: the stray would land in that project's folder."""
    path = _session(store, str(OLD), A, folder="-a-elsewhere")
    other = _session(store, "/y/other", B)
    (store / "-x-new").symlink_to(other.parent.name, target_is_directory=True)
    before = path.read_bytes()

    plans = _service(tmp_path).plan(OLD, NEW)
    result = _service(tmp_path).move(plans)["claude"]

    assert path.read_bytes() == before
    assert sorted(p.name for p in other.parent.iterdir()) == [other.name]
    assert "symbolic link" in plans["claude"].blocked[path]
    assert plans["claude"].relocations == [] and result.failed


def test_a_folder_linked_to_another_project_after_planning_receives_no_stray(tmp_path, store):
    """The plan-time check passed; the run-time check catches the link."""
    path = _session(store, str(OLD), A, folder="-a-elsewhere")
    other = _session(store, "/y/other", B)
    before = path.read_bytes()
    service = _service(tmp_path)
    plans = service.plan(OLD, NEW)
    (store / "-x-new").symlink_to(other.parent.name, target_is_directory=True)

    result = service.move(plans)["claude"]

    assert path.read_bytes() == before
    assert sorted(p.name for p in other.parent.iterdir()) == [other.name]
    assert "symbolic link" in result.refused[path]


def test_a_folder_whose_destination_links_to_another_project_is_refused_at_planning(
    tmp_path, store
):
    """The plan itself says so: nothing is planned into the link, so the preview cannot promise it."""
    path = _session(store, str(OLD), A)
    other = _session(store, "/y/other", B)
    link = store / "-x-new"
    link.symlink_to(other.parent.name, target_is_directory=True)
    before = path.read_bytes()

    plans = _service(tmp_path).plan(OLD, NEW)
    result = _service(tmp_path).move(plans)["claude"]

    plan = plans["claude"]
    assert link in plan.blocked and plan.relocations == [] and plan.rewrites == []
    assert sorted(p.name for p in other.parent.iterdir()) == [other.name]
    assert path.read_bytes() == before and result.failed


def test_a_destination_linked_to_another_project_after_planning_refuses_the_folder_whole(
    tmp_path, store
):
    """The link is not a folder to merge into: the folder is refused as one, by name."""
    path = _session(store, str(OLD), A)
    other = _session(store, "/y/other", B)
    before = path.read_bytes()
    service = _service(tmp_path)
    plans = service.plan(OLD, NEW)
    (store / "-x-new").symlink_to(other.parent.name, target_is_directory=True)

    result = service.move(plans)["claude"]

    assert sorted(p.name for p in other.parent.iterdir()) == [other.name]
    assert path.read_bytes() == before
    assert "already exists at destination" in result.refused[store / "-x-old"]


# --- a session moving on its own moves whole, or not at all ------------------------------


def test_a_stray_whose_sidecar_name_is_taken_stays_whole_and_unrewritten(tmp_path, store):
    """Its transcript must not go on without its ``<session-id>/`` folder."""
    path = _session(store, str(OLD), A, folder="-a-elsewhere")
    (path.parent / A / "subagents").mkdir(parents=True)
    (path.parent / A / "subagents" / "agent-1.jsonl").write_text("{}\n", encoding="utf-8")
    taken = store / "-x-new" / A
    taken.mkdir(parents=True)
    (taken / "other.txt").write_text("someone else's\n", encoding="utf-8")
    before = path.read_bytes()

    result = _move(tmp_path)

    assert path.read_bytes() == before
    assert (path.parent / A / "subagents" / "agent-1.jsonl").is_file()
    assert not (store / "-x-new" / path.name).exists()
    assert sorted(p.name for p in taken.iterdir()) == ["other.txt"]
    assert result.rewritten == [] and result.moved == [] and result.failed


def test_a_name_taken_after_planning_is_refused_before_anything_is_rewritten(tmp_path, store):
    """Not rewritten and then restored: the file is never touched (same inode, same mtime)."""
    path = _session(store, str(OLD), A, folder="-a-elsewhere")
    service = _service(tmp_path)
    plans = service.plan(OLD, NEW)
    write_compact_jsonl(store / "-x-new" / path.name, [{"type": "user", "cwd": "keep"}])
    before = os.stat(path)

    result = service.move(plans)["claude"]

    after = os.stat(path)
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert "already exists" in result.refused[path]
