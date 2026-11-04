"""Moving Oh My Pi sessions: M1, M3, M4, M7, M12 and M13.

Every store is synthetic, under ``tmp_path``; ``omp`` is never run. A running
omp is played by this process holding a ``flock`` on the lease file. The
builders and the autouse sandbox come from :mod:`tests.test_omp_adapter`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sx.adapters import base as base_module
from sx.adapters.omp import OmpAdapter, memory_folder_name
from sx.adapters.pi import encode_session_folder
from sx.model import MovePlan
from tests import test_omp_adapter as base

sandbox = base.sandbox
old = base.old
new = base.new

HOUR = base.HOUR


def _header_of(path: Path) -> dict:
    """Return the session header of an omp transcript (its second line)."""
    return json.loads(base._lines(path)[1])


def _destination(sandbox: Path) -> Path:
    """Return the bucket ``~/work/renamed`` encodes to."""
    return base._sessions(sandbox) / "-work-renamed"


# --- M3: session data no transcript accounts for keeps the bucket in place ------------------


def _foreign(bucket: Path, kind: str) -> Path:
    """Leave one kind of session data in ``bucket`` that no transcript there owns."""
    if kind == "backup":
        entry = bucket / f"{base._name('R')}.0123456789abcdef.bak"
        entry.write_bytes(b"a session omp would restore wherever this lands")
    elif kind == "draft":
        entry = bucket / base._name("D1")[: -len(".jsonl")]
        entry.mkdir()
        (entry / ".draft-only-session").write_text("half a prompt", encoding="utf-8")
    elif kind == "unpacked":
        entry = bucket / f"{base._name('S1')[: -len('.jsonl')]}.unpack-0123456789abcdef.bak"
        (entry / "logs").mkdir(parents=True)
    elif kind == "archived artifacts":
        entry = bucket / base._name("A1")[: -len(".jsonl")]
        entry.mkdir()
    else:
        entry = bucket / f"{base._name('A2')}.gz"
        entry.write_bytes(b"\x1f\x8b archived in place")
    return entry


@pytest.mark.parametrize(
    "kind", ["backup", "draft", "unpacked", "archived artifacts", "archive in the bucket"]
)
def test_unattributable_session_data_forces_per_session_moves(sandbox, old, new, kind):
    """The bucket stays, with what nobody can vouch for; the sessions move out one by one."""
    path = base._session(old)
    artifacts = path.parent / path.stem
    artifacts.mkdir()
    backup = path.parent / f"{path.name}.0123456789abcdef.bak"
    backup.write_bytes(b"its own backup travels with it")
    entry = _foreign(path.parent, kind)
    adapter = OmpAdapter()

    plan, result = base._move(adapter, old, new)

    destination = _destination(sandbox)
    assert "move individually" in (plan.note or "")
    assert entry.exists() and entry.parent == path.parent
    assert {child.name for child in destination.iterdir()} == {path.name, path.stem, backup.name}
    assert not path.exists() and not artifacts.exists() and not backup.exists()
    assert result.refused == {}


def test_an_entry_moving_on_its_own_is_checked_against_the_lease_its_name_gives(
    sandbox, old, new
):
    """An artifacts directory with no transcript yet may be a session omp is writing."""
    bucket = base._bucket(old)
    bucket.mkdir(parents=True)
    draft = _foreign(bucket, "draft")
    target = _destination(sandbox) / draft.name
    adapter = OmpAdapter()
    plan = MovePlan(harness="omp", old=old, new=new, relocations=[(draft, target)])

    with base._flocked(base._lease_dir(sandbox) / "D1.lock"):
        refused = adapter.move(plan)

    assert "has this session open" in refused.refused[draft] and draft.is_dir()
    moved = adapter.move(plan)
    assert moved.moved == [(draft, target)] and target.is_dir()


# --- M4: a session moves whole, or not at all ------------------------------------------------


def test_a_session_whose_backup_cannot_move_is_put_back_whole(sandbox, old, new, monkeypatch):
    """Merged into a bucket that exists: the failed unit is undone, its rewrite included."""
    path = base._session(old)
    artifacts = path.parent / path.stem
    artifacts.mkdir()
    (artifacts / "tool.log").write_text("precious tool output", encoding="utf-8")
    backup = path.parent / f"{path.name}.abc123.bak"
    backup.write_bytes(path.read_bytes())
    base._session(new, "N1")  # the destination bucket exists: merged entry by entry
    before = path.read_bytes()
    real = base_module.rename_no_clobber

    def rename(src: Path, dst: Path) -> str | None:
        """Fail to move the backup, as a full disk or a permission would."""
        return "move failed: [Errno 13] Permission denied" if src == backup else real(src, dst)

    monkeypatch.setattr(base_module, "rename_no_clobber", rename)
    adapter = OmpAdapter()

    _plan, result = base._move(adapter, old, new)

    assert path.read_bytes() == before
    assert (artifacts / "tool.log").exists() and backup.exists()
    assert not (_destination(sandbox) / path.name).exists()
    assert not (_destination(sandbox) / path.stem).exists()
    assert "Permission denied" in result.refused[backup]
    assert result.moved == [] and result.rewritten == []


def test_session_units_carry_the_transcript_last(sandbox, old, new):
    """``S/`` and every backup go first, so a failure never strands them without it."""
    path = base._session(old)
    (path.parent / path.stem).mkdir()
    backups = [path.parent / f"{path.name}.{name}.bak" for name in ("0a", "a.b")]
    for backup in backups:
        backup.write_bytes(b"x")
    base._session(new, "N1")
    adapter = OmpAdapter()

    plan = adapter.plan_move(adapter.sessions_for_project(old), old, new)

    (unit,) = [unit for unit in plan.units if unit.kind == "session"]
    sources = [src for src, _ in unit.relocations]
    assert sources[-1] == path
    assert sorted(sources[:-1]) == sorted([path.parent / path.stem, *backups])
    assert unit.rewrites == [path]


# --- M7: the lease is asked again just before the unit runs ------------------------------------


def test_a_lease_taken_after_planning_refuses_the_move(sandbox, old, new):
    """Planned while free; omp starts writing; nothing is rewritten or moved."""
    path = base._session(old)
    before = path.read_bytes()
    adapter = OmpAdapter()
    plan = adapter.plan_move(adapter.sessions_for_project(old), old, new)
    assert plan.relocations and not plan.blocked

    with base._flocked(base._lease_dir(sandbox) / "S1.lock"):
        result = adapter.move(plan)

    assert "has this session open" in result.refused[path]
    assert path.read_bytes() == before and result.moved == [] and result.rewritten == []


# --- M1: a refused relocation is never rewritten, and buckets merge as planned ---------------


def test_a_session_refused_a_relocation_is_not_rewritten(sandbox, old, new):
    """Its artifacts collide at the destination: it stays whole, re-pointed nowhere."""
    first = base._session(old, "O1")
    (first.parent / first.stem / "tool-1.txt").parent.mkdir()
    (first.parent / first.stem / "tool-1.txt").write_text("artifact", encoding="utf-8")
    second = base._session(old, "O2")
    squatter = _destination(sandbox) / first.stem
    squatter.mkdir(parents=True)
    (squatter / "other.txt").write_text("unrelated", encoding="utf-8")
    base._session(new, "N1")
    before = first.read_bytes()
    adapter = OmpAdapter()

    plan, result = base._move(adapter, old, new)

    assert first.read_bytes() == before and first not in plan.rewrites
    assert "stays with the rest of its session" in result.refused[first]
    assert _header_of(_destination(sandbox) / second.name)["cwd"] == str(new)
    assert [s.session_id for s in adapter.sessions_for_project(old)] == ["O1"]


def test_a_legacy_bucket_and_the_current_one_move_into_one(sandbox, old, new):
    """The first is renamed; the second merges into the folder that rename creates."""
    adapter = OmpAdapter()
    legacy = adapter.sessions_dir() / encode_session_folder(str(old))
    assert legacy.name in adapter.folder_names(str(old)) and legacy != base._bucket(old)
    from_legacy = base._write(
        legacy / base._name("L1"), [base._header("L1", str(old))], slot=base._slot()
    )
    base._age(from_legacy)
    current = base._session(old, "C1")
    lock = current.parent / f".{current.name}.lock.os"
    lock.touch()
    base._age(lock)

    _plan, result = base._move(adapter, old, new)

    destination = _destination(sandbox)
    assert result.refused == {}
    for name in (from_legacy.name, current.name):
        assert _header_of(destination / name)["cwd"] == str(new)
    assert lock.exists() and not current.exists() and not legacy.exists()


# --- M12: memory written moments ago stays ----------------------------------------------------


def test_a_memory_folder_written_in_the_last_90_seconds_does_not_move(sandbox, old, new):
    """omp's memory pipeline runs after a session ends; the plan says why the folder stays."""
    base._session(old)
    memory = base._memories(sandbox) / memory_folder_name(str(old))
    memory.mkdir(parents=True)
    (memory / "memory_summary.md").write_text("written just now", encoding="utf-8")
    adapter = OmpAdapter()

    plan, _result = base._move(adapter, old, new)

    assert "in the last 90 seconds" in plan.blocked[memory]
    assert (memory / "memory_summary.md").exists()
    assert not (base._memories(sandbox) / memory_folder_name(str(new))).exists()


# --- M13: the header a move writes ------------------------------------------------------------


def test_a_move_writes_additional_directories_as_omps_move_to_does(sandbox, old, new):
    """The new ``cwd`` is dropped, an empty list goes, and nothing else is re-pointed."""
    lib = str(old / "lib")
    first = base._session(old, "O1", additionalDirectories=[str(new), lib])
    second = base._session(old, "O2", additionalDirectories=[])
    adapter = OmpAdapter()

    base._move(adapter, old, new)

    moved = {path.name: _header_of(_destination(sandbox) / path.name) for path in (first, second)}
    assert moved[first.name]["additionalDirectories"] == [lib]
    assert "additionalDirectories" not in moved[second.name]
    assert all(header["cwd"] == str(new) for header in moved.values())
