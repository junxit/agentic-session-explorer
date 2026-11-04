"""Case-only project renames, and a destination that is the source itself.

On a case-insensitive file system — APFS and HFS+ by default — ``/x/App`` and
``/x/app`` are one directory, so the folder a project's new spelling encodes to
"exists" before anything moves: it is the old folder. A move used to take that
for a destination to merge into, re-plan the same rename, and recurse until
Python gave up, reporting nothing done. These tests hold the core to a plain
rename that changes only the case, or a clean refusal, and never to unlinking
the only copy of a file that is reached under two spellings.

Claude Code is the harness under test, as in ``test_core_move``. The tests that
need a case-insensitive file system skip themselves on any other.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from sx.adapters import base as base_mod
from sx.adapters import claude as claude_mod
from sx.adapters.claude import ClaudeAdapter, encode_project_dir
from sx.adapters.relocation import rename_no_clobber, same_entry
from sx.model import MoveUnit
from sx.service import MoveService
from tests.conftest import write_compact_jsonl

TS = "2026-05-01T10:00:00.000Z"
A = "aaaaaaaa-0000-0000-0000-000000000001"
B = "aaaaaaaa-0000-0000-0000-000000000002"
OLD, NEW = Path("/x/Proj"), Path("/x/proj")


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point Claude at a sandbox and return its projects folder."""
    monkeypatch.setattr(claude_mod, "home", lambda: tmp_path)
    monkeypatch.setattr(claude_mod, "scratchpad_root", lambda: tmp_path / "scratch")
    projects = tmp_path / ".claude" / "projects"
    projects.mkdir(parents=True)
    return projects


@pytest.fixture()
def case_insensitive(tmp_path: Path) -> None:
    """Skip unless ``tmp_path`` is on a case-insensitive file system."""
    probe = tmp_path / "CaseProbe"
    probe.mkdir()
    try:
        insensitive = (tmp_path / "caseprobe").exists()
    finally:
        probe.rmdir()
    if not insensitive:
        pytest.skip("needs a case-insensitive file system")


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


def _service(tmp_path: Path, adapter: ClaudeAdapter | None = None) -> MoveService:
    """Return a move service over one Claude adapter, logging under ``tmp_path``."""
    return MoveService({"claude": adapter or ClaudeAdapter()}, log_path=tmp_path / "ops.log")


def _units(tmp_path: Path) -> list[dict]:
    """Return the op-log's per-unit records."""
    lines = (tmp_path / "ops.log").read_text(encoding="utf-8").splitlines()
    return [entry for entry in map(json.loads, lines) if entry["action"] == "move_unit"]


# --- the move ----------------------------------------------------------------------------


def test_a_case_only_rename_renames_the_folder_in_place(tmp_path, store, case_insensitive):
    """``/x/Proj -> /x/proj``: the folder takes the new spelling, its sessions the new path."""
    path = _session(store, str(OLD), A)
    service = _service(tmp_path)
    preview = service.move(service.plan(OLD, NEW), dry_run=True)["claude"]

    result = service.move(service.plan(OLD, NEW))["claude"]

    assert os.listdir(store) == ["-x-proj"]
    assert _cwds(store / "-x-proj" / path.name) == ["/x/proj", "/x/proj"]
    assert result.refused == {} and result.moved == [(store / "-x-Proj", store / "-x-proj")]
    assert result.moved == preview.moved
    assert [unit["status"] for unit in _units(tmp_path)] == ["done"]


def test_a_case_only_rename_and_its_inverse_restore_the_original(tmp_path, store, case_insensitive):
    """There and back, byte for byte, under the original spelling."""
    path = _session(store, str(OLD), A)
    before = path.read_bytes()
    service = _service(tmp_path)

    service.move(service.plan(OLD, NEW))
    assert os.listdir(store) == ["-x-proj"]
    service.move(service.plan(NEW, OLD))

    assert os.listdir(store) == ["-x-Proj"]
    assert path.read_bytes() == before


def test_a_stray_follows_its_project_into_the_renamed_folder(tmp_path, store, case_insensitive):
    """A session filed elsewhere lands in the folder under its new spelling."""
    _session(store, str(OLD), A)
    _session(store, str(OLD), B, folder="-a-elsewhere")

    result = _service(tmp_path).move(_service(tmp_path).plan(OLD, NEW))["claude"]

    assert sorted(os.listdir(store)) == ["-x-proj"]
    assert sorted(os.listdir(store / "-x-proj")) == [f"{A}.jsonl", f"{B}.jsonl"]
    assert _cwds(store / "-x-proj" / f"{B}.jsonl") == ["/x/proj", "/x/proj"]
    assert result.refused == {}


def test_a_session_whose_destination_is_itself_is_refused_and_kept(tmp_path, store, case_insensitive):
    """Its folder also holds another project's session, so it would move alone — onto itself."""
    path = _session(store, str(OLD), A)
    other = _session(store, "/x-Proj", B)  # encodes alike: the folder cannot move whole
    assert path.parent == other.parent
    before = path.read_bytes()

    plans = _service(tmp_path).plan(OLD, NEW)
    result = _service(tmp_path).move(plans)["claude"]

    assert path.read_bytes() == before and other.exists()
    assert "same file under another spelling" in plans["claude"].blocked[path]
    assert result.failed and result.rewritten == []


# --- the rename itself ----------------------------------------------------------------------


def test_rename_no_clobber_changes_only_the_case_of_a_folder(tmp_path, case_insensitive):
    """The destination exists only because it is the source: the rename goes ahead."""
    (tmp_path / "Dir").mkdir()
    (tmp_path / "Dir" / "kept.jsonl").write_text("{}\n", encoding="utf-8")

    assert rename_no_clobber(tmp_path / "Dir", tmp_path / "dir") is None

    assert os.listdir(tmp_path) == ["dir"]
    assert (tmp_path / "dir" / "kept.jsonl").read_text(encoding="utf-8") == "{}\n"


def test_rename_no_clobber_never_unlinks_a_file_reached_through_a_case_variant(
    tmp_path, case_insensitive
):
    """``Dir/f`` and ``dir/f`` are one file: it used to be unlinked as "our own link"."""
    (tmp_path / "Dir").mkdir()
    only = tmp_path / "Dir" / "f.jsonl"
    only.write_text("only copy\n", encoding="utf-8")

    reason = rename_no_clobber(only, tmp_path / "dir" / "f.jsonl")

    assert only.read_text(encoding="utf-8") == "only copy\n"
    assert reason is not None and "same file" in reason


def test_rename_no_clobber_never_unlinks_a_file_reached_through_a_linked_folder(tmp_path):
    """Any file system: ``link/f`` is ``real/f`` through a symbolic link to its folder."""
    real = tmp_path / "real"
    real.mkdir()
    only = real / "f.jsonl"
    only.write_text("only copy\n", encoding="utf-8")
    (tmp_path / "link").symlink_to(real, target_is_directory=True)

    reason = rename_no_clobber(only, tmp_path / "link" / "f.jsonl")

    assert only.read_text(encoding="utf-8") == "only copy\n"
    assert reason is not None and "same file" in reason


def test_two_hard_links_in_one_folder_are_two_entries_not_one(tmp_path):
    """Any file system: the folder lists both names, so neither is the other's spelling.

    Taken for one entry, the rename "changed only the case" and left both names
    in place — the session twice. As two entries, the destination is our own
    link from an interrupted move, and the rename finishes by unlinking the
    old name.
    """
    first = tmp_path / "a.jsonl"
    first.write_text("only copy\n", encoding="utf-8")
    second = tmp_path / "b.jsonl"
    os.link(first, second)

    assert not same_entry(first, second)
    assert rename_no_clobber(first, second) is None

    assert os.listdir(tmp_path) == ["b.jsonl"]
    assert second.read_text(encoding="utf-8") == "only copy\n"


def test_ctrl_c_just_after_a_case_only_rename_still_puts_it_back(
    tmp_path, store, case_insensitive, monkeypatch
):
    """The rename finished but was never recorded: it is found and undone like any other."""
    path = _session(store, str(OLD), A)
    before = path.read_bytes()
    real = base_mod.rename_no_clobber
    state = {"done": False}

    def renaming(src, dst):
        """Rename, then Ctrl-C once before the unit hears of it."""
        error = real(src, dst)
        if not state["done"]:
            state["done"] = True
            raise KeyboardInterrupt
        return error

    monkeypatch.setattr(base_mod, "rename_no_clobber", renaming)
    service = _service(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        service.move(service.plan(OLD, NEW))

    assert os.listdir(store) == ["-x-Proj"], "the folder has its old spelling back"
    assert path.read_bytes() == before
    (unit,) = _units(tmp_path)
    assert unit["moved"] == [] and "put back" in unit["skipped"][str(store / "-x-Proj")]

    service.move(service.plan(OLD, NEW))
    assert os.listdir(store) == ["-x-proj"]
    assert _cwds(store / "-x-proj" / path.name) == ["/x/proj", "/x/proj"]


# --- never recurse -----------------------------------------------------------------------------


class _AlwaysMerges(ClaudeAdapter):
    """A Claude adapter whose run-time check keeps calling for a merge, as a case variant did."""

    def _unit_check(self, unit, plan, simulation):
        """Ask for a merge for every folder unit that would otherwise run."""
        problem, merge = super()._unit_check(unit, plan, simulation)
        if unit.kind == MoveUnit.FOLDER and problem is None:
            return None, True
        return problem, merge


def test_a_merge_that_asks_to_merge_again_is_refused_not_recursed(tmp_path, store):
    """A destination that keeps "appearing" is refused once, cleanly, and logged as refused."""
    path = _session(store, "/x/old", A)
    before = path.read_bytes()
    service = _service(tmp_path, _AlwaysMerges())

    result = service.move(service.plan(Path("/x/old"), Path("/x/new")))["claude"]

    assert path.read_bytes() == before and not (store / "-x-new").exists()
    assert "changed while the move ran" in result.refused[store / "-x-old"]
    assert not any("RecursionError" in reason for reason in result.skipped.values())
    assert [unit["status"] for unit in _units(tmp_path)] == ["refused"]


class _BrokenCheck(ClaudeAdapter):
    """A Claude adapter whose liveness probe crashes."""

    def busy_transcript(self, path):
        """Fail the way a broken probe would."""
        raise RuntimeError("probe crashed")


def test_a_unit_whose_check_crashes_is_not_logged_as_nothing_to_do(tmp_path, store):
    """The per-unit record names the failure; it used to read "nothing to do"."""
    _session(store, "/x/old", A)
    service = _service(tmp_path, _BrokenCheck())

    result = service.move(service.plan(Path("/x/old"), Path("/x/new")))["claude"]

    (unit,) = _units(tmp_path)
    assert unit["status"] == "refused"
    assert any("probe crashed" in reason for reason in unit["skipped"].values())
    assert result.failed
