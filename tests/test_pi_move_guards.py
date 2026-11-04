"""Guards a Pi move keeps when it finishes a half-done link, or meets a linked folder.

A transcript a move cut short between linking it at its destination and
unlinking its old name is finished by the next run — but only as safely as
any relocation: both names inside the store, the destination folder a real
directory, the transcript neither refused nor in a format ``sx`` does not
know. One file reached under two spellings of a case-insensitive file system
is not a half-done link. And no unit renames a project folder that has become
a symbolic link, even one with nothing to rewrite.

Every store is synthetic, under ``tmp_path`` (the builders and the autouse
``home`` fixture come from :mod:`tests.test_pi_fixes`); ``pi`` is never run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sx.adapters.pi import PiAdapter, encode_session_folder
from sx.model import MovePlan, MoveUnit
from tests.test_pi_fixes import (
    _age_tree,
    _folder,
    _header,
    _name,
    _service,
    _session,
    _user,
    _write,
    home,
)

__all__ = ["home"]  # the autouse sandbox

OLD, NEW = Path("/x/old"), Path("/x/new")


def _half_moved(path: Path) -> Path:
    """Link ``path`` into /x/new's folder, as a move cut short between link and unlink leaves it."""
    linked = _folder(NEW) / path.name
    linked.parent.mkdir(parents=True, exist_ok=True)
    os.link(path, linked)
    return linked


def _as_version(path: Path, version: int) -> None:
    """Rewrite a transcript's header in place — the same file, under every name it has."""
    lines = path.read_bytes().splitlines(keepends=True)
    header = {**json.loads(lines[0]), "version": version}
    with path.open("r+b") as fh:
        fh.write(json.dumps(header, separators=(",", ":")).encode() + b"\n" + b"".join(lines[1:]))
        fh.truncate()


@pytest.mark.parametrize("where", ["outside-the-store", "inside-the-store"])
def test_a_half_done_link_is_not_finished_into_a_folder_that_became_a_link(home, tmp_path, where):
    """Planned to finish A1; then /x/new's folder became a link: A1's old name stays."""
    path = _session(OLD, "A1")
    _session(NEW, "N1")
    path.write_bytes(path.read_bytes().replace(b'"cwd":"/x/old"', b'"cwd":"/x/new"', 1))
    _half_moved(path)
    _age_tree(PiAdapter().sessions_dir(), 3600)
    adapter = PiAdapter()
    service = _service(home)
    plan = service.plan(OLD, NEW)["pi"]
    assert plan.relocations == [(path, _folder(NEW) / path.name)]
    folder = _folder(NEW)
    target = tmp_path / "elsewhere" if where == "outside-the-store" else folder.parent / "real"
    folder.rename(target)
    folder.symlink_to(target, target_is_directory=True)

    result = adapter.move(plan)

    assert path.exists() and os.stat(path).st_nlink == 2
    assert path in result.refused and not result.moved


def test_a_half_done_link_of_a_file_sx_may_not_rewrite_is_not_finished(home):
    """A transcript traveling with its folder, half moved, then rewritten in an unknown version."""
    _session(OLD, "S1")
    _session(NEW, "N1")
    rider = _session("/elsewhere", "X1")  # recorded elsewhere first: it travels with the folder
    rider = rider.rename(_folder(OLD) / rider.name)
    _half_moved(rider)
    _age_tree(PiAdapter().sessions_dir(), 3600)
    adapter = PiAdapter()
    plan = _service(home).plan(OLD, NEW)["pi"]
    assert (rider, _folder(NEW) / rider.name) in plan.relocations
    _as_version(rider, 4)

    result = adapter.move(plan)

    assert rider.exists() and "not safe to rewrite" in result.refused[rider]
    assert result.moved == [(_folder(OLD) / _name("S1"), _folder(NEW) / _name("S1"))]


@pytest.fixture
def case_insensitive(tmp_path: Path) -> None:
    """Skip unless ``tmp_path`` is on a case-insensitive file system."""
    probe = tmp_path / "CaseProbe"
    probe.touch()
    try:
        insensitive = (tmp_path / "caseprobe").exists()
    finally:
        probe.unlink()
    if not insensitive:
        pytest.skip("needs a case-insensitive file system")


def test_one_file_under_two_spellings_is_no_half_done_link(home, case_insensitive):
    """/x/App to /x/app: A1 moves on its own, and its destination is A1 itself, not a link.

    The folder also holds /x-App's session, which encodes alike, so A1 moves
    on its own. It is refused, and a dry run does not say it moved.
    """
    old, new = Path("/x/App"), Path("/x/app")
    path = _session(old, "A1")
    _session("/x-App", "B1")
    _age_tree(PiAdapter().sessions_dir(), 3600)
    service = _service(home)

    plan = service.plan(old, new)["pi"]
    dry = service.move({"pi": plan}, dry_run=True)["pi"]

    assert not plan.relocations and "same file under another spelling" in plan.blocked[path]
    assert not dry.moved and path.exists()


def test_a_unit_never_renames_a_project_folder_that_became_a_link(home):
    """Even a unit with nothing to rewrite: the link would carry its target off under a new name."""
    sessions = PiAdapter().sessions_dir()
    real = sessions / "real-folder"
    _write(real / _name("S1"), [_header("S1", "/work/app"), _user("u1", None, "hi")])
    linked = sessions / encode_session_folder("/work/app")
    linked.symlink_to(real, target_is_directory=True)
    _age_tree(sessions, 3600)
    pair = (linked, sessions / encode_session_folder("/work/renamed"))
    plan = MovePlan(
        harness="pi",
        old=Path("/work/app"),
        new=Path("/work/renamed"),
        relocations=[pair],
        units=[MoveUnit(MoveUnit.FOLDER, relocations=[pair])],
    )

    result = PiAdapter().move(plan)

    assert linked.is_symlink() and not os.path.lexists(pair[1])
    assert "symbolic link" in result.refused[linked]
