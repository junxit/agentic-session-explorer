"""Unit tests for the shared filesystem guards in :mod:`sx.fsguard`.

``project_state`` decides whether a project is gone — the question every dead
project orphan hangs on — so each way of failing to look must come out as
"unknown", never "missing". Everything here runs under ``tmp_path``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sx.fsguard import MISSING, PRESENT, UNKNOWN, project_missing, project_state, survey

needs_permissions = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root traverses directories whatever their mode",
)


@pytest.fixture()
def locked(tmp_path: Path):
    """Yield a project directory inside a parent nobody may traverse."""
    guarded = tmp_path / "guarded"
    project = guarded / "app"
    project.mkdir(parents=True)
    os.chmod(guarded, 0o000)
    try:
        yield project
    finally:
        os.chmod(guarded, 0o755)


def test_a_directory_that_is_there_is_present(tmp_path):
    """``lstat`` succeeds: present."""
    assert project_state(tmp_path) == PRESENT
    assert project_state(str(tmp_path)) == PRESENT


def test_a_deleted_directory_is_missing(tmp_path):
    """Its own absence, under a parent that answers: missing."""
    gone = tmp_path / "gone"
    assert project_state(gone) == MISSING
    assert project_missing(gone) is True


def test_a_path_running_through_a_file_is_missing(tmp_path):
    """``ENOTDIR``: a directory cannot exist beneath a file."""
    file = tmp_path / "notes.txt"
    file.write_text("x", encoding="utf-8")
    assert project_state(file / "proj") == MISSING


@needs_permissions
def test_an_untraversable_parent_makes_the_project_unknown(locked):
    """``Path.exists()`` says False here; the project is very much alive."""
    assert locked.exists() is False  # the trap the helper exists to avoid
    assert project_state(locked) == UNKNOWN
    assert project_missing(locked) is False


def test_an_unmounted_volume_is_unknown():
    """A project on an unplugged drive is unavailable, not deleted."""
    assert project_state("/Volumes/DefinitelyNotMounted42/proj") == UNKNOWN
    assert project_state("/mnt/definitely-not-mounted-42/proj") == UNKNOWN
    assert project_state("/media/nobody-42/USB/proj") == UNKNOWN


def test_a_dangling_link_on_the_way_is_unknown(tmp_path):
    """``~/work -> /Volumes/X/work`` with the drive away: unknown, not missing."""
    link = tmp_path / "work"
    link.symlink_to(tmp_path / "away" / "work")
    assert project_state(link / "proj") == UNKNOWN


def test_a_dangling_link_as_the_project_itself_is_present(tmp_path):
    """``lstat`` sees the link; what it points at is not this check's business."""
    link = tmp_path / "proj"
    link.symlink_to(tmp_path / "nowhere")
    assert project_state(link) == PRESENT


@pytest.mark.parametrize("value", [None, "", "relative/proj", "./proj", "/bad\x00path"])
def test_values_that_are_not_absolute_paths_are_unknown(value):
    """Nothing can be judged from them, so nothing is called gone."""
    assert project_state(value) == UNKNOWN


def test_a_survey_changes_when_an_entry_is_added_and_not_otherwise(tmp_path):
    """The fingerprint is stable for an unchanged tree and moves with any write."""
    folder = tmp_path / "folder"
    (folder / "sub").mkdir(parents=True)
    (folder / "sub" / "a.txt").write_text("a", encoding="utf-8")
    first = survey(folder)
    assert survey(folder) == first
    (folder / "sub" / "b.jsonl").write_text("{}\n", encoding="utf-8")
    second = survey(folder)
    assert second.fingerprint != first.fingerprint
    assert second.newest_ns >= first.newest_ns


def test_a_survey_never_follows_a_link(tmp_path):
    """A directory swapped for a link to an identical directory is a change."""
    real = tmp_path / "real"
    real.mkdir()
    (real / "x.txt").write_text("x", encoding="utf-8")
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "x.txt").write_text("x", encoding="utf-8")
    before = survey(folder).fingerprint
    (folder / "x.txt").unlink()
    folder.rmdir()
    folder.symlink_to(real)
    assert survey(folder).fingerprint != before


def test_a_survey_of_nothing_is_still_a_survey(tmp_path):
    """An absent path has a fingerprint of its own and no modification time."""
    result = survey(tmp_path / "absent")
    assert result.newest_ns is None and result.fingerprint
