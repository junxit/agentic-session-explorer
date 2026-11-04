"""Regression tests for the second round of fixes to Oh My Pi's adapter.

Every store is synthetic, under ``tmp_path``; ``omp`` is never run. The
builders and the autouse sandbox come from :mod:`tests.test_omp_adapter`.
"""

from __future__ import annotations

import pytest

from sx.adapters.omp import OmpAdapter
from sx.adapters.pi import TranscriptKind
from sx.model import Orphan, OrphanKind
from tests import test_omp_adapter as base

sandbox = base.sandbox
old = base.old
new = base.new


# --- 9: only the first line itself makes a fragment ----------------------------------


@pytest.mark.parametrize("first", [b"\n", b"{not json\n", b"null\n"], ids=["blank", "bad", "null"])
def test_an_entry_after_a_first_line_that_is_not_one_is_unrecognized(sandbox, old, first):
    """omp's writer leaves debris that starts with an entry, on its first line."""
    path = base._bucket(old) / base._name("F1")
    path.parent.mkdir(parents=True)
    path.write_bytes(first + base._compact(base._user("u1", None, "x")) + b"\n")
    base._age_tree(path.parent, base.DAY)
    adapter = OmpAdapter()

    assert adapter.transcript_kind(path) is TranscriptKind.UNRECOGNIZED
    assert adapter.find_orphans() == []


def test_debris_named_for_a_session_is_still_a_fragment(sandbox, old):
    """The rule keeps what omp's writer really leaves: an entry first, in a session's name."""
    path = base._write(base._bucket(old) / base._name("F1"), [base._user("u1", None, "x")])
    base._age_tree(path.parent, base.DAY)
    adapter = OmpAdapter()

    assert adapter.transcript_kind(path) is TranscriptKind.FRAGMENT
    assert [o.paths for o in adapter.find_orphans() if o.kind is OrphanKind.STRAY_TEMP] == [[path]]


# --- 2: a session in a format sx does not know keeps its bucket ------------------------


def _gone_session(sandbox, sid: str, *, slot: bytes, **extra):
    """Write a session whose project directory does not exist."""
    gone = sandbox / "gone" / sid.lower()
    path = base._bucket(gone) / base._name(sid)
    base._write(
        path, [base._header(sid, str(gone), **extra), base._user("u1", None, "x")], slot=slot
    )
    base._age_tree(path.parent, base.DAY)
    return path


@pytest.mark.parametrize(
    ("sid", "slot", "extra", "why"),
    [
        ("V4", base._slot(), {"version": 4}, "version 4"),
        ("SLOT", base._slot(size=300), {}, "title slot is not 256 bytes"),
        (
            "BADSLOT",
            b'{"type":"title","v":2,"title":"","updatedAt":"x","pad":""}\n',
            {},
            "does not begin the way Oh My Pi writes",
        ),
    ],
    ids=["version", "slot-size", "slot-version"],
)
def test_a_dead_projects_bucket_holding_an_unknown_format_is_kept(sandbox, sid, slot, extra, why):
    """Refused for delete and move alike, so never removed by cleanup either."""
    path = _gone_session(sandbox, sid, slot=slot, **extra)
    adapter = OmpAdapter()
    (session,) = adapter.discover()
    assert why in adapter.delete(session, dry_run=True).refused[path]

    assert adapter.find_orphans() == []

    by_hand = Orphan("omp", OrphanKind.DEAD_PROJECT, [path.parent], "project directory gone")
    result = adapter.delete_orphan(by_hand)
    assert result.failed and why in result.refused[path.parent]
    assert path.exists()


# --- 3: memory folders in a chained move --------------------------------------------------


def _memory(sandbox, project, files: dict[str, str]):
    """Write a project's memory folder, idle, and return it."""
    from sx.adapters.omp import memory_folder_name

    folder = base._memories(sandbox) / memory_folder_name(str(project))
    folder.mkdir(parents=True)
    for name, text in files.items():
        (folder / name).write_text(text, encoding="utf-8")
    base._age_tree(folder, base.HOUR)
    return folder


def _contents(folder):
    """Return a folder's files and their text, or None when it is gone."""
    if not folder.exists():
        return None
    return {path.name: path.read_text(encoding="utf-8") for path in sorted(folder.iterdir())}


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "real"])
def test_memory_folders_follow_a_chained_move_whole(sandbox, dry_run):
    """``app`` -> ``app-v1`` while ``app/v1`` has memory: both names are ``--…-app-v1--``.

    ``app/v1``'s folder must move on first, so ``app``'s takes its name whole;
    merged into it instead, ``app``'s memory used to be split in two.
    """
    old, new = sandbox / "x" / "app", sandbox / "x" / "app-v1"
    sub = old / "v1"
    base._session(old, "O1")
    base._session(sub, "O2")
    app = _memory(sandbox, old, {"memory_summary.md": "APP summary", "a.md": "APP a"})
    v1 = _memory(sandbox, sub, {"memory_summary.md": "V1 summary", "b.md": "V1 b"})
    v1_after = _memory_name(sandbox, new / "v1")
    adapter = OmpAdapter()

    plan, result = base._move(adapter, old, new, dry_run=dry_run)

    assert not plan.blocked and not result.refused
    assert (app, v1) in plan.relocations and (v1, v1_after) in plan.relocations
    if dry_run:
        return
    assert _contents(app) is None
    assert _contents(v1) == {"a.md": "APP a", "memory_summary.md": "APP summary"}
    assert _contents(v1_after) == {"b.md": "V1 b", "memory_summary.md": "V1 summary"}


def _memory_name(sandbox, project):
    """Return where omp keeps ``project``'s memory."""
    from sx.adapters.omp import memory_folder_name

    return base._memories(sandbox) / memory_folder_name(str(project))


# --- a case-only rename of a bucket that holds lock files ------------------------------------


@pytest.fixture
def case_insensitive(tmp_path):
    """Skip unless ``tmp_path`` is on a case-insensitive file system."""
    probe = tmp_path / "CaseProbe"
    probe.mkdir()
    try:
        insensitive = (tmp_path / "caseprobe").exists()
    finally:
        probe.rmdir()
    if not insensitive:
        pytest.skip("needs a case-insensitive file system")


def test_a_case_only_rename_of_a_bucket_holding_lock_files_renames_it_whole(
    sandbox, case_insensitive
):
    """``Proj`` -> ``proj``: the bucket is its own destination, so it is renamed, locks and all.

    A bucket with lock files otherwise moves entry by entry around them, which
    here would move each file onto itself.
    """
    import os

    old, new = sandbox / "work" / "Proj", sandbox / "work" / "proj"
    path = base._session(old, "O1")
    lock = path.parent / "x.lock"
    lock.write_text("held\n", encoding="utf-8")
    base._age(lock)
    adapter = OmpAdapter()
    renamed = adapter.sessions_dir() / adapter.folder_for(str(new))
    assert renamed != path.parent and renamed.name.lower() == path.parent.name.lower()

    _, result = base._move(adapter, old, new)

    assert result.moved == [(path.parent, renamed)] and not result.refused
    assert os.listdir(adapter.sessions_dir()) == [renamed.name]
    assert sorted(os.listdir(renamed)) == sorted([path.name, lock.name])
    (session,) = adapter.discover()
    assert session.project_path == str(new)


def test_a_symlinked_bucket_is_never_offered_as_leftovers(sandbox, old, tmp_path):
    """P25: its contents lie outside the store; omp has no idle window to catch it otherwise."""
    outside = tmp_path / "outside"
    outside.mkdir()
    base._age(outside, seconds=base.DAY)
    bucket = base._bucket(old)
    bucket.parent.mkdir(parents=True)
    bucket.symlink_to(outside, target_is_directory=True)
    assert OmpAdapter().project_leftovers(str(old)) is None
