"""Deleting Oh My Pi sessions and cleaning up after them: D2, D3/G8, D7, D8, D12.

Every store is synthetic, under ``tmp_path``; ``omp`` is never run. A running
omp is played by this process holding a ``flock`` on the lease file. The
builders and the autouse sandbox come from :mod:`tests.test_omp_adapter`.
"""

from __future__ import annotations

import errno
import json
import os
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from sx.adapters.omp import OmpAdapter
from sx.model import OrphanKind
from sx.tui.app import SxApp
from sx.tui.screens import ConfirmDeleteScreen
from sx.util import is_within
from tests import test_omp_adapter as base
from tests.test_orphan_screen import _open_orphans, _type, _wire

sandbox = base.sandbox
old = base.old
new = base.new

HOUR = base.HOUR


def _with_artifacts(path: Path) -> Path:
    """Give a transcript an artifacts directory holding tool output."""
    artifacts = path.parent / path.stem
    artifacts.mkdir()
    (artifacts / "out.txt").write_text("tool output the session references", encoding="utf-8")
    return artifacts


# --- D2: files sx does not recognize ---------------------------------------------------


def _unrecognized(bucket: Path) -> dict[str, Path]:
    """Write the header-less files omp does not leave as debris, one of each kind."""
    slot_only = base._write(bucket / base._name("A"), [base._user("u1", None, "hi")], slot=base._slot())
    # A header line JSON cannot parse (a raw control character), which omp's
    # session list still lists by scanning its text — but will not resume.
    torn = bucket / base._name("B")
    torn.write_bytes(
        b'{"type":"session","version":3,"id":"B","cwd":"/x","title":"a\x01b"}\n'
        + base._compact(base._user("u1", None, "hi"))
        + b"\n"
    )
    long_line = base._write(
        bucket / base._name("C"), [base._user("u1", None, "x" * (1024 * 1024 + 10))], slot=None
    )
    for path in (slot_only, torn, long_line):
        base._age(path)
    base._age(bucket)
    return {"slot only": slot_only, "text-scanned header": torn, "first line too long": long_line}


def test_files_sx_does_not_recognize_are_listed_and_never_deleted_moved_or_offered(
    sandbox, old, new
):
    """A slot with no header, a header omp could only text-scan: listed, browsable, kept."""
    keep = base._session(old)
    files = _unrecognized(keep.parent)
    adapter = OmpAdapter()
    sessions = {s.paths[0]: s for s in adapter.discover()}

    for label, path in files.items():
        session = sessions[path]
        before = path.read_bytes()
        assert session.title == f"(unrecognized format) {path.name}", label
        assert session.extra.get("unrecognized"), label
        assert [m.text for m in adapter.load_transcript(session).current], label
        result = adapter.delete(session)
        assert result.failed and "does not recognize" in result.refused[path], label
        plan = adapter.plan_move([session], old, new)
        assert path in plan.blocked and not plan.relocations, label
        assert path.read_bytes() == before, label
    assert [o for o in adapter.find_orphans() if set(o.paths) & set(files.values())] == []
    assert any("do not look like Oh My Pi sessions" in w for w in adapter.store_warnings())

    _plan, result = base._move(adapter, old, new)

    assert all(path.exists() for path in files.values()) and not keep.exists()
    assert result.refused == {}


# --- D7: a transcript that is missing ----------------------------------------------------


def test_a_session_omp_is_rewriting_is_left_whole(sandbox, old):
    """omp's rewrite can move ``S.jsonl`` aside while it holds the lease and publish lock."""
    path = base._session(old)
    artifacts = _with_artifacts(path)
    adapter = OmpAdapter()
    session = base._only(adapter)
    aside = path.parent / f"{path.name}.0123456789abcdef.bak"
    temp = path.parent / f".{path.name}.0123456789abcdef.tmp"

    with base._flocked(base._lease_dir(sandbox) / "S1.lock"):
        os.rename(path, aside)  # omp's EPERM fallback, mid-flight
        temp.write_bytes(aside.read_bytes())
        result = adapter.delete(session)

    assert result.removed == [] and "has this session open" in result.refused[path]
    assert aside.exists() and temp.exists() and (artifacts / "out.txt").exists()


def test_the_publish_lock_is_checked_before_the_transcript_is_looked_for(sandbox, old):
    """While omp publishes, the transcript may be missing for a moment."""
    path = base._session(old)
    artifacts = _with_artifacts(path)
    adapter = OmpAdapter()
    session = base._only(adapter)
    (path.parent / f".{path.name}.lock").write_text("123:456\n", encoding="utf-8")
    path.unlink()

    result = adapter.delete(session)

    assert result.removed == [] and "is writing this session" in result.refused[path]
    assert artifacts.is_dir()


def test_a_missing_transcript_deletes_nothing_and_says_so(sandbox, old):
    """Its artifacts and backups stay: omp would restore the session from a backup."""
    path = base._session(old)
    artifacts = _with_artifacts(path)
    backup = path.parent / f"{path.name}.0123456789abcdef.bak"
    backup.write_bytes(path.read_bytes())
    adapter = OmpAdapter()
    session = base._only(adapter)
    path.unlink()

    preview = adapter.delete(session, dry_run=True)
    result = adapter.delete(session)

    assert preview.removed == [] and result.removed == [] and result.failed
    assert "transcript is missing" in result.refused[path]
    assert artifacts.is_dir() and backup.exists()


def test_a_session_already_gone_is_reported_gone(sandbox, old):
    """Nothing of it left at all: not a refusal."""
    path = base._session(old)
    adapter = OmpAdapter()
    session = base._only(adapter)
    path.unlink()
    result = adapter.delete(session)
    assert result.skipped == {path: "does not exist"} and not result.failed


# --- D8: all or nothing -------------------------------------------------------------------


def _fail_unlink(monkeypatch: pytest.MonkeyPatch, victim: Path) -> None:
    """Make removing ``victim`` fail the way an immutable file does."""
    real = Path.unlink

    def unlink(self: Path, missing_ok: bool = False) -> None:
        """Refuse the victim; remove anything else."""
        if self == victim:
            raise PermissionError(errno.EPERM, "Operation not permitted", str(self))
        real(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)


def test_a_backup_that_cannot_be_removed_keeps_the_rest_of_the_session(sandbox, old, monkeypatch):
    """Backups go first; the first one that fails stops the delete before the transcript."""
    path = base._session(old)
    artifacts = _with_artifacts(path)
    first = path.parent / f"{path.name}.0aaaaaaaaaaaaaaa.bak"
    second = path.parent / f"{path.name}.1bbbbbbbbbbbbbbb.bak"
    for backup in (first, second):
        backup.write_bytes(path.read_bytes())
    adapter = OmpAdapter()
    session = base._only(adapter)
    _fail_unlink(monkeypatch, second)

    result = adapter.delete(session)

    assert result.removed == [first]
    assert path.exists() and (artifacts / "out.txt").exists() and second.exists()
    assert "Operation not permitted" in result.refused[second]
    for kept in (path, artifacts):
        assert "could not be removed" in result.refused[kept]
        assert "would restore the session" in result.refused[kept]


def test_a_transcript_that_cannot_be_removed_keeps_its_artifacts(sandbox, old, monkeypatch):
    """The session stays usable: its tool output is not taken from under it."""
    path = base._session(old)
    artifacts = _with_artifacts(path)
    adapter = OmpAdapter()
    session = base._only(adapter)
    _fail_unlink(monkeypatch, path)

    result = adapter.delete(session)

    assert result.removed == [] and result.failed
    assert path.exists() and (artifacts / "out.txt").exists()
    assert "its artifacts stay with it" in result.refused[artifacts]


@pytest.mark.skipif(not hasattr(os, "chflags"), reason="needs BSD file flags")
def test_an_immutable_backup_keeps_the_whole_session(sandbox, old):
    """The kernel refuses the unlink; nothing of the session is removed."""
    path = base._session(old)
    artifacts = _with_artifacts(path)
    backup = path.parent / f"{path.name}.1234567890.bak"
    backup.write_bytes(path.read_bytes())
    adapter = OmpAdapter()
    session = base._only(adapter)
    os.chflags(backup, stat.UF_IMMUTABLE)
    try:
        result = adapter.delete(session)
    finally:
        os.chflags(backup, 0)

    assert result.removed == [] and result.failed
    assert path.exists() and artifacts.is_dir() and backup.exists()
    assert "would restore the session" in result.refused[path]


# --- D12: every backup omp's own delete takes ------------------------------------------


def test_delete_takes_every_backup_omps_delete_takes(sandbox, old):
    """Every ``S.jsonl.*.bak`` goes, restorable or not; what omp's delete leaves stays."""
    path = base._session(old)
    bucket = path.parent
    taken = [
        bucket / f"{path.name}.0123456789abcdef.bak",
        bucket / f"{path.name}.a.b.bak",
        bucket / f"{path.name}.bak",
        bucket / f"{path.name}.unpack-0123456789abcdef.bak",
    ]
    left = [
        bucket / f"x{path.name}.0123.bak",
        bucket / f"{base._name('S2')}.0123.bak",
    ]
    for item in taken + left:
        item.write_bytes(b"x")
    unpacked = bucket / f"{path.stem}.unpack-0123456789abcdef.bak"
    (unpacked / "logs").mkdir(parents=True)
    adapter = OmpAdapter()

    result = adapter.delete(base._only(adapter))

    assert sorted(result.removed) == sorted([*taken, path]) and result.refused == {}
    assert all(item.exists() for item in left) and unpacked.is_dir()


def test_a_dead_bucket_takes_its_sessions_own_backups_with_it(sandbox, tmp_path):
    """They belong to a session that goes with the bucket, as a session delete takes them."""
    gone = tmp_path / "sx-gone" / "proj"
    path = base._session(gone, "D")
    backup = path.parent / f"{path.name}.a.b.bak"
    backup.write_bytes(b"older copy")
    base._age_tree(path.parent, HOUR)
    adapter = OmpAdapter()

    (orphan,) = adapter.find_orphans()

    assert (orphan.kind, orphan.paths) == (OrphanKind.DEAD_PROJECT, [path.parent])
    assert adapter.delete_orphan(orphan).removed == [path.parent]


# --- D3 / G8: a dead project's bucket is checked again -------------------------------------


@pytest.fixture
def dead(sandbox: Path, tmp_path: Path):
    """Return a dead project's bucket offered for cleanup: ``(project, transcript, orphan)``."""
    gone = tmp_path / "sx-gone" / "proj"
    path = base._session(gone, "D")
    base._age_tree(path.parent, HOUR)
    adapter = OmpAdapter()
    (orphan,) = adapter.find_orphans()
    assert (orphan.kind, orphan.paths) == (OrphanKind.DEAD_PROJECT, [path.parent])
    return gone, path, orphan


def test_a_dead_bucket_whose_session_omp_now_holds_is_refused(sandbox, dead):
    """Nothing in the bucket changed — only the lease, which lives outside it."""
    _gone, path, orphan = dead
    with base._flocked(base._lease_dir(sandbox) / "D.lock"):
        result = OmpAdapter().delete_orphan(orphan)
    assert result.failed and "has this session open" in result.refused[path.parent]
    assert path.exists()


def test_a_dead_bucket_whose_project_came_back_is_refused(sandbox, dead):
    """The project is re-cloned: its sessions are no longer orphans."""
    gone, path, orphan = dead
    gone.mkdir(parents=True)
    result = OmpAdapter().delete_orphan(orphan)
    assert result.failed and "project directory is there" in result.refused[path.parent]
    assert path.exists()


@pytest.mark.asyncio
async def test_the_orphan_screen_checks_a_bucket_emptied_around_its_lock_files_again(
    sandbox, tmp_path, monkeypatch
):
    """The project comes back while the phrase is typed: nothing in the bucket is deleted.

    The confirmation lists what the delete takes from the bucket — the
    transcript and its artifacts, not the lock file. The bucket must still
    reach the delete whole: handed those two alone, the delete never asked
    whether the bucket's project was still gone, and took the session.
    """
    monkeypatch.setattr("sx.tui.app.build_registry", lambda: ([], []))
    gone = tmp_path / "sx-gone" / "proj"
    path = base._session(gone, "D")
    artifacts = _with_artifacts(path)
    (path.parent / f".{path.name}.lock.os").touch()
    base._age_tree(path.parent, HOUR)
    adapter = OmpAdapter()
    assert all(is_within(root, [tmp_path]) for root in adapter.store_roots())
    log = tmp_path / "ops.log"

    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        _wire(app, None, [], log, adapter=adapter)
        screen = await _open_orphans(app, pilot)
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        assert isinstance(app.screen, ConfirmDeleteScreen)
        assert "1 lock file(s) stay" in app.screen._preview.plain
        await _type(pilot, "DELETE 2")
        gone.mkdir(parents=True)
        await pilot.click("#confirm")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.screen is screen

    assert path.exists() and (artifacts / "out.txt").exists()
    assert len(screen._orphans) == 1
    entry = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert entry["removed"] == []
    assert "project directory is there" in entry["skipped"][str(path.parent)]


def test_a_dead_bucket_omp_wrote_a_new_session_in_is_refused(sandbox, dead):
    """A new transcript, written seconds ago: the bucket is not what was offered."""
    gone, path, orphan = dead
    gone.mkdir(parents=True)
    fresh = base._write(path.parent / base._name("NEW"), [base._header("NEW", str(gone))], slot=base._slot())
    adapter = OmpAdapter()

    result = adapter.delete_orphan(orphan)
    unprinted = adapter.delete_orphan(replace(orphan, fingerprints=None))

    assert result.failed and "changed since it was listed" in result.refused[path.parent]
    assert unprinted.failed and fresh.exists() and path.exists()


def test_a_dead_bucket_written_in_the_last_90_seconds_is_refused(sandbox, tmp_path):
    """Listed while omp was still writing it: refused until it has been quiet a while."""
    gone = tmp_path / "sx-gone" / "proj"
    path = base._session(gone, "D")
    os.utime(path)  # written just now
    adapter = OmpAdapter()
    (orphan,) = adapter.find_orphans()
    result = adapter.delete_orphan(orphan)
    assert result.failed and "in the last 90 seconds" in result.refused[path.parent]


def test_an_artifacts_directory_whose_session_omp_holds_keeps_its_bucket(sandbox, old):
    """An artifacts directory without its transcript names a session: its lease is asked."""
    bucket = base._bucket(old)
    draft = base._stale_artifacts(bucket, "D1")
    base._age(bucket)
    adapter = OmpAdapter()
    (orphan,) = adapter.find_orphans()
    assert orphan.paths == [bucket]
    with base._flocked(base._lease_dir(sandbox) / "D1.lock"):
        result = adapter.delete_orphan(orphan)
    assert result.failed and "has this session open" in result.refused[bucket]
    assert draft.is_dir()


@pytest.mark.parametrize("records", [[], [base._user("u1", None, "x")]], ids=["empty", "entry"])
def test_a_fragment_without_an_omp_name_is_never_offered(sandbox, old, records):
    """No session id in its name: no lease can vouch for it, so it is not omp's debris.

    It used to be offered, and then every delete refused it. Now it is listed
    as a file sx does not recognize, and nothing destroys it.
    """
    bucket = base._bucket(old)
    fragment = base._write(bucket / "notes.jsonl", records, slot=None)
    base._age_tree(bucket, base.DAY)
    adapter = OmpAdapter()

    assert adapter.find_orphans() == []
    (listed,) = adapter.discover()
    assert listed.paths == [fragment] and listed.title.startswith("(unrecognized format)")
    assert "never deletes or moves" in adapter.delete(listed).refused[fragment]
    assert any("do not look like Oh My Pi sessions" in w for w in adapter.store_warnings())
    assert fragment.exists()
