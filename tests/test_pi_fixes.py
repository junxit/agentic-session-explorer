"""Regression tests for the second round of fixes to Pi's adapter, on SYNTHETIC stores.

Every store lives under ``tmp_path``: the module's ``home`` is redirected and
the suite's autouse fixtures scrub the ``PI_*`` variables, so the real ``~/.pi``
is never read. ``pi`` is never run.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from sx.adapters import pi as pi_module
from sx.adapters.pi import PiAdapter, TranscriptKind, encode_session_folder
from sx.model import Orphan, OrphanKind

TS = "2026-05-01T10:00:00.000Z"
DAY = 24 * 3600


# --- fixtures and builders ---------------------------------------------------------


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point Pi's home at a sandbox directory and return it."""
    sandbox = tmp_path / "home"
    sandbox.mkdir()
    monkeypatch.setattr(pi_module, "home", lambda: sandbox)
    return sandbox


def _compact(record: object) -> bytes:
    """Serialize a record as ``JSON.stringify`` does."""
    return json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _header(sid: str, cwd: str, **extra) -> dict:
    """Build a version 3 session header."""
    return {"type": "session", "version": 3, "id": sid, "timestamp": TS, "cwd": cwd, **extra}


def _user(entry_id: str, parent: str | None, text: str) -> dict:
    """Build a user message entry."""
    return {
        "type": "message",
        "id": entry_id,
        "parentId": parent,
        "timestamp": TS,
        "message": {"role": "user", "content": text, "timestamp": 1},
    }


def _assistant(entry_id: str, parent: str, text: str) -> dict:
    """Build an assistant message entry."""
    return {
        "type": "message",
        "id": entry_id,
        "parentId": parent,
        "timestamp": TS,
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def _name(sid: str) -> str:
    """Return the file name Pi gives a session."""
    return f"2026-05-01T10-00-00-000Z_{sid}.jsonl"


def _folder(project: Path | str) -> Path:
    """Return the folder Pi keeps ``project``'s sessions in."""
    return PiAdapter().sessions_dir() / encode_session_folder(str(project))


def _write(path: Path, records: list[dict], *, first: bytes = b"") -> Path:
    """Write ``first`` (raw bytes), then one compact JSON line per record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(first + b"".join(_compact(record) + b"\n" for record in records))
    return path


def _session(project: Path | str, sid: str = "S1", **extra) -> Path:
    """Write a session for ``project`` into its folder."""
    records = [_header(sid, str(project), **extra), _user("u1", None, "hello")]
    return _write(_folder(project) / _name(sid), records)


def _age_tree(root: Path, seconds: float = 3 * DAY) -> None:
    """Backdate a tree, deepest entries first, so nothing in it counts as in use."""
    past = time.time() - seconds
    for top, dirs, files in os.walk(root, topdown=False):
        for name in (*files, *dirs):
            os.utime(Path(top, name), (past, past), follow_symlinks=False)
    os.utime(root, (past, past), follow_symlinks=False)


# --- 9: a damaged header line is not debris ------------------------------------------


def test_a_session_whose_header_line_is_damaged_is_unrecognized_not_a_fragment(home):
    """Its conversation is intact after the header: it is listed, and never offered or deleted."""
    project = home / "work" / "app"
    project.mkdir(parents=True)
    path = _folder(project) / _name("S1")
    header = _compact(_header("S1", str(project)))
    body = [_user("u1", None, "a long conversation"), _assistant("a1", "u1", "answer " * 200)]
    _write(path, body, first=header[:20] + b"\x00\x00garbage" + header[20:] + b"\n")
    _age_tree(PiAdapter().sessions_dir())
    adapter = PiAdapter()

    assert adapter.transcript_kind(path) is TranscriptKind.UNRECOGNIZED
    assert adapter.find_orphans() == []
    (listed,) = adapter.discover()
    assert listed.title.startswith("(unrecognized format)")
    assert "never deletes or moves" in adapter.delete(listed).refused[path]
    assert path.exists()


@pytest.mark.parametrize(
    "first",
    [b"\n", b"null\n", b"{not json\n", b"\xff\xfe" + b'{"type":"message"}\n'],
    ids=["blank", "null", "malformed", "not-utf8"],
)
def test_only_the_first_line_itself_can_make_a_file_debris(home, first):
    """An entry after a blank, a null or a damaged first line does not make a fragment."""
    path = _write(_folder("/gone/proj") / _name("F1"), [_user("u1", None, "x")], first=first)
    assert PiAdapter().transcript_kind(path) is TranscriptKind.UNRECOGNIZED


def test_a_first_line_too_long_to_vouch_for_is_not_debris(home, monkeypatch):
    """A first line past the limit is not read in full, so the file is not called debris."""
    path = _write(_folder("/gone/proj") / _name("F1"), [_user("u1", None, "x" * 200)])
    monkeypatch.setattr(PiAdapter, "fragment_line_limit", 100)
    assert PiAdapter().transcript_kind(path) is TranscriptKind.UNRECOGNIZED
    monkeypatch.setattr(PiAdapter, "fragment_line_limit", 10_000)
    assert PiAdapter().transcript_kind(path) is TranscriptKind.FRAGMENT


# --- 2: a session in a format sx does not know keeps its folder ------------------------


def test_a_dead_projects_folder_holding_an_unknown_version_is_not_offered(home):
    """Pi v4 is refused for delete and move, so cleanup must not delete it either."""
    gone = str(home / "gone" / "proj")
    path = _session(gone, "V4", version=4)
    _age_tree(PiAdapter().sessions_dir())
    adapter = PiAdapter()

    assert "version 4" in adapter.delete(next(iter(adapter.discover()))).refused[path]
    assert adapter.find_orphans() == []
    assert path.exists()


def test_a_dead_projects_folder_that_gains_an_unknown_version_is_refused_at_deletion(home):
    """Asked again at the moment of deletion, whatever the scan saw."""
    gone = str(home / "gone" / "proj")
    _session(gone, "S1")
    _age_tree(PiAdapter().sessions_dir())
    adapter = PiAdapter()
    (orphan,) = adapter.find_orphans()
    assert orphan.kind is OrphanKind.DEAD_PROJECT
    later = _session(gone, "V4", version=4)
    _age_tree(later.parent)
    unfingerprinted = Orphan(orphan.harness, orphan.kind, orphan.paths, orphan.reason)

    result = adapter.delete_orphan(unfingerprinted)

    assert result.failed and "version 4" in result.refused[later.parent]
    assert later.exists()


# --- 10: a symlinked project folder ---------------------------------------------------


def test_a_folder_linked_to_another_lists_each_transcript_once(home):
    """Reached through the link and through the real folder: listed once, by its real path."""
    path = _session("/work/app", "S1")
    link = path.parent.parent / "--work-alias--"
    link.symlink_to(path.parent, target_is_directory=True)
    adapter = PiAdapter()

    assert list(adapter.session_files()) == [path]
    assert [session.paths for session in adapter.discover()] == [[path]]


def test_nothing_is_deleted_or_moved_through_a_symlinked_folder(home, tmp_path):
    """A folder linked to a directory outside the store is browsed, never acted on."""
    outside = tmp_path / "outside"
    path = _write(outside / _name("S1"), [_header("S1", "/work/app"), _user("u1", None, "hi")])
    _age_tree(outside)
    sessions = PiAdapter().sessions_dir()
    sessions.mkdir(parents=True)
    (sessions / encode_session_folder("/work/app")).symlink_to(outside, target_is_directory=True)
    adapter = PiAdapter()
    (session,) = adapter.discover()
    before = path.read_bytes()

    assert "symbolic link" in adapter.delete(session).refused[session.primary_path]
    plan = adapter.plan_move([session], Path("/work/app"), Path("/work/renamed"))
    assert "symbolic link" in plan.blocked[session.primary_path]
    assert not plan.relocations and not plan.rewrites
    assert path.read_bytes() == before


def test_a_folder_that_becomes_a_link_after_planning_is_refused_when_the_move_runs(home):
    """Planning saw a real folder; the unit asks again right before it runs."""
    path = _session("/work/app", "S1")
    _age_tree(path.parent)
    adapter = PiAdapter()
    plan = adapter.plan_move(list(adapter.discover()), Path("/work/app"), Path("/work/renamed"))
    assert plan.relocations
    moved_away = path.parent.parent / "real-folder"  # inside the store: only the link is new
    path.parent.rename(moved_away)
    path.parent.symlink_to(moved_away, target_is_directory=True)
    before = (moved_away / path.name).read_bytes()

    result = adapter.move(plan)

    assert not result.moved and not result.rewritten
    assert any("symbolic link" in reason for reason in result.refused.values())
    assert (moved_away / path.name).read_bytes() == before


# --- 4: fork links after an interrupted move -------------------------------------------


def _forked(home: Path) -> tuple[Path, Path]:
    """Write S1 in /x/old (forked by F1 in /elsewhere) and S2 in /x/old/sub, all idle."""
    s1 = _session("/x/old", "S1")
    _session("/x/old/sub", "S2")
    f1 = _session("/elsewhere", "F1", parentSession=str(s1))
    _age_tree(PiAdapter().sessions_dir(), 3600)
    return s1, f1


def _interrupt_second_rename(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the second rename of the move raise Ctrl-C, as a user would."""
    import sx.adapters.base as base_module

    real = base_module.rename_no_clobber
    calls = {"n": 0}

    def rename(src: Path, dst: Path) -> str | None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt("ctrl-c on the second unit")
        return real(src, dst)

    monkeypatch.setattr(base_module, "rename_no_clobber", rename)


def _link(path: Path) -> str:
    """Return a transcript's fork link."""
    return json.loads(path.read_bytes().splitlines()[0])["parentSession"]


def _service(home: Path):
    """Return a move service for Pi alone, logging into the sandbox."""
    from sx.service import MoveService

    return MoveService({"pi": PiAdapter()}, log_path=home / "ops.log")


def test_an_interrupted_move_still_repoints_the_links_of_the_units_that_finished(home, monkeypatch):
    """S1's unit finished before Ctrl-C: F1 must not be left naming S1's old path."""
    s1, f1 = _forked(home)
    _interrupt_second_rename(monkeypatch)
    service = _service(home)

    with pytest.raises(KeyboardInterrupt):
        service.move(service.plan(Path("/x/old"), Path("/x/new")))

    moved = _folder("/x/new") / s1.name
    assert moved.exists() and not s1.exists()
    assert _link(f1) == str(moved)


def test_a_rerun_repairs_a_link_an_interrupted_run_left_dangling(home, monkeypatch):
    """A run killed before it re-pointed anything leaves F1 dangling; the re-run fixes it."""
    s1, f1 = _forked(home)
    with monkeypatch.context() as killed:  # as an older sx, or kill -9, would leave it
        killed.setattr(PiAdapter, "_repoint_fork_links", lambda *a, **k: None)
        _interrupt_second_rename(killed)
        service = _service(home)
        with pytest.raises(KeyboardInterrupt):
            service.move(service.plan(Path("/x/old"), Path("/x/new")))
    moved = _folder("/x/new") / s1.name
    assert moved.exists() and _link(f1) == str(s1)  # the premise: dangling

    service = _service(home)
    dry = service.move(service.plan(Path("/x/old"), Path("/x/new")), dry_run=True)["pi"]
    assert f1 in dry.rewritten and _link(f1) == str(s1)
    result = service.move(service.plan(Path("/x/old"), Path("/x/new")))["pi"]

    assert f1 in result.rewritten and _link(f1) == str(moved)


def test_a_dangling_link_the_move_did_not_cause_is_left_alone(home):
    """A link to a deleted transcript is not 'repaired' onto some other file."""
    s1, f1 = _forked(home)
    gone = _session("/x/old", "GONE")
    lost = _session("/elsewhere", "F2", parentSession=str(gone))
    gone.unlink()
    _age_tree(PiAdapter().sessions_dir(), 3600)
    service = _service(home)

    service.move(service.plan(Path("/x/old"), Path("/x/new")))

    assert _link(lost) == str(gone)
    assert _link(f1) == str(_folder("/x/new") / s1.name)


# --- 8: mutants that survived ----------------------------------------------------------


def test_a_fork_link_changed_after_it_was_read_is_left_as_it_now_is(home, monkeypatch):
    """P19: the user re-forks CHILD between sx reading its header and rewriting it."""
    parent = _session("/old/proj", "PARENT")
    child = _session("/other/proj", "CHILD", parentSession=str(parent))
    _age_tree(PiAdapter().sessions_dir(), 3600)
    real_read = PiAdapter.read_header
    real_repoint = PiAdapter._repoint_fork_links

    def stale_then_changed(self, path):
        """Return the header as read, then let the user change the link on disk."""
        header = real_read(self, path)
        if path == child and header and header.get("parentSession") == str(parent):
            lines = child.read_bytes().splitlines()
            first = {**json.loads(lines[0]), "parentSession": "/somewhere/else.jsonl"}
            child.write_bytes(b"\n".join([_compact(first), *lines[1:]]) + b"\n")
            _age_tree(child.parent, 3600)
        return header

    def repoint(self, *args, **kwargs):
        """Re-point links while every header read is stale."""
        monkeypatch.setattr(PiAdapter, "read_header", stale_then_changed)
        try:
            return real_repoint(self, *args, **kwargs)
        finally:
            monkeypatch.setattr(PiAdapter, "read_header", real_read)

    monkeypatch.setattr(PiAdapter, "_repoint_fork_links", repoint)
    service = _service(home)

    service.move(service.plan(Path("/old/proj"), Path("/new/proj")))

    assert _link(child) == "/somewhere/else.jsonl"


def _leftover_store(home: Path) -> Path:
    """Write Pi's only session for /proj, idle for days, and return its folder."""
    path = _session("/proj", "S1")
    _age_tree(path.parent)
    return path.parent


def test_a_leftover_delete_is_refused_once_oh_my_pi_marks_the_store(home):
    """P29: the folder was offered from Pi's store; omp starts using the store before the delete."""
    folder = _leftover_store(home)
    adapter = PiAdapter()
    offer = adapter.project_leftovers("/proj")
    assert offer is not None and offer.paths == [folder]
    (session,) = adapter.discover()
    assert adapter.delete(session).removed
    (adapter.agent_dir() / "agent.db").write_bytes(b"")

    result = adapter.delete_project_leftovers(offer)

    assert result.failed and "holds Oh My Pi's data" in result.refused[folder]
    assert folder.is_dir()


def test_no_leftovers_are_offered_from_a_store_oh_my_pi_marked(home):
    """P29o: the offer itself is not made."""
    _leftover_store(home)
    adapter = PiAdapter()
    (adapter.agent_dir() / "agent.db").write_bytes(b"")
    assert adapter.project_leftovers("/proj") is None


def test_a_move_is_refused_once_oh_my_pi_marks_the_store_after_planning(home):
    """P43: re-checked when the move runs, not only when it is planned."""
    path = _session("/old/proj", "S1")
    _age_tree(path.parent, 3600)
    adapter = PiAdapter()
    plan = adapter.plan_move(list(adapter.discover()), Path("/old/proj"), Path("/new/proj"))
    assert plan.relocations
    (adapter.agent_dir() / "agent.db").write_bytes(b"")
    before = path.read_bytes()

    result = adapter.move(plan)

    assert result.failed and "holds Oh My Pi's data" in result.refused[adapter.sessions_dir()]
    assert path.read_bytes() == before and not _folder("/new/proj").exists()


@pytest.mark.parametrize("stray", [False, True], ids=["own-folder", "stray"])
def test_a_fork_link_spelled_through_a_symlink_is_repointed(home, stray):
    """P48: the link names the parent through another path to the store; matched by realpath.

    A stray parent — in a folder named for another directory — is the case only
    that match covers: nothing else could tell which folder it left.
    """
    if stray:
        records = [_header("PARENT", "/old/proj"), _user("u1", None, "hello")]
        parent = _write(_folder("/elsewhere") / _name("PARENT"), records)
    else:
        parent = _session("/old/proj", "PARENT")
    alias = home / "alias"
    alias.symlink_to(home / ".pi", target_is_directory=True)
    spelled = str(alias / parent.relative_to(home / ".pi"))
    child = _session("/other/proj", "CHILD", parentSession=spelled)
    _age_tree(PiAdapter().sessions_dir(), 3600)
    service = _service(home)

    service.move(service.plan(Path("/old/proj"), Path("/new/proj")))

    link = _link(child)
    assert link == str(_folder("/new/proj") / parent.name) and Path(link).exists()


# --- 14: an empty folder Pi still uses ---------------------------------------------------


def test_an_empty_folders_reason_says_pi_uses_it_when_it_starts_there(home):
    """Pi creates the folder only when it is missing: an old one is used again as is."""
    folder = PiAdapter().sessions_dir() / encode_session_folder("/work/app")
    folder.mkdir(parents=True)
    _age_tree(folder)

    (orphan,) = PiAdapter().find_orphans()

    assert orphan.kind is OrphanKind.EMPTY
    assert "Pi uses this folder when it starts in that project" in orphan.reason
    assert "next prompt fails" in orphan.reason


# --- a move cut short between link and unlink ------------------------------------------------


def test_a_rerun_finishes_a_transcript_left_linked_at_both_names(home):
    """A crash between ``link`` and ``unlink`` leaves one file under two names.

    Both are listed — they are two names — and the re-run unlinks the old
    one, since the file is already where it goes.
    """
    path = _session("/x/old", "A1")
    _session("/x/new", "N1")  # the destination exists, so the session moves on its own
    path.write_bytes(path.read_bytes().replace(b'"cwd":"/x/old"', b'"cwd":"/x/new"', 1))
    linked = _folder("/x/new") / path.name
    os.link(path, linked)
    _age_tree(PiAdapter().sessions_dir(), 3600)
    content = path.read_bytes()
    service = _service(home)
    assert sorted(str(p) for p in PiAdapter().session_files()) == sorted(
        [str(path), str(linked), str(_folder("/x/new") / _name("N1"))]
    )

    plan = service.plan(Path("/x/old"), Path("/x/new"))["pi"]
    assert plan.relocations == [(path, linked)] and not plan.blocked
    dry = service.move({"pi": plan}, dry_run=True)["pi"]
    assert dry.moved == [(path, linked)] and not dry.refused and path.exists()
    result = service.move(service.plan(Path("/x/old"), Path("/x/new")))["pi"]

    assert result.moved == [(path, linked)] and not result.refused
    assert not path.exists() and linked.read_bytes() == content
    assert os.stat(linked).st_nlink == 1


def test_two_different_files_of_one_name_are_never_taken_for_a_half_done_link(home):
    """Only one file under two names is finished; a different file there is refused."""
    path = _session("/x/old", "A1")
    _session("/x/new", "N1")
    path.write_bytes(path.read_bytes().replace(b'"cwd":"/x/old"', b'"cwd":"/x/new"', 1))
    other = _folder("/x/new") / path.name
    other.write_bytes(path.read_bytes())  # equal content, another file
    _age_tree(PiAdapter().sessions_dir(), 3600)
    service = _service(home)

    plan = service.plan(Path("/x/old"), Path("/x/new"))["pi"]

    assert not plan.relocations and "already exists" in plan.blocked[path]
    service.move({"pi": plan})
    assert path.exists() and other.exists()


def test_a_file_that_cannot_be_examined_counts_as_just_written(tmp_path):
    """P32: not knowing when a file was written counts as in use, the fail-safe way."""
    now = time.time()
    assert pi_module._written_within(tmp_path / "gone" / "f.jsonl", 90, now) is True
    old = tmp_path / "old.jsonl"
    old.write_text("x", encoding="utf-8")
    os.utime(old, (now - 3600, now - 3600))
    assert pi_module._written_within(old, 90, now) is False
