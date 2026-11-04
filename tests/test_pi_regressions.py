"""Regression tests for the Pi adapter's delete, store and format findings.

Each test reproduces one way the Pi adapter could destroy, hide or misread a
user's sessions, on SYNTHETIC stores under ``tmp_path``. They use only the
adapters' public surface — no helper added with the fixes — so the same file
can be run against an earlier ``sx`` to watch every one of them fail there.

Both Pi's and Oh My Pi's modules have ``home`` redirected into the sandbox, and
the suite's autouse fixtures scrub every ``PI_*``/``OMP_*``/``XDG_*`` override.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from sx.adapters import omp as omp_module
from sx.adapters import pi as pi_module
from sx.adapters.omp import OmpAdapter
from sx.adapters.pi import PiAdapter, agent_dir_override, encode_session_folder
from sx.model import Orphan, OrphanKind, Session
from sx.service import DeleteService

TS = "2026-05-01T10:00:00.000Z"
HOUR = 3600
DAY = 24 * HOUR

needs_permissions = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root reads and traverses whatever the mode",
)


# --- fixtures and builders ---------------------------------------------------------


@pytest.fixture(autouse=True)
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point both Pi adapters' home, and the temp directory, into the sandbox.

    Returns:
        The sandboxed home directory.
    """
    home = tmp_path / "home"
    home.mkdir()
    systmp = tmp_path / "systmp"
    systmp.mkdir()
    monkeypatch.setattr(pi_module, "home", lambda: home)
    monkeypatch.setattr(omp_module, "home", lambda: home)
    monkeypatch.setenv("TMPDIR", str(systmp))
    return home


@pytest.fixture()
def project(sandbox: Path) -> Path:
    """Return a project directory that exists."""
    path = sandbox / "work" / "app"
    path.mkdir(parents=True)
    return path


def _sessions(home: Path) -> Path:
    """Return Pi's default sessions directory under the sandboxed home."""
    return home / ".pi" / "agent" / "sessions"


def _name(sid: str) -> str:
    """Return the file name Pi gives a session."""
    return f"2026-05-01T10-00-00-000Z_{sid}.jsonl"


def _header(sid: str, cwd: str | None, **extra) -> dict:
    """Build a version 3 session header."""
    header: dict = {"type": "session", "version": 3, "id": sid, "timestamp": TS}
    if cwd is not None:
        header["cwd"] = cwd
    return {**header, **extra}


def _user(entry_id: str, parent: str | None, text: str) -> dict:
    """Build a user message entry."""
    message = {"role": "user", "content": text, "timestamp": 1}
    return {"type": "message", "id": entry_id, "parentId": parent, "timestamp": TS,
            "message": message}


def _write(path: Path, records: list[dict], *, prefix: bytes = b"") -> Path:
    """Write compact JSON lines, after ``prefix``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = b"".join(
        json.dumps(r, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        for r in records
    )
    path.write_bytes(prefix + body)
    return path


def _age(path: Path, seconds: float = HOUR) -> None:
    """Backdate one path."""
    past = time.time() - seconds
    os.utime(path, (past, past), follow_symlinks=False)


def _age_tree(root: Path, seconds: float = 2 * DAY) -> None:
    """Backdate a folder and everything in it, deepest entries first."""
    for top, dirs, files in os.walk(root, topdown=False):
        for name in (*files, *dirs):
            _age(Path(top, name), seconds)
    _age(root, seconds)


def _session(home: Path, cwd: Path | str, sid: str = "S1", **header_extra) -> Path:
    """Write a two-turn Pi session for ``cwd``, aged an hour."""
    path = _sessions(home) / encode_session_folder(str(cwd)) / _name(sid)
    _write(path, [_header(sid, str(cwd), **header_extra), _user("u1", None, "hello pi")])
    _age(path)
    return path


def _slot(title: str = "") -> bytes:
    """Build Oh My Pi's title slot: one line padded to 256 bytes, newline included."""
    fields = {"type": "title", "v": 1, "title": title, "updatedAt": TS}
    bare = json.dumps({**fields, "pad": ""}, separators=(",", ":")).encode("utf-8") + b"\n"
    padded = {**fields, "pad": " " * (256 - len(bare))}
    return json.dumps(padded, separators=(",", ":")).encode("utf-8") + b"\n"


def _service(adapter: PiAdapter, home: Path) -> DeleteService:
    """Return a delete service logging inside the sandbox."""
    return DeleteService({adapter.name: adapter}, log_path=home / "ops.log")


# --- D1: a folder Pi just created is not an orphan ---------------------------------------


def test_a_folder_pi_just_created_is_not_offered(sandbox, project):
    """Pi makes its project folder at startup, long before the first session file."""
    folder = _sessions(sandbox) / encode_session_folder(str(project))
    folder.mkdir(parents=True)
    _age(folder, 23 * HOUR)  # most of a day: still not idle long enough

    assert [o for o in PiAdapter().find_orphans() if o.paths == [folder]] == []


def test_a_fresh_file_keeps_an_old_folder_from_being_offered(sandbox, project):
    """The folder and everything in it must have been idle for a day."""
    folder = _sessions(sandbox) / encode_session_folder(str(project))
    folder.mkdir(parents=True)
    (folder / "notes.txt").write_text("just written", encoding="utf-8")
    _age(folder, 2 * DAY)

    assert [o for o in PiAdapter().find_orphans() if o.paths == [folder]] == []


def test_an_empty_folder_idle_for_a_day_is_offered(sandbox, project):
    """Once nothing in it has changed for a day, it is debris."""
    folder = _sessions(sandbox) / encode_session_folder(str(project))
    folder.mkdir(parents=True)
    (folder / "notes.txt").write_text("old", encoding="utf-8")
    _age_tree(folder, 25 * HOUR)

    (orphan,) = PiAdapter().find_orphans()

    assert (orphan.kind, orphan.paths) == (OrphanKind.EMPTY, [folder])


def test_an_empty_folder_is_checked_again_when_it_is_deleted(sandbox, project):
    """However the orphan was obtained, a folder Pi may have just created stays."""
    folder = _sessions(sandbox) / encode_session_folder(str(project))
    folder.mkdir(parents=True)
    adapter = PiAdapter()
    orphan = Orphan(adapter.name, OrphanKind.EMPTY, [folder], "empty project folder")

    preview = _service(adapter, sandbox).preview_orphan(orphan)
    result = _service(adapter, sandbox).delete_orphan(orphan)

    assert preview.removed == [] and folder in preview.refused
    assert folder.is_dir() and folder in result.refused


def test_leftovers_wait_until_the_folder_has_been_idle_for_a_day(sandbox, project):
    """The project's last session, written an hour ago, may still be open in Pi."""
    path = _session(sandbox, project)
    adapter = PiAdapter()

    assert adapter.project_leftovers(str(project)) is None
    _age_tree(path.parent)
    assert adapter.project_leftovers(str(project)).paths == [path.parent]


def test_leftovers_stay_if_anything_in_the_folder_was_written_since(sandbox, project):
    """Not only transcripts: any recent write means the folder may be in use."""
    path = _session(sandbox, project)
    _age_tree(path.parent)
    adapter = PiAdapter()
    (session,) = adapter.discover()
    leftovers = adapter.project_leftovers(str(project))
    adapter.delete(session)
    (path.parent / "notes.txt").write_text("written just now", encoding="utf-8")

    result = adapter.delete_project_leftovers(leftovers)

    assert path.parent.is_dir() and path.parent in result.refused


# --- D2: only what Pi itself leaves behind is a fragment -----------------------------------


def test_a_file_whose_header_is_not_first_is_listed_never_offered(sandbox, project):
    """A future format with a record before the header: browsable, never deleted."""
    folder = _sessions(sandbox) / encode_session_folder(str(project))
    future = _write(
        folder / _name("FUTURE"),
        [
            {"type": "preamble", "format": 4},
            _header("FUTURE", str(project), version=4),
            _user("u1", None, "important design discussion"),
        ],
    )
    _age_tree(folder)
    adapter = PiAdapter()

    listed = list(adapter.discover())
    orphans = adapter.find_orphans()

    assert [(s.title, s.project_path, s.paths) for s in listed] == [
        (f"(unrecognized format) {future.name}", None, [future])
    ]
    assert orphans == []
    result = _service(adapter, sandbox).delete(listed[0])
    assert future.exists() and result.failed and future in result.refused
    assert [m.text for m in adapter.load(listed[0])] == ["important design discussion"]


@needs_permissions
def test_a_session_sx_cannot_read_is_never_deleted(sandbox, project):
    """Unlinking needs only the folder's permission; an unreadable file is not debris."""
    path = _session(sandbox, project)
    _age_tree(path.parent)
    os.chmod(path, 0o000)
    try:
        adapter = PiAdapter()
        listed = list(adapter.discover())
        orphans = adapter.find_orphans()
        result = adapter.delete(listed[0]) if listed else None
    finally:
        os.chmod(path, 0o644)

    assert orphans == []
    assert [s.title for s in listed] == [f"(unrecognized format) {path.name}"]
    assert path.exists() and path in result.refused


def test_only_what_pi_appends_after_a_delete_is_a_fragment(sandbox, project):
    """An empty file, or a tree entry first: a fragment. Anything else: unrecognized."""
    folder = _sessions(sandbox) / encode_session_folder(str(project))
    folder.mkdir(parents=True)
    empty = folder / _name("E")
    empty.write_bytes(b"")
    appended = _write(folder / _name("A"), [_user("u9", "u8", "one more turn")])
    no_parent = _write(folder / _name("X"), [{"type": "message", "id": "u1"}])
    unknown_type = _write(folder / _name("Y"), [{"type": "banner", "id": "b", "parentId": None}])
    _age_tree(folder)
    adapter = PiAdapter()

    offered = sorted(o.paths[0].name for o in adapter.find_orphans())
    listed = sorted(s.title for s in adapter.discover())

    assert offered == sorted([empty.name, appended.name])
    assert listed == sorted(f"(unrecognized format) {p.name}" for p in (no_parent, unknown_type))


def test_a_fragment_that_became_a_session_is_refused(sandbox, project):
    """Pi wrote a header after the scan; one without a ``cwd`` is a session all the same."""
    folder = _sessions(sandbox) / encode_session_folder(str(project))
    path = _write(folder / _name("C"), [_header("C", None), _user("u1", None, "real prompt")])
    _age(path)
    adapter = PiAdapter()
    orphan = Orphan(adapter.name, OrphanKind.STRAY_TEMP, [path], "a fragment")

    result = _service(adapter, sandbox).delete_orphan(orphan)

    assert path.exists() and "now holds a session" in result.refused[path]


def test_a_dead_projects_folder_that_gained_a_fragment_is_refused(sandbox):
    """At delete time the folder must still hold only sessions of gone projects."""
    folder = _sessions(sandbox) / "--sx-gone--"
    _write(folder / _name("D"), [_header("D", "/sx-gone")])
    _write(folder / _name("F"), [_user("u9", "u8", "appended after the scan")])
    _age_tree(folder)
    adapter = PiAdapter()
    orphan = Orphan(adapter.name, OrphanKind.DEAD_PROJECT, [folder], "project gone: /sx-gone")

    result = _service(adapter, sandbox).delete_orphan(orphan)

    assert folder.is_dir() and folder in result.refused


def test_an_orphan_naming_a_file_sx_does_not_recognize_is_refused(sandbox, project):
    """At delete time the file must still be a fragment, whatever the orphan says."""
    folder = _sessions(sandbox) / encode_session_folder(str(project))
    path = _write(folder / _name("X"), [{"type": "preamble"}, _header("X", str(project))])
    _age_tree(folder)
    adapter = PiAdapter()
    orphan = Orphan(adapter.name, OrphanKind.STRAY_TEMP, [path], "a fragment")

    result = _service(adapter, sandbox).delete_orphan(orphan)

    assert path.exists() and path in result.refused


def test_a_file_sx_does_not_recognize_never_travels_with_its_folder(sandbox):
    """The session moves out on its own; the unrecognized file stays, byte for byte."""
    old, new = Path("/old/proj"), Path("/new/proj")
    path = _session(sandbox, old)
    unknown = _write(path.parent / _name("X"), [{"type": "preamble"}, _header("X", str(old))])
    _age_tree(path.parent)
    before = unknown.read_bytes()
    adapter = PiAdapter()

    result = adapter.move(adapter.plan_move(adapter.sessions_for_project(old), old, new))

    destination = _sessions(sandbox) / encode_session_folder(str(new))
    assert (destination / path.name).is_file() and not path.exists()
    assert unknown.exists() and unknown.read_bytes() == before
    assert not (destination / unknown.name).exists()
    assert not result.refused


def test_files_sx_does_not_recognize_are_counted_in_the_warnings(sandbox, project):
    """``sx harnesses`` says how many there are, and that they are left alone."""
    folder = _sessions(sandbox) / encode_session_folder(str(project))
    _write(folder / _name("X"), [{"type": "preamble"}])
    (folder / _name("Y")).write_bytes(b"[1,2,3]\n")
    _session(sandbox, project)

    warnings = PiAdapter().store_warnings()

    assert any("2 file(s)" in w and "never deletes or moves" in w for w in warnings)


# --- D2(c) and M10: a store Oh My Pi wrote to is Oh My Pi's -------------------------------


def test_pi_leaves_a_store_holding_oh_my_pis_database_alone(sandbox, project):
    """Oh My Pi shared Pi's agent directory once; its database is still there."""
    path = _session(sandbox, project)
    (sandbox / ".pi" / "agent" / "agent.db").write_bytes(b"")
    adapter = PiAdapter()
    session = Session(harness="pi", session_id="S1", project_path=str(project), paths=[path])

    assert adapter.available() is False
    assert list(adapter.discover()) == [] and adapter.find_orphans() == []
    assert path.exists() and path in adapter.delete(session).refused
    plan = adapter.plan_move([session], project, sandbox / "elsewhere")
    assert path in plan.blocked and not plan.relocations and not plan.rewrites
    assert any("Oh My Pi" in w and "agent.db" in w for w in adapter.store_warnings())


def test_pi_leaves_a_shared_store_alone_whatever_profile_oh_my_pi_reads(
    sandbox, project, monkeypatch
):
    """Oh My Pi wrote a slot-first transcript there; sx now runs under a profile."""
    shared = sandbox / "shared-agent"
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(shared))
    live = _write(
        shared / "sessions" / "-work-app" / _name("LIVE"),
        [_header("LIVE", str(project)), _user("u1", None, "x")],
        prefix=_slot("x"),
    )
    _age(live)
    monkeypatch.setenv("OMP_PROFILE", "work")
    adapter = PiAdapter()

    assert OmpAdapter().sessions_dir() != adapter.sessions_dir()  # the premise
    assert list(adapter.discover()) == [] and adapter.find_orphans() == []
    assert live.exists()


def test_pi_leaves_oh_my_pis_header_first_sessions_and_backups_alone(
    sandbox, project, monkeypatch
):
    """A Pi delete would leave omp's artifacts and backup, and omp would restore it."""
    shared = sandbox / "shared-agent"
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(shared))
    bucket = shared / "sessions" / "-work-app"
    transcript = _write(bucket / _name("LEG1"), [_header("LEG1", str(project))])
    (bucket / transcript.stem).mkdir()
    (bucket / transcript.stem / "out.txt").write_text("x", encoding="utf-8")
    (bucket / f"{transcript.name}.0123abcd.bak").write_bytes(transcript.read_bytes())
    _age_tree(bucket, HOUR)
    monkeypatch.setenv("OMP_PROFILE", "work")
    adapter = PiAdapter()

    assert list(adapter.discover()) == [] and adapter.find_orphans() == []
    assert any("Oh My Pi" in w for w in adapter.store_warnings())


def test_a_pi_store_without_oh_my_pi_marks_stays_pis(sandbox, project, monkeypatch):
    """A Pi user's PI_CODING_AGENT_DIR, where Oh My Pi would look too: still Pi's."""
    agent = sandbox / "pi-agent"
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent))
    path = _write(
        agent / "sessions" / encode_session_folder(str(project)) / _name("S1"),
        [_header("S1", str(project)), _user("u1", None, "hello pi")],
    )
    _age(path)
    adapter = PiAdapter()

    assert OmpAdapter().sessions_dir() == adapter.sessions_dir()  # the premise
    assert adapter.available() is True
    assert [s.session_id for s in adapter.discover()] == ["S1"]
    assert not any("Oh My Pi" in w for w in adapter.store_warnings())


# --- D4: a project that cannot be checked is not gone ---------------------------------------


@pytest.fixture()
def locked(sandbox: Path):
    """Yield a live project inside a parent ``sx`` may not traverse."""
    guarded = sandbox / "guarded"
    project = guarded / "app"
    project.mkdir(parents=True)
    (project / "main.py").write_text("print('live')\n", encoding="utf-8")
    os.chmod(guarded, 0o000)
    try:
        yield project
    finally:
        os.chmod(guarded, 0o755)


@needs_permissions
def test_a_project_behind_an_unreadable_parent_is_not_dead(sandbox, locked):
    """``Path.exists()`` says False for EACCES; the project is intact."""
    path = _session(sandbox, locked)
    _age_tree(path.parent)
    adapter = PiAdapter()

    sessions = list(adapter.discover())
    orphans = adapter.find_orphans()

    assert [(s.session_id, s.is_orphan) for s in sessions] == [("S1", False)]
    assert orphans == []


# --- D13: PI_CODING_AGENT_DIR is read as Pi's normalizePath reads it -------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("~//pi-elsewhere", "{home}/pi-elsewhere"),
        ("~/./alt//agent/", "{home}/alt/agent"),
        ("/data/pi-agent ", "/data/pi-agent "),
        ("file:///data/pi-agent", "/data/pi-agent"),
        ("file://localhost/data/pi%20agent/", "/data/pi agent"),
        ("file:///data/x/../pi-agent", "/data/pi-agent"),
        ("file:///data/caf%C3%A9", "/data/café"),
    ],
)
def test_the_agent_dir_is_read_as_pi_reads_it(sandbox, monkeypatch, value, expected):
    """No trimming, Node's ``path.join`` for ``~/``, and ``file://`` URLs as paths."""
    monkeypatch.setenv("PI_CODING_AGENT_DIR", value)

    assert agent_dir_override(sandbox) == (Path(expected.format(home=sandbox)), None)
    assert PiAdapter().sessions_dir() == Path(expected.format(home=sandbox)) / "sessions"


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        ("  /data/pi-agent", "a relative path"),
        ("   ", "a relative path"),
        ("file://server/share/agent", "a file: URL"),
        ("file:///data/a%2Fb", "a file: URL"),
        ("file:///data/agent#top", "a file: URL"),
    ],
)
def test_a_value_pi_would_resolve_elsewhere_is_ignored(sandbox, monkeypatch, value, reason):
    """sx reads the default store and says why, rather than follow a guess."""
    monkeypatch.setenv("PI_CODING_AGENT_DIR", value)
    adapter = PiAdapter()

    assert adapter.sessions_dir() == _sessions(sandbox)
    assert any("PI_CODING_AGENT_DIR" in w and reason in w for w in adapter.store_warnings())


# --- D14: a format version sx does not know is never deleted --------------------------------


@pytest.mark.parametrize("version", [4, "3", True, 2.5])
def test_an_unknown_format_version_is_never_deleted(sandbox, project, version):
    """The same tripwire as for a move: sx refuses what it was not built against."""
    path = _session(sandbox, project, version=version)
    adapter = PiAdapter()
    (session,) = adapter.discover()
    before = path.read_bytes()

    preview = _service(adapter, sandbox).preview(session)
    result = _service(adapter, sandbox).delete(session)

    assert preview.removed == [] and f"version {version!r}" in preview.refused[path]
    assert result.failed and f"version {version!r}" in result.refused[path]
    assert path.read_bytes() == before
