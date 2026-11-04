"""Tests for sx.adapters.omp on SYNTHETIC stores under tmp_path.

The real ``~/.omp`` is never read and ``omp`` is never run. Every test points
the module's ``home`` (adapters import it into their own namespace) at a
sandbox, the suite's autouse fixtures scrub the ``PI_*``/``OMP_PROFILE``/XDG
variables, and ``TMPDIR`` is pointed at a sandbox directory so the temp-dir
bucket encoding is predictable. Lease files are real files under the sandbox,
locked by this process or a child it starts.
"""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sx.adapters import omp as omp_module
from sx.adapters import pi as pi_module
from sx.adapters.omp import (
    LEASE_FREE,
    LEASE_HELD,
    LEASE_UNKNOWN,
    SLOT_BYTES,
    OmpAdapter,
    memory_folder_name,
)
from sx.adapters.pi import PiAdapter, encode_session_folder
from sx.model import BranchMode, Capability, Message, OrphanKind, Role, Session
from sx.service import MoveService
from sx.util import file_mtime, is_within

TS = "2026-05-01T10:00:00.000Z"
HOUR = 3600
DAY = 24 * HOUR


# --- fixtures and builders ---------------------------------------------------------


@pytest.fixture(autouse=True)
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point both Pi adapters' home at a sandbox, and the temp dir beside it.

    The sandbox models a machine where omp has written sessions, so its
    private lease directory exists: omp creates it before its first write, and
    without one sx counts every session as open. No omp process runs in it:
    listing the processes finds none, so a test never depends on whether the
    developer has omp open (tests of the process scan replace the listing).

    Returns:
        The sandboxed home directory.
    """
    sandbox_home = tmp_path / "home"
    sandbox_home.mkdir()
    system_tmp = tmp_path / "systmp"
    system_tmp.mkdir()
    monkeypatch.setattr(omp_module, "home", lambda: sandbox_home)
    monkeypatch.setattr(pi_module, "home", lambda: sandbox_home)
    # raising=False: the stub is harmless where the seam does not exist.
    monkeypatch.setattr(omp_module, "_list_processes", list, raising=False)
    monkeypatch.setenv("TMPDIR", str(system_tmp))
    for name in ("TMP", "TEMP", "PI_TEST_SESSION_OWNERS_DIR", "LC_ALL", "LC_TIME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANG", "en_GB.UTF-8")
    _lease_dir(sandbox_home)
    return sandbox_home


def _compact(record: object) -> bytes:
    """Serialize a record as ``JSON.stringify`` does."""
    return json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _slot(title: str = "", *, source: str | None = None, size: int = SLOT_BYTES) -> bytes:
    """Build a title slot line as omp does: padded until it is ``size`` bytes.

    Args:
        title: The slot's title.
        source: ``auto`` or ``user``, when set.
        size: The line's length in bytes, newline included.

    Returns:
        The line.
    """
    fields: dict = {"type": "title", "v": 1, "title": title}
    if source:
        fields["source"] = source
    fields["updatedAt"] = TS
    bare = _compact({**fields, "pad": ""}) + b"\n"
    line = _compact({**fields, "pad": " " * (size - len(bare))}) + b"\n"
    assert len(line) == size
    return line


def _header(sid: str = "S1", cwd: str | None = None, **extra) -> dict:
    """Build a version 3 session header."""
    header: dict = {"type": "session", "version": 3, "id": sid, "timestamp": TS}
    if cwd is not None:
        header["cwd"] = cwd
    return {**header, **extra}


def _entry(entry_id: str, parent: str | None, entry_type: str = "message", **fields) -> dict:
    """Build a tree entry."""
    return {"type": entry_type, "id": entry_id, "parentId": parent, "timestamp": TS, **fields}


def _user(entry_id: str, parent: str | None, content) -> dict:
    """Build a user message entry."""
    return _entry(entry_id, parent, message={"role": "user", "content": content, "timestamp": 1})


def _assistant(entry_id: str, parent: str | None, text: str, **fields) -> dict:
    """Build an assistant message entry holding one text block."""
    message = {"role": "assistant", "content": [{"type": "text", "text": text}], "timestamp": 2}
    return _entry(entry_id, parent, message=message, **fields)


def _write(path: Path, records: list[dict], *, slot: bytes | None = b"") -> Path:
    """Write a transcript: an optional slot line, then compact JSON lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((slot or b"") + b"".join(_compact(r) + b"\n" for r in records))
    return path


def _age(*paths: Path, seconds: float = HOUR) -> None:
    """Backdate files and directories so they no longer count as being written."""
    past = time.time() - seconds
    for path in paths:
        os.utime(path, (past, past), follow_symlinks=False)


def _age_tree(root: Path, seconds: float) -> None:
    """Backdate a whole tree, deepest entries first."""
    for top, dirs, files in os.walk(root, topdown=False):
        _age(*(Path(top, name) for name in (*files, *dirs)), seconds=seconds)
    _age(root, seconds=seconds)


def _name(sid: str) -> str:
    """Return the file name omp gives a session."""
    return f"2026-05-01T10-00-00-000Z_{sid}.jsonl"


def _sessions(home: Path) -> Path:
    """Return the default sessions directory under the sandboxed home."""
    return home / ".omp" / "agent" / "sessions"


def _memories(home: Path) -> Path:
    """Return the default memories directory under the sandboxed home."""
    return home / ".omp" / "agent" / "memories"


def _bucket(project: Path | str) -> Path:
    """Return the bucket omp uses today for ``project``."""
    adapter = OmpAdapter()
    return adapter.sessions_dir() / adapter.folder_for(str(project))


def _session(
    project: Path | str,
    sid: str = "S1",
    *,
    title: str = "",
    slot: bool = True,
    records: list[dict] | None = None,
    **header_extra,
) -> Path:
    """Write an aged omp transcript for ``project`` into its bucket."""
    body = records if records is not None else [
        _user("u1", None, f"edit {project}/src/app.py ✓"),
        _assistant("a1", "u1", f"done in {project}"),
    ]
    path = _bucket(project) / _name(sid)
    header = _header(sid, str(project), **header_extra)
    _write(path, [header, *body], slot=_slot(title) if slot else None)
    _age(path)
    return path


def _only(adapter: OmpAdapter) -> Session:
    """Return the single session an adapter discovers."""
    sessions = list(adapter.discover())
    assert len(sessions) == 1, sessions
    return sessions[0]


def _lines(path: Path) -> list[bytes]:
    """Return a file's lines, terminators kept."""
    return path.read_bytes().splitlines(keepends=True)


def _texts(messages: list[Message]) -> list[str]:
    """Return each message's text."""
    return [message.text for message in messages]


def _move(adapter: OmpAdapter, old: Path, new: Path, *, dry_run: bool = False):
    """Plan and perform a move of every session the adapter finds under ``old``."""
    plan = adapter.plan_move(adapter.sessions_for_project(old), old, new)
    return plan, adapter.move(plan, dry_run=dry_run)


def _lease_dir(home: Path) -> Path:
    """Create the default lease directory, private as omp requires."""
    directory = home / ".omp" / "run" / "session-owners"
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    return directory


@contextmanager
def _flocked(lease: Path) -> Iterator[None]:
    """Hold an exclusive ``flock`` on ``lease`` from another open file description."""
    lease.touch()
    fd = os.open(lease, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


_CHILD_LOCKF = """
import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDWR)
fcntl.lockf(fd, fcntl.LOCK_EX)
print("ready", flush=True)
sys.stdin.read()
"""


@contextmanager
def _lockf_in_child(lease: Path) -> Iterator[None]:
    """Hold a POSIX ``lockf`` lock on ``lease`` from a child process.

    POSIX locks are per process, so this process could not observe its own.
    """
    lease.touch()
    child = subprocess.Popen(
        [sys.executable, "-I", "-c", _CHILD_LOCKF, str(lease)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None and child.stdout.readline().strip() == "ready"
        yield
    finally:
        assert child.stdin is not None and child.stdout is not None
        child.stdin.close()
        child.wait(timeout=10)
        child.stdout.close()


@pytest.fixture
def old(sandbox: Path) -> Path:
    """A project directory under the sandboxed home, which exists."""
    project = sandbox / "work" / "app"
    project.mkdir(parents=True)
    return project


@pytest.fixture
def new(sandbox: Path) -> Path:
    """Where the project is moved to."""
    return sandbox / "work" / "renamed"


# --- where the store is -------------------------------------------------------------


def test_the_default_store(sandbox):
    """Sessions and memories under ``~/.omp/agent``; leases under ``~/.omp/run``."""
    adapter = OmpAdapter()
    agent = sandbox / ".omp" / "agent"
    assert adapter.sessions_dir() == agent / "sessions"
    assert adapter.memories_dir() == agent / "memories"
    assert adapter.lease_dir() == sandbox / ".omp" / "run" / "session-owners"
    assert adapter.store_roots() == [agent / "sessions", agent / "memories"]
    assert adapter.protected_paths() == [
        agent / "memories" / "sharpshooter",
        agent / "memories" / "mnemopi",
    ]
    assert adapter.store_notes() == [] and adapter.store_warnings() == []
    assert adapter.capabilities == (
        Capability.BROWSE | Capability.ORPHANS | Capability.DELETE | Capability.MOVE
    )
    assert adapter.supports_branches is True


@pytest.mark.parametrize(
    ("value", "root"),
    [("omp-config", "omp-config"), ("/etc/omp", "etc/omp"), ("nested/dir/", "nested/dir")],
)
def test_pi_config_dir_is_a_name_joined_onto_home(sandbox, monkeypatch, value, root):
    """Even an absolute value lands inside home, as Node's ``path.join`` puts it."""
    monkeypatch.setenv("PI_CONFIG_DIR", value)
    adapter = OmpAdapter()
    assert adapter.sessions_dir() == sandbox / root / "agent" / "sessions"
    assert adapter.lease_dir() == sandbox / root / "run" / "session-owners"
    assert any("PI_CONFIG_DIR" in note for note in adapter.store_notes())


@pytest.mark.parametrize("variable", ["OMP_PROFILE", "PI_PROFILE"])
def test_a_profile_selects_its_own_agent_dir(sandbox, monkeypatch, variable):
    """The value is trimmed; leases stay shared by every profile."""
    monkeypatch.setenv(variable, "  work.1  ")
    adapter = OmpAdapter()
    agent = sandbox / ".omp" / "profiles" / "work.1" / "agent"
    assert adapter.sessions_dir() == agent / "sessions"
    assert adapter.memories_dir() == agent / "memories"
    assert adapter.lease_dir() == sandbox / ".omp" / "run" / "session-owners"
    assert any(variable in note and "'work.1'" in note for note in adapter.store_notes())


def test_omp_profile_wins_even_when_empty(sandbox, monkeypatch):
    """An empty ``OMP_PROFILE`` still overrides ``PI_PROFILE``."""
    monkeypatch.setenv("PI_PROFILE", "legacy")
    monkeypatch.setenv("OMP_PROFILE", "work")
    assert OmpAdapter().sessions_dir() == sandbox / ".omp/profiles/work/agent/sessions"
    monkeypatch.setenv("OMP_PROFILE", "")
    assert OmpAdapter().sessions_dir() == _sessions(sandbox)
    monkeypatch.setenv("OMP_PROFILE", "default")
    assert OmpAdapter().sessions_dir() == _sessions(sandbox)


@pytest.mark.parametrize(
    "value", ["Work", "a..", "x.", ".", "..", "con", "NUL.txt", "-x", "a/b", "a" * 65, "é"]
)
def test_an_invalid_profile_is_ignored_with_a_warning(sandbox, monkeypatch, value):
    """Oh My Pi refuses to start with it; sx reads the default profile and says so."""
    monkeypatch.setenv("OMP_PROFILE", value)
    adapter = OmpAdapter()
    assert adapter.sessions_dir() == _sessions(sandbox)
    assert any("not a valid Oh My Pi profile" in w for w in adapter.store_warnings())


def test_an_absolute_agent_dir_override_is_followed(sandbox, tmp_path, monkeypatch):
    """Sessions and memories follow it; leases stay under the config root."""
    agent = tmp_path / "elsewhere" / "agent"
    monkeypatch.setenv("PI_CODING_AGENT_DIR", f"{agent}/")
    adapter = OmpAdapter()
    assert adapter.sessions_dir() == agent / "sessions"
    assert adapter.memories_dir() == agent / "memories"
    assert adapter.lease_dir() == sandbox / ".omp" / "run" / "session-owners"
    assert any("PI_CODING_AGENT_DIR" in note for note in adapter.store_notes())


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        ("relative/agent", "relative path"),
        ("~/agent", "relative path"),
        (" /leading/space", "relative path"),
        ("/", "filesystem root"),
        ("HOME", "home directory"),
    ],
)
def test_an_unusable_agent_dir_override_is_ignored(sandbox, monkeypatch, value, reason):
    """Oh My Pi resolves against its own cwd and never expands ``~``; sx will not guess."""
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(sandbox) if value == "HOME" else value)
    adapter = OmpAdapter()
    assert adapter.sessions_dir() == _sessions(sandbox)
    assert any(reason in w for w in adapter.store_warnings())


def test_a_profile_ignores_the_agent_dir_override(sandbox, tmp_path, monkeypatch):
    """Named profiles never honor ``PI_CODING_AGENT_DIR``; the warning says so."""
    monkeypatch.setenv("OMP_PROFILE", "work")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "elsewhere"))
    adapter = OmpAdapter()
    assert adapter.sessions_dir() == sandbox / ".omp/profiles/work/agent/sessions"
    assert any("ignores it under profile 'work'" in w for w in adapter.store_warnings())


def test_an_override_naming_pi_profiles_agent_dir_counts_as_unset(sandbox, monkeypatch):
    """Oh My Pi's own rule for a child switched back to the default profile."""
    monkeypatch.setenv("OMP_PROFILE", "")
    monkeypatch.setenv("PI_PROFILE", "work")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(sandbox / ".omp/profiles/work/agent"))
    adapter = OmpAdapter()
    assert adapter.sessions_dir() == _sessions(sandbox)
    assert adapter.store_warnings() == []


def test_xdg_data_home_takes_over_only_once_its_omp_dir_exists(sandbox, tmp_path, monkeypatch):
    """No ``agent/`` segment there; memories stay in the agent dir."""
    xdg = tmp_path / "xdg-data"
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg))
    assert OmpAdapter().sessions_dir() == _sessions(sandbox)

    (xdg / "omp").mkdir(parents=True)
    adapter = OmpAdapter()
    assert adapter.sessions_dir() == xdg / "omp" / "sessions"
    assert adapter.memories_dir() == _memories(sandbox)
    assert any("XDG_DATA_HOME" in note for note in adapter.store_notes())


def test_xdg_data_home_with_a_profile_needs_the_profiles_dir(sandbox, tmp_path, monkeypatch):
    """``<xdg>/omp`` alone does not move a named profile's sessions."""
    xdg = tmp_path / "xdg-data"
    (xdg / "omp").mkdir(parents=True)
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg))
    monkeypatch.setenv("OMP_PROFILE", "work")
    assert OmpAdapter().sessions_dir() == sandbox / ".omp/profiles/work/agent/sessions"
    (xdg / "omp" / "profiles" / "work").mkdir(parents=True)
    assert OmpAdapter().sessions_dir() == xdg / "omp/profiles/work/sessions"


def test_xdg_is_ignored_when_the_agent_dir_is_overridden(sandbox, tmp_path, monkeypatch):
    """Only the default agent directory moves under XDG."""
    xdg = tmp_path / "xdg-data"
    (xdg / "omp").mkdir(parents=True)
    agent = tmp_path / "custom" / "agent"
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg))
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent))
    assert OmpAdapter().sessions_dir() == agent / "sessions"


def test_xdg_state_home_takes_memories_and_leases(sandbox, tmp_path, monkeypatch):
    """Leases follow ``$XDG_STATE_HOME/omp`` under any profile or override."""
    state = tmp_path / "xdg-state"
    (state / "omp").mkdir(parents=True)
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
    adapter = OmpAdapter()
    assert adapter.memories_dir() == state / "omp" / "memories"
    assert adapter.lease_dir() == state / "omp" / "run" / "session-owners"
    assert adapter.sessions_dir() == _sessions(sandbox)

    monkeypatch.setenv("OMP_PROFILE", "work")
    adapter = OmpAdapter()
    assert adapter.memories_dir() == sandbox / ".omp/profiles/work/agent/memories"
    assert adapter.lease_dir() == state / "omp" / "run" / "session-owners"


def test_relative_xdg_dirs_are_ignored_and_make_liveness_unknown(sandbox, monkeypatch):
    """Oh My Pi would resolve them against its own cwd; sx cannot place the leases."""
    monkeypatch.setenv("XDG_DATA_HOME", "relative-data")
    monkeypatch.setenv("XDG_STATE_HOME", "relative-state")
    adapter = OmpAdapter()
    assert adapter.sessions_dir() == _sessions(sandbox)
    assert adapter.memories_dir() == _memories(sandbox)
    assert adapter.lease_dir() is None
    assert adapter.lease_state("S1") == LEASE_UNKNOWN
    warnings = adapter.store_warnings()
    assert any("XDG_DATA_HOME is a relative path" in w for w in warnings)
    assert any("XDG_STATE_HOME is a relative path" in w for w in warnings)


def test_a_custom_session_dir_is_warned_about(sandbox, monkeypatch):
    """Flat custom session directories are not read yet."""
    monkeypatch.setenv("PI_CODING_AGENT_SESSION_DIR", "/somewhere/flat")
    assert any("/somewhere/flat" in w for w in OmpAdapter().store_warnings())


def test_other_profiles_sessions_are_named_in_a_warning(sandbox, monkeypatch):
    """Only the active profile is listed; the others are pointed out."""
    (sandbox / ".omp/profiles/work/agent/sessions").mkdir(parents=True)
    (sandbox / ".omp/profiles/idle/agent").mkdir(parents=True)
    _sessions(sandbox).mkdir(parents=True)
    warnings = OmpAdapter().store_warnings()
    assert len(warnings) == 1 and "profile 'work'" in warnings[0]

    monkeypatch.setenv("OMP_PROFILE", "work")
    warnings = OmpAdapter().store_warnings()
    assert len(warnings) == 1 and "default profile" in warnings[0]


def test_availability_needs_the_sessions_directory(sandbox):
    """Another tool's extension in the agent dir does not mean omp is used."""
    extensions = sandbox / ".omp" / "agent" / "extensions"
    extensions.mkdir(parents=True)
    (extensions / "superset-hooks.ts").write_text("// not omp's", encoding="utf-8")
    adapter = OmpAdapter()
    assert adapter.available() is False
    _sessions(sandbox).mkdir()
    assert adapter.available() is True


def test_sessions_dir_never_raises(monkeypatch):
    """Without a home directory nothing is listed, cleaned or deletable."""

    def broken() -> Path:
        """Fail like ``Path.home`` without ``HOME``."""
        raise RuntimeError("no home directory")

    monkeypatch.setattr(omp_module, "home", broken)
    adapter = OmpAdapter()
    assert isinstance(adapter.sessions_dir(), Path)
    assert adapter.available() is False
    assert list(adapter.discover()) == [] and adapter.find_orphans() == []
    assert any("home directory" in w for w in adapter.store_warnings())


# --- bucket names ----------------------------------------------------------------------


def test_buckets_under_home_are_home_relative(sandbox):
    """``-<rel>``, separators and colons turned into ``-``; home itself is ``-``."""
    adapter = OmpAdapter()
    assert adapter.folder_for(str(sandbox / "src" / "app")) == "-src-app"
    assert adapter.folder_for(str(sandbox)) == "-"
    assert (
        adapter.folder_for(str(sandbox / "Documents" / "My Project" / "v1.2_x"))
        == "-Documents-My Project-v1.2_x"
    )
    assert adapter.folder_for(str(sandbox / "a:b" / "c\\d")) == "-a-b-c-d"


def test_buckets_under_the_temp_dir_are_temp_relative(tmp_path, monkeypatch):
    """``-tmp-<rel>``; the temp dir itself is ``-tmp``; a trailing slash is dropped."""
    system_tmp = tmp_path / "systmp"
    adapter = OmpAdapter()
    assert adapter.folder_for(str(system_tmp)) == "-tmp"
    assert adapter.folder_for(str(system_tmp / "scratch" / "x")) == "-tmp-scratch-x"
    monkeypatch.setenv("TMPDIR", f"{system_tmp}/")
    assert adapter.folder_for(str(system_tmp / "y")) == "-tmp-y"


@pytest.mark.parametrize("variable", ["TMP", "TEMP"])
def test_tmp_and_temp_stand_in_for_tmpdir(tmp_path, monkeypatch, variable):
    """Node's ``os.tmpdir()`` falls back to ``TMP``, then ``TEMP``."""
    other = tmp_path / "other-tmp"
    other.mkdir()
    monkeypatch.delenv("TMPDIR")
    monkeypatch.setenv(variable, str(other))
    assert OmpAdapter().folder_for(str(other / "x")) == "-tmp-x"


def test_other_directories_get_the_absolute_name(tmp_path):
    """Neither under home nor the temp dir: Pi's ``--<abs>--`` name."""
    project = tmp_path / "elsewhere" / "proj"
    assert OmpAdapter().folder_for(str(project)) == encode_session_folder(str(project))


def test_a_child_named_like_dot_dot_counts_as_outside(sandbox):
    """Oh My Pi tests the relative path's text, so ``..cache`` is not inside home."""
    project = sandbox / "..cache" / "x"
    assert OmpAdapter().folder_for(str(project)) == encode_session_folder(str(project))


def test_symlinks_are_resolved_before_encoding(sandbox, tmp_path):
    """Oh My Pi names a bucket for the real path, so aliases share it."""
    real = tmp_path / "real" / "proj"
    real.mkdir(parents=True)
    (sandbox / "linked").symlink_to(tmp_path / "real", target_is_directory=True)
    adapter = OmpAdapter()
    assert adapter.folder_for(str(sandbox / "linked" / "proj")) == encode_session_folder(str(real))

    (sandbox / "src" / "app").mkdir(parents=True)
    (tmp_path / "alias").symlink_to(sandbox / "src", target_is_directory=True)
    assert adapter.folder_for(str(tmp_path / "alias" / "app")) == "-src-app"


def test_symlinked_home_and_temp_dirs_are_resolved(sandbox, tmp_path, monkeypatch):
    """Like macOS's ``/var`` → ``/private/var``: either spelling reaches the bucket."""
    (tmp_path / "home-link").symlink_to(sandbox, target_is_directory=True)
    monkeypatch.setattr(omp_module, "home", lambda: tmp_path / "home-link")
    assert OmpAdapter().folder_for(str(sandbox / "x")) == "-x"

    (tmp_path / "private" / "var" / "folders" / "zz" / "T").mkdir(parents=True)
    (tmp_path / "var").symlink_to(tmp_path / "private" / "var", target_is_directory=True)
    monkeypatch.setenv("TMPDIR", f"{tmp_path}/var/folders/zz/T/")
    adapter = OmpAdapter()
    assert adapter.folder_for(f"{tmp_path}/private/var/folders/zz/T/scratch") == "-tmp-scratch"
    assert adapter.folder_for(f"{tmp_path}/var/folders/zz/T/scratch") == "-tmp-scratch"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS's /var is a symlink to /private/var")
def test_real_macos_temp_and_home_paths(monkeypatch):
    """The paths omp sees on a Mac: ``$TMPDIR`` under ``/var/folders``, home under ``/Users``."""
    monkeypatch.setenv("TMPDIR", "/var/folders/zz/sxtest0000000000000000000000gn/T/")
    monkeypatch.setattr(omp_module, "home", lambda: Path("/Users/sxtest-nobody"))
    adapter = OmpAdapter()
    private = "/private/var/folders/zz/sxtest0000000000000000000000gn/T"
    assert adapter.folder_for(f"{private}/scratch") == "-tmp-scratch"
    assert adapter.folder_for("/var/folders/zz/sxtest0000000000000000000000gn/T") == "-tmp"
    assert adapter.folder_for("/Users/sxtest-nobody/source/junxit/sx") == "-source-junxit-sx"
    assert adapter.folder_for("/Users/sxtest-nobody") == "-"
    assert (
        adapter.folder_for("/Volumes/sx-test-unplugged/proj")
        == "--Volumes-sx-test-unplugged-proj--"
    )


def _sha(path: Path | str) -> str:
    """Return the SHA-256 omp's hashed bucket names end with."""
    return hashlib.sha256(os.path.realpath(path).encode()).hexdigest()


def test_folder_names_include_the_names_omp_migrates_from(sandbox, tmp_path):
    """Today's name, ``--<abs>--`` for the real and lexical paths, and the hashed one."""
    project = sandbox / "src" / "app"
    real = os.path.realpath(project)
    assert OmpAdapter().folder_names(str(project)) == {
        "-src-app",
        encode_session_folder(real),
        f"home-app-{_sha(project)}",
    }

    (tmp_path / "real").mkdir()
    (sandbox / "link").symlink_to(tmp_path / "real", target_is_directory=True)
    aliased = sandbox / "link" / "proj"
    assert encode_session_folder(str(aliased)) in OmpAdapter().folder_names(str(aliased))


def test_hashed_names_slug_the_basename(tmp_path):
    """Runs of other characters become ``-``, edges are trimmed, the last 80 kept."""
    adapter = OmpAdapter()
    odd = tmp_path / "x" / "--My Project (v2)!--"
    assert f"abs-My-Project-v2-{_sha(odd)}" in adapter.folder_names(str(odd))
    long = tmp_path / ("a" * 50 + "b" * 50)
    assert f"abs-{'a' * 30}{'b' * 50}-{_sha(long)}" in adapter.folder_names(str(long))
    assert f"abs-project-{_sha('/')}" in adapter.folder_names("/")


def test_a_temp_dir_inside_home_wins_and_keeps_the_home_names(sandbox, monkeypatch):
    """The two shadowed home names still count as the project's."""
    cache = sandbox / "cache" / "T"
    cache.mkdir(parents=True)
    monkeypatch.setenv("TMPDIR", str(cache))
    project = cache / "scratch"
    adapter = OmpAdapter()
    assert adapter.folder_for(str(project)) == "-tmp-scratch"
    assert adapter.folder_names(str(project)) == {
        "-tmp-scratch",
        f"tmp-scratch-{_sha(project)}",
        "-cache-T-scratch",
        f"home-scratch-{_sha(project)}",
        encode_session_folder(os.path.realpath(project)),
    }


def test_memory_folders_use_the_recorded_path_as_is():
    """Oh My Pi's ``HEe``: always the absolute form, never resolved."""
    assert memory_folder_name("/Users/me/src/app") == "--Users-me-src-app--"
    assert memory_folder_name("C:\\work\\x") == "--C--work-x--"


# --- files, layouts and discovery ---------------------------------------------------


def test_slot_first_and_legacy_files_are_both_sessions(old):
    """Neither layout is mistaken for a fragment."""
    slotted = _session(old, "S1", title="Slot title")
    legacy = _session(old, "S2", slot=False)
    adapter = OmpAdapter()

    sessions = {s.session_id: s for s in adapter.discover()}

    assert set(sessions) == {"S1", "S2"}
    assert sessions["S1"].paths == [slotted] and sessions["S2"].paths == [legacy]
    assert sessions["S1"].project_path == str(old)
    assert sessions["S1"].created == datetime(2026, 5, 1, 10, tzinfo=UTC)
    assert sessions["S1"].size_bytes == slotted.stat().st_size
    assert adapter.read_header(slotted)["id"] == "S1"
    assert adapter.read_header(legacy)["id"] == "S2"
    assert adapter.find_orphans() == []


def test_a_slot_line_is_exactly_256_bytes():
    """The builder mirrors omp: padding fills the line, newline included."""
    line = _slot("Fix the flaky test", source="user")
    assert len(line) == SLOT_BYTES and line.endswith(b"\n")
    assert json.loads(line)["pad"].strip() == ""


def test_only_entry_first_files_are_fragments(sandbox, old):
    """omp's writer leaves entries with no slot and no header; a slot without one is not debris.

    Every full write omp makes (slot, header, entries) is a temp file renamed
    into place, so a slot-first file with no header is a file sx does not
    recognize: listed, never offered for cleanup.
    """
    bucket = _bucket(old)
    _write(bucket / _name("A"), [], slot=_slot("orphaned"))
    _write(bucket / _name("B"), [_user("u1", None, "hi")], slot=_slot())
    _write(bucket / _name("C"), [_user("u1", None, "hi")], slot=None)
    (bucket / _name("E")).write_bytes(b"")
    adapter = OmpAdapter()

    listed = sorted((s.title, s.extra.get("unrecognized")) for s in adapter.discover())
    assert listed == [
        (f"(unrecognized format) {_name(sid)}", "does not start with a session header")
        for sid in "AB"
    ]
    kinds = sorted((o.kind, o.paths[0].name) for o in adapter.find_orphans())
    assert kinds == [(OrphanKind.STRAY_TEMP, _name(sid)) for sid in "CE"]


def test_only_top_level_transcripts_are_sessions(sandbox, old):
    """Sub-agent transcripts in ``S/`` belong to their session."""
    path = _session(old)
    _write(path.parent / path.stem / "subagent.jsonl", [_header("SUB", str(old))], slot=_slot())
    assert [s.session_id for s in OmpAdapter().discover()] == ["S1"]


def test_the_slot_and_header_are_not_entries():
    """``is_entry`` passes everything else through."""
    adapter = OmpAdapter()
    assert not adapter.is_entry({"type": "title", "v": 1})
    assert not adapter.is_entry(_header())
    assert adapter.is_entry(_user("u1", None, "hi"))


# --- titles ----------------------------------------------------------------------------


def _title(path: Path) -> str:
    """Return the title the adapter discovers for the transcript at ``path``."""
    return next(s for s in OmpAdapter().discover() if s.paths == [path]).title


def test_the_slot_title_wins(old):
    """Oh My Pi rewrites the slot in place; the header's copy may be stale."""
    path = _bucket(old) / _name("S1")
    _write(path, [_header("S1", str(old), title="stale")], slot=_slot("Fix the parser"))
    assert _title(path) == "Fix the parser"


def test_a_blank_slot_title_hides_the_header_title(old):
    """Oh My Pi ignores the header's title whenever a slot is present."""
    path = _bucket(old) / _name("S1")
    _write(
        path,
        [_header("S1", str(old), title="stale"), _user("u1", None, "the prompt")],
        slot=_slot("   "),
    )
    assert _title(path) == "the prompt"


def test_without_a_slot_the_header_title_is_used(old):
    """Legacy files keep their title in the header."""
    path = _bucket(old) / _name("S1")
    _write(path, [_header("S1", str(old), title="Header title"), _user("u1", None, "x")], slot=None)
    assert _title(path) == "Header title"


def test_a_title_keeps_only_its_first_line_without_control_characters(old):
    """As omp's session list shows it."""
    path = _bucket(old) / _name("S1")
    _write(path, [_header("S1", str(old))], slot=_slot("\x1b[1mLine one\x07 \nLine two"))
    assert _title(path) == "[1mLine one"


def test_the_first_user_message_comes_next(old):
    """No title anywhere: the first line of the first user prompt — blank, it counts as none."""
    path = _bucket(old) / _name("S1")
    _write(path, [_header("S1", str(old)), _user("u1", None, "Fix it\nplease")], slot=_slot())
    assert _title(path) == "Fix it"
    _write(path, [_header("S1", str(old)), _user("u1", None, "\n\nFix it")], slot=_slot())
    assert _title(path).startswith("Untitled · ")


def test_an_empty_session_is_untitled_with_its_start_time(old):
    """``Untitled · HH:MM`` in local time, from the header's timestamp."""
    path = _bucket(old) / _name("S1")
    _write(path, [_header("S1", str(old))], slot=_slot())
    local = datetime(2026, 5, 1, 10, tzinfo=UTC).astimezone()
    assert _title(path) == f"Untitled · {local:%H:%M}"


def test_forks_and_continuations_are_marked(old):
    """``parentSession`` adds ``· fork``; ``continuationOf`` adds ``· continued``."""
    bucket = _bucket(old)
    fork = _write(bucket / _name("F"), [_header("F", str(old), parentSession="P")], slot=_slot("t"))
    both = _write(
        bucket / _name("B"),
        [_header("B", str(old), parentSession="P", continuationOf="P")],
        slot=_slot("t"),
    )
    assert _title(fork) == "t · fork"
    assert _title(both) == "t · continued · fork"


# --- the current branch ------------------------------------------------------------


def _branched(old: Path, last: dict) -> Session:
    """Write ``u1`` answered twice (``a1``, then ``u2``/``a2``), then ``last``."""
    records = [
        _user("u1", None, "the prompt"),
        _assistant("a1", "u1", "first answer"),
        _user("u2", "u1", "retried prompt"),
        _assistant("a2", "u2", "second answer"),
        last,
    ]
    _session(old, records=records)
    return _only(OmpAdapter())


def test_the_last_entrys_active_leaf_picks_the_current_branch(old):
    """A background entry on branch B leaves the session on branch A."""
    last = _entry("m1", "a2", "model_usage", activeLeafId="a1")
    transcript = OmpAdapter().load_transcript(_branched(old, last))
    assert _texts(transcript.current) == ["the prompt", "first answer"]
    branches = [_texts(branch.messages) for branch in transcript.branches]
    assert branches == [["retried prompt", "second answer"]]


def test_an_active_leaf_naming_no_entry_falls_back_as_omp_does(old):
    """The newest entry without a leaf pointer of its own wins (omp's ``Pbe``)."""
    last = _entry("m1", "a2", "model_usage", activeLeafId="gone")
    transcript = OmpAdapter().load_transcript(_branched(old, last))
    assert _texts(transcript.current) == ["the prompt", "retried prompt", "second answer"]


def test_a_null_active_leaf_resumes_an_empty_branch(old):
    """Oh My Pi's leaf was reset to the root: nothing is current, everything is elsewhere."""
    last = _entry("m1", "a2", "model_usage", activeLeafId=None)
    session = _branched(old, last)
    transcript = OmpAdapter().load_transcript(session)
    assert transcript.current == [] and transcript.branched
    shown = _texts(transcript.messages(BranchMode.APPENDIX))
    assert "first answer" in shown and "second answer" in shown
    tree = transcript.messages(BranchMode.TREE)
    assert [m.text for m in tree if not m.marker] == [
        "the prompt",
        "first answer",
        "retried prompt",
        "second answer",
    ]
    assert OmpAdapter().load(session) == []


def test_without_a_leaf_pointer_the_last_entry_is_the_leaf(old):
    """Pi's rule, which omp keeps for ordinary appends."""
    transcript = OmpAdapter().load_transcript(_branched(old, _assistant("a3", "a2", "third")))
    assert _texts(transcript.current)[-1] == "third"


# --- entries -------------------------------------------------------------------------------


def _render(message: dict) -> Message | None:
    """Render one message entry with a fresh adapter."""
    return OmpAdapter().entry_message(_entry("e1", None, message=message))


def test_developer_turns_are_skipped():
    """Developer instructions are not transcript."""
    assert _render({"role": "developer", "content": "be terse"}) is None


def test_python_runs_render_like_bash_runs():
    """Code on one line as the summary; output and exit code as the text."""
    run = _render(
        {"role": "pythonExecution", "code": "print(1)\nprint(2)", "output": "1\n2\n",
         "exitCode": 0, "cancelled": False}
    )
    assert (run.role, run.tool_summary) == (Role.TOOL, "python: print(1) print(2)")
    assert run.text == "1\n2\n(exit code 0)"
    assert _render({"role": "pythonExecution"}).tool_summary == "python"


def test_file_mentions_list_the_files_without_their_contents():
    """As omp shows them: what was read, never the contents."""
    mention = _render(
        {
            "role": "fileMention",
            "files": [
                {"path": "src/a.py", "content": "SECRET CONTENTS", "lineCount": 12},
                {"path": "logo.png", "content": "", "image": {"type": "image", "data": "x"}},
                {"path": "huge.bin", "skippedReason": "binary"},
                {"path": "one.txt", "content": "x", "lineCount": 1},
            ],
        }
    )
    assert mention.role is Role.OTHER
    assert mention.text == (
        "Read src/a.py (12 lines)\nRead logo.png (image)\n"
        "Read huge.bin (skipped: binary)\nRead one.txt (1 line)"
    )
    assert _render({"role": "fileMention", "files": "nonsense"}) is None


def test_blob_references_show_as_images_and_the_blob_store_is_not_read(sandbox):
    """``blob:sha256:<hex>`` is a pointer into a shared store sx never opens."""
    blob = "blob:sha256:" + "ab" * 32
    message = _render(
        {"role": "user", "content": [{"type": "text", "text": "look"},
                                     {"type": "image", "data": blob, "mimeType": "image/png"}]}
    )
    assert message.text == "look\n[image]"
    assert not (sandbox / ".omp" / "agent" / "blobs").exists()


def test_omps_model_change_holds_one_model_string():
    """``provider/id``, with a non-default role; Pi's two fields still work."""
    adapter = OmpAdapter()
    plain = adapter.entry_message(_entry("m", None, "model_change", model="openai/gpt-4o"))
    assert (plain.role, plain.text) == (Role.SYSTEM, "model → openai/gpt-4o")
    role = adapter.entry_message(
        _entry("m", None, "model_change", model="anthropic/claude", role="smol")
    )
    assert role.text == "model → anthropic/claude (smol)"
    pi_style = adapter.entry_message(
        _entry("m", None, "model_change", provider="openai", modelId="gpt-5")
    )
    assert pi_style.text == "model → openai/gpt-5"


@pytest.mark.parametrize(
    "entry_type",
    [
        "model_usage",
        "service_tier_change",
        "reset_boundary",
        "title_change",
        "ttsr_injection",
        "credential_pin",
        "session_init",
        "mode_change",
    ],
)
def test_omps_bookkeeping_entries_are_skipped(entry_type):
    """Usage, tiers, titles, rules, credentials, prompts and modes are not turns."""
    entry = _entry("x", None, entry_type, summary="not shown", title="t", model="m")
    assert OmpAdapter().entry_message(entry) is None


# --- leases --------------------------------------------------------------------------------


def test_a_missing_lease_file_is_free_but_a_missing_lease_directory_is_unknown(sandbox):
    """omp creates its lease directory before it first writes: without one, sx looks in the wrong place."""
    adapter = OmpAdapter()
    assert adapter.lease_state("S1") == LEASE_FREE  # the directory is there, the file is not
    _lease_dir(sandbox).rmdir()
    assert adapter.lease_state("S1") == LEASE_UNKNOWN


def test_a_lease_flocked_from_another_open_file_is_held(sandbox):
    """The way omp holds it on macOS: an exclusive ``flock``."""
    lease = _lease_dir(sandbox) / "S1.lock"
    adapter = OmpAdapter()
    with _flocked(lease):
        assert adapter.lease_state("S1") == LEASE_HELD
    assert adapter.lease_state("S1") == LEASE_FREE


def test_a_lease_lockf_held_by_another_process_is_held(sandbox):
    """A POSIX lock held elsewhere counts as well."""
    lease = _lease_dir(sandbox) / "S1.lock"
    with _lockf_in_child(lease):
        assert OmpAdapter().lease_state("S1") == LEASE_HELD
    assert OmpAdapter().lease_state("S1") == LEASE_FREE


def test_the_probe_releases_what_it_takes_and_changes_nothing(sandbox):
    """After a probe, omp can still take the lease either way; the file is untouched."""
    lease = _lease_dir(sandbox) / "S1.lock"
    lease.write_text("123:456\n", encoding="utf-8")
    _age(lease)
    before = lease.stat()

    assert OmpAdapter().lease_state("S1") == LEASE_FREE

    fd = os.open(lease, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)
    taker = (
        "import fcntl, os, sys\n"
        "fcntl.lockf(os.open(sys.argv[1], os.O_RDWR), fcntl.LOCK_EX | fcntl.LOCK_NB)"
    )
    subprocess.run([sys.executable, "-I", "-c", taker, str(lease)], check=True, timeout=10)
    after = lease.stat()
    assert lease.read_text(encoding="utf-8") == "123:456\n"
    assert (after.st_ino, after.st_mtime_ns, after.st_size) == (
        before.st_ino,
        before.st_mtime_ns,
        before.st_size,
    )


@pytest.mark.parametrize("sid", ["", "a.b", "a/b", "../x", "x" * 129, "with space", "ü"])
def test_ids_omp_would_hash_are_unknown(sandbox, sid):
    """Oh My Pi files those under a hashed name sx cannot compute."""
    _lease_dir(sandbox)
    assert OmpAdapter().lease_state(sid) == LEASE_UNKNOWN


def test_leases_are_unknown_where_omp_keeps_no_lease_file(sandbox, monkeypatch, old):
    """On Linux omp leases an abstract socket; finding no file proves nothing."""
    path = _session(old)
    monkeypatch.setattr(omp_module, "_LEASES_ARE_FILES", False)
    adapter = OmpAdapter()
    _lease_dir(sandbox)

    assert adapter.lease_state("S1") == LEASE_UNKNOWN
    assert any("abstract sockets" in w for w in adapter.store_warnings())
    assert "cannot tell" in adapter.delete(_only(adapter)).refused[path] and path.exists()


def test_an_unsafe_lease_directory_or_file_is_unknown(sandbox, tmp_path):
    """Oh My Pi refuses a lease directory others can enter; sx refuses to guess."""
    directory = _lease_dir(sandbox)
    adapter = OmpAdapter()
    directory.chmod(0o755)
    assert adapter.lease_state("S1") == LEASE_UNKNOWN
    directory.chmod(0o700)
    target = tmp_path / "elsewhere.lock"
    target.touch()
    (directory / "S1.lock").symlink_to(target)
    assert adapter.lease_state("S1") == LEASE_UNKNOWN
    (directory / "S2.lock").mkdir()
    assert adapter.lease_state("S2") == LEASE_UNKNOWN


def test_a_held_lease_makes_the_session_live_and_blocks_delete_and_move(sandbox, old, new):
    """Liveness is the lease, not just the mtime."""
    path = _session(old)
    adapter = OmpAdapter()
    session = _only(adapter)
    before = path.read_bytes()
    assert adapter.last_activity(session) == file_mtime(path)

    with _flocked(_lease_dir(sandbox) / "S1.lock"):
        last = adapter.last_activity(session)
        assert last is not None and abs((datetime.now() - last).total_seconds()) < 60
        deleted = adapter.delete(session)
        plan = adapter.plan_move([session], old, new)

    assert path.read_bytes() == before
    assert "has this session open" in deleted.refused[path]
    assert "has this session open" in plan.blocked[path]
    assert not plan.rewrites and not plan.relocations


def test_unknown_liveness_counts_as_live(sandbox, old):
    """A session whose lease cannot be checked is treated as open."""
    path = _bucket(old) / "odd-name.jsonl"
    _write(path, [_header("odd.id", str(old))], slot=_slot())
    _age(path)
    adapter = OmpAdapter()
    session = _only(adapter)
    last = adapter.last_activity(session)
    assert last is not None and abs((datetime.now() - last).total_seconds()) < 60
    assert "cannot tell" in adapter.delete(session).refused[path]


# --- delete -----------------------------------------------------------------------------


def test_delete_removes_the_session_and_nothing_shared(sandbox, old):
    """Transcript, ``S/`` and backups — as omp's own delete — plus stale temp files."""
    agent = sandbox / ".omp" / "agent"
    path = _session(old)
    bucket = path.parent
    artifacts = bucket / path.stem
    (artifacts / "tool.log").parent.mkdir()
    (artifacts / "tool.log").write_text("output", encoding="utf-8")
    _write(artifacts / "subagent.jsonl", [_header("SUB", str(old))], slot=_slot())
    backup = bucket / f"{path.name}.0123456789abcdef.bak"
    backup.write_bytes(b"older copy")
    stale_temp = bucket / f".{path.name}.0123456789abcdef.tmp"
    stale_temp.write_bytes(b"crashed rewrite")
    _age(stale_temp, seconds=HOUR)
    fresh_temp = bucket / f".{path.name}.fedcba9876543210.tmp"
    fresh_temp.write_bytes(b"in flight")
    locks = [bucket / f".{path.name}.lock.os", bucket / f"{path.name}.lock"]
    for lock in locks:
        lock.touch()
        _age(lock)
    other = _session(old, "S2")
    other_backup = bucket / f"{other.name}.0123456789abcdef.bak"
    other_backup.write_bytes(b"theirs")
    restorable = bucket / f"{_name('S3')}.0123456789abcdef.bak"
    restorable.write_bytes(b"a deleted session omp would restore")
    shared = [
        agent / "history.db",
        agent / "agent.db",
        agent / "session-pins.json",
        agent / "blobs" / ("ab" * 32),
        agent / "terminal-sessions" / "tty1",
        agent / "custom-session-files" / "marker",
        sandbox / ".omp" / "stats.db",
        _memories(sandbox) / memory_folder_name(str(old)) / "MEMORY.md",
    ]
    for item in shared:
        item.parent.mkdir(parents=True, exist_ok=True)
        item.write_text("keep", encoding="utf-8")
    lease = _lease_dir(sandbox) / "S1.lock"
    lease.touch()
    adapter = OmpAdapter()
    session = next(s for s in adapter.discover() if s.session_id == "S1")

    result = adapter.delete(session)

    assert result.removed == [backup, path, artifacts, stale_temp]
    assert result.refused == {}
    for survivor in [fresh_temp, *locks, other, other_backup, restorable, lease, *shared]:
        assert survivor.exists(), survivor
    assert [s.session_id for s in adapter.discover()] == ["S2"]


def test_correlated_paths_are_built_from_the_transcripts_own_name(old):
    """Every ``S.jsonl.*.bak``, as omp's delete takes them; never a lock file."""
    path = _session(old)
    bucket = path.parent
    artifacts = bucket / path.stem
    artifacts.mkdir()
    wanted = [
        bucket / f"{path.name}.abc.bak",
        bucket / f"{path.name}..bak",
        bucket / f"{path.name}.bak",
        bucket / f"{path.name}.a.b.bak",
        bucket / f"{path.name}.unpack-0123456789abcdef.bak",
    ]
    unwanted = [
        bucket / f"x{path.name}.abc.bak",
        bucket / f"{path.stem}.unpack-0123456789abcdef.bak",
        bucket / f".{path.name}.0123456789ABCDEF.tmp",
        bucket / f".{path.name}.lock.os",
    ]
    for item in wanted + unwanted:
        item.write_bytes(b"x")
        _age(item)
    adapter = OmpAdapter()
    session = _only(adapter)

    found = adapter.correlated_paths(session)

    assert sorted(found) == sorted([artifacts, *wanted])
    assert all(path.stem in item.name for item in found)
    assert adapter.session_sidecars(session) == [artifacts, *sorted(wanted)]


def test_a_dry_run_deletes_nothing(old):
    """The preview lists the same targets and removes none."""
    path = _session(old)
    (path.parent / path.stem).mkdir()
    adapter = OmpAdapter()

    result = adapter.delete(_only(adapter), dry_run=True)

    assert result.dry_run and path in result.removed and path.exists()
    assert (path.parent / path.stem).is_dir()


@pytest.mark.parametrize(
    ("problem", "reason"),
    [
        ("unknown first line", "format may have changed"),
        ("slot version 2", "format may have changed"),
        ("different id", "different session"),
        ("publish lock", "is writing this session"),
        ("fresh os lock", "is writing this session"),
        ("fresh delete lock", "is writing this session"),
        ("sub-agent lease", "sub-agent"),
    ],
)
def test_delete_refuses_what_it_cannot_vouch_for(sandbox, old, problem, reason):
    """The file stays exactly as it was."""
    path = _session(old)
    adapter = OmpAdapter()
    session = _only(adapter)
    if problem == "unknown first line":
        path.write_bytes(b'{"type":"banner"}\n' + path.read_bytes())
    elif problem == "slot version 2":
        path.write_bytes(path.read_bytes().replace(b'"v":1', b'"v":2', 1))
    elif problem == "different id":
        _write(path, [_header("S9", str(old))], slot=_slot())
    elif problem == "publish lock":
        lock = path.parent / f".{path.name}.lock"
        lock.write_text("123:456\n", encoding="utf-8")
        _age(lock, seconds=DAY)  # however old: omp removes it after every publish
    elif problem == "fresh os lock":
        (path.parent / f".{path.name}.lock.os").touch()
    elif problem == "fresh delete lock":
        (path.parent / f"{path.name}.lock").touch()
    else:
        _write(path.parent / path.stem / "sub.jsonl", [_header("SUB", str(old))], slot=_slot())
    _age(path)
    before = path.read_bytes()
    held = _lease_dir(sandbox) / "SUB.lock" if problem == "sub-agent lease" else None

    with _flocked(held) if held else nullcontext():
        result = adapter.delete(session)

    assert path.read_bytes() == before
    assert result.failed and reason in result.refused[path]


@pytest.mark.parametrize("problem", ["slot size", "version"])
def test_an_unknown_format_refuses_delete_as_well_as_move(old, new, problem):
    """A title slot that is not 256 bytes, or a format version omp never wrote: both refused."""
    path = _bucket(old) / _name("S1")
    if problem == "slot size":
        _write(path, [_header("S1", str(old))], slot=_slot(size=300))
    else:
        _write(path, [_header("S1", str(old), version=4)], slot=_slot())
    _age(path)
    before = path.read_bytes()
    adapter = OmpAdapter()
    session = _only(adapter)

    deleted = adapter.delete(session)
    plan = adapter.plan_move([session], old, new)

    expected = "not 256 bytes" if problem == "slot size" else "version 4"
    assert deleted.failed and expected in deleted.refused[path]
    assert expected in plan.blocked[path] and not plan.relocations
    assert path.read_bytes() == before


# --- orphans -------------------------------------------------------------------------------


def _stale_artifacts(bucket: Path, sid: str = "S9") -> Path:
    """Write an artifacts directory whose transcript is gone, untouched for two days."""
    artifacts = bucket / _name(sid)[: -len(".jsonl")]
    (artifacts / "logs").mkdir(parents=True)
    (artifacts / "logs" / "bash.log").write_text("old output", encoding="utf-8")
    _age_tree(artifacts, seconds=2 * DAY)
    return artifacts


def test_stale_artifacts_are_offered_on_their_own(old):
    """The session is gone, nothing can restore it, and nothing touched it for a day."""
    _session(old)
    artifacts = _stale_artifacts(_bucket(old))
    adapter = OmpAdapter()

    orphans = adapter.find_orphans()

    assert [(o.kind, o.paths) for o in orphans] == [(OrphanKind.STALE_SESSION, [artifacts])]
    assert adapter.delete_orphan(orphans[0]).removed == [artifacts]


def test_recent_artifacts_are_not_offered(old):
    """A new session writes artifacts before its first transcript line."""
    _session(old)
    artifacts = _stale_artifacts(_bucket(old))
    (artifacts / "logs" / "new.log").write_text("just now", encoding="utf-8")
    assert OmpAdapter().find_orphans() == []


@pytest.mark.parametrize("keeper", ["draft", "backup"])
def test_a_draft_or_a_backup_keeps_its_bucket(old, keeper):
    """An unsent draft, or a session omp would restore, is not debris."""
    bucket = _bucket(old)
    artifacts = _stale_artifacts(bucket)
    if keeper == "draft":
        (artifacts / ".draft-only-session").touch()
        (artifacts / "draft.txt").write_text("half a prompt", encoding="utf-8")
        _age_tree(artifacts, seconds=2 * DAY)
    else:
        (bucket / f"{artifacts.name}.jsonl.0123456789abcdef.bak").write_bytes(b"restorable")
    assert OmpAdapter().find_orphans() == []


def test_backups_are_never_offered_on_their_own(old):
    """Even beside other sessions, a backup only ever goes with its session."""
    path = _session(old)
    (path.parent / f"{_name('S8')}.0123456789abcdef.bak").write_bytes(b"restorable")
    assert OmpAdapter().find_orphans() == []


def test_an_empty_bucket_is_offered_with_what_it_holds(sandbox, old):
    """A bucket with nothing in it, or only debris, goes whole."""
    bucket = _bucket(old)
    bucket.mkdir(parents=True)
    adapter = OmpAdapter()
    (orphan,) = adapter.find_orphans()
    assert (orphan.kind, orphan.paths, orphan.reason) == (
        OrphanKind.EMPTY,
        [bucket],
        "empty project folder",
    )

    temp = bucket / f".{_name('S1')}.0123456789abcdef.tmp"
    temp.write_bytes(b"crashed rewrite")
    _age_tree(bucket, HOUR)
    (orphan,) = adapter.find_orphans()
    assert "1 file(s)" in orphan.reason
    assert adapter.delete_orphan(orphan).removed == [bucket] and not bucket.exists()


def test_a_bucket_omp_is_writing_in_is_not_offered(old):
    """A fresh lock file or rewrite temp file means a session is on its way."""
    bucket = _bucket(old)
    bucket.mkdir(parents=True)
    (bucket / f".{_name('S1')}.lock.os").touch()
    assert OmpAdapter().find_orphans() == []


def test_lock_files_are_never_removed_by_cleanup(sandbox, old):
    """Only lock files left: not offered. More than that: emptied around them."""
    bucket = _bucket(old)
    bucket.mkdir(parents=True)
    locks = [bucket / f".{_name('S1')}.lock.os", bucket / f"{_name('S1')}.lock"]
    for lock in locks:
        lock.touch()
        _age(lock)
    adapter = OmpAdapter()
    assert adapter.find_orphans() == []

    junk = bucket / "notes.txt"
    junk.write_text("left behind", encoding="utf-8")
    _age_tree(bucket, HOUR)
    (orphan,) = adapter.find_orphans()
    assert orphan.kind is OrphanKind.EMPTY and "2 lock file(s) stay" in orphan.reason

    result = adapter.delete_orphan(orphan)

    assert result.removed == [junk] and all(lock.exists() for lock in locks)
    assert adapter.find_orphans() == []
    direct = adapter.delete_paths([locks[0]])
    assert locks[0].exists() and "lock file" in direct.refused[locks[0]]


def test_a_dead_projects_bucket_is_emptied_around_its_lock_files(sandbox, tmp_path):
    """The sessions go, as a session delete would take them; the lock files stay."""
    gone = tmp_path / "sx-gone" / "proj"
    path = _session(gone, "D")
    (path.parent / path.stem).mkdir()
    lock = path.parent / f".{path.name}.lock.os"
    lock.touch()
    _age_tree(path.parent, HOUR)
    adapter = OmpAdapter()
    (orphan,) = adapter.find_orphans()
    assert orphan.kind is OrphanKind.DEAD_PROJECT and "1 lock file(s) stay" in orphan.reason

    result = adapter.delete_orphan(orphan)

    assert sorted(result.removed) == sorted([path, path.parent / path.stem])
    assert lock.exists() and list(adapter.discover()) == []


def test_every_orphan_lies_inside_the_store(sandbox, old, tmp_path):
    """Fragments, stale artifacts, empty and dead buckets: all deletable by the guard."""
    _session(old)
    _stale_artifacts(_bucket(old))
    (_sessions(sandbox) / "-empty").mkdir(parents=True)
    _write(_sessions(sandbox) / "-frag" / _name("F"), [_user("u", "p", "torn")], slot=None)
    gone = tmp_path / "sx-gone" / "proj"
    _session(gone, "D")
    _age_tree(_sessions(sandbox), HOUR)
    _age_tree(_bucket(old) / _name("S9")[: -len(".jsonl")], 2 * DAY)
    adapter = OmpAdapter()

    orphans = adapter.find_orphans()

    kinds = sorted(o.kind.value for o in orphans)
    assert kinds == ["dead_project", "empty", "stale_session", "stray_temp"]
    roots = adapter.store_roots()
    assert all(is_within(path, roots) for orphan in orphans for path in orphan.paths)
    for orphan in orphans:
        result = adapter.delete_orphan(orphan, dry_run=True)
        assert result.refused == {} and result.removed == orphan.paths, orphan


def test_an_orphan_that_came_back_is_refused(sandbox, old):
    """Every path is checked again at the moment of deletion, by its fingerprint and by omp's rules."""
    _session(old)
    artifacts = _stale_artifacts(_bucket(old))
    adapter = OmpAdapter()
    (stale,) = adapter.find_orphans()
    _session(old, "S9")
    assert "exists again" in adapter.delete_orphan(stale).refused[artifacts]

    empty = _sessions(sandbox) / "-empty"
    empty.mkdir()
    _age(empty)
    (orphan,) = [o for o in adapter.find_orphans() if o.paths == [empty]]
    _age(_write(empty / _name("N"), [_header("N", str(old))], slot=_slot()))
    _age(empty)
    assert "changed since it was listed" in adapter.delete_orphan(orphan).refused[empty]
    unprinted = replace(orphan, fingerprints=None)  # omp's own re-check, without the scan's
    assert "now holds a session" in adapter.delete_orphan(unprinted).refused[empty]
    assert empty.is_dir()


def test_a_fragment_omp_is_writing_is_refused(sandbox, old):
    """A header-less file whose session omp holds stays put."""
    fragment = _write(_bucket(old) / _name("F"), [_user("u", "p", "torn")], slot=None)
    _age_tree(fragment.parent, HOUR)
    adapter = OmpAdapter()
    (orphan,) = adapter.find_orphans()
    with _flocked(_lease_dir(sandbox) / "F.lock"):
        result = adapter.delete_orphan(orphan)
    assert fragment.exists() and "has this session open" in result.refused[fragment]
    assert adapter.delete_orphan(orphan).removed == [fragment]


# --- sessions omp gc archived, and other session data sx cannot show ---------------------


def _archive(project: Path | str, sid: str, *, in_bucket: bool = False) -> Path:
    """Write an archived transcript where ``omp gc`` puts one.

    Args:
        project: The session's project directory.
        sid: Its session id.
        in_bucket: Put it beside the bucket's transcripts instead, as a session
            compressed in place by something other than omp 18.15 would be.

    Returns:
        The archive.
    """
    adapter = OmpAdapter()
    bucket = adapter.folder_for(str(project))
    folder = adapter.sessions_dir() / bucket if in_bucket else adapter.archive_dir() / bucket
    folder.mkdir(parents=True, exist_ok=True)
    body = _slot("archived") + _compact(_header(sid, str(project))) + b"\n"
    path = folder / f"{_name(sid)}.gz"
    path.write_bytes(gzip.compress(body))
    return path


def test_gc_archives_live_outside_the_store(sandbox, tmp_path, monkeypatch, old):
    """Beside the sessions directory, never inside a store root; sx notes them."""
    adapter = OmpAdapter()
    assert adapter.archive_dir() == sandbox / ".omp" / "agent" / "archive" / "sessions"
    assert not any(
        is_within(adapter.archive_dir(), [root]) for root in adapter.store_roots()
    )
    assert adapter.store_notes() == []

    archive = _archive(old, "S1")
    _bucket(old).mkdir(parents=True)

    assert any("archived sessions" in note for note in adapter.store_notes())
    assert list(adapter.discover()) == []
    assert all(
        not is_within(path, [adapter.archive_dir()])
        for orphan in adapter.find_orphans()
        for path in orphan.paths
    )
    assert archive.exists()

    xdg = tmp_path / "xdg-data"
    (xdg / "omp").mkdir(parents=True)
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg))
    assert OmpAdapter().archive_dir() == xdg / "omp" / "archive" / "sessions"
    monkeypatch.delenv("XDG_DATA_HOME")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "custom" / "agent"))
    assert OmpAdapter().archive_dir() == tmp_path / "custom" / "agent" / "archive" / "sessions"


def test_a_bucket_gc_emptied_is_offered_and_its_archives_are_untouched(old):
    """The gc moved the session out; the empty bucket can go, the archive stays."""
    archive = _archive(old, "S1")
    bucket = _bucket(old)
    bucket.mkdir(parents=True)
    _age(bucket)
    adapter = OmpAdapter()

    (orphan,) = adapter.find_orphans()

    assert (orphan.kind, orphan.paths) == (OrphanKind.EMPTY, [bucket])
    assert "archived sessions" in orphan.reason and "not affected" in orphan.reason
    assert adapter.delete_orphan(orphan).removed == [bucket]
    assert archive.exists()


def test_artifacts_of_an_archived_session_are_kept(old):
    """The gc stopped between archiving a transcript and moving its ``S/``."""
    _session(old, "LIVE")
    artifacts = _stale_artifacts(_bucket(old), "S9")
    adapter = OmpAdapter()
    (stale,) = adapter.find_orphans()
    assert stale.paths == [artifacts]

    _archive(old, "S9")

    assert adapter.find_orphans() == []
    result = adapter.delete_orphan(stale)
    assert artifacts.is_dir() and "archived" in result.refused[artifacts]


@pytest.mark.parametrize("kind", ["empty", "dead"])
def test_a_bucket_holding_an_archive_is_never_offered_whole(tmp_path, old, kind):
    """Not empty, not dead: its stale artifacts are still offered on their own."""
    project = old if kind == "empty" else tmp_path / "sx-gone" / "proj"
    if kind == "dead":
        _session(project, "D")
    archive = _archive(project, "A1", in_bucket=True)
    bucket = archive.parent
    adapter = OmpAdapter()
    assert adapter.find_orphans() == []

    leftover = _stale_artifacts(bucket, "S9")
    _age(bucket)
    (orphan,) = adapter.find_orphans()

    assert (orphan.kind, orphan.paths) == (OrphanKind.STALE_SESSION, [leftover])
    assert adapter.delete_orphan(orphan).removed == [leftover]
    assert archive.exists() and bucket.is_dir()


@pytest.mark.parametrize("kind", ["empty", "dead"])
def test_deleting_a_whole_bucket_checks_for_archives_again(tmp_path, old, kind):
    """An archive that appeared after the offer keeps the bucket."""
    project = old if kind == "empty" else tmp_path / "sx-gone" / "proj"
    if kind == "dead":
        _session(project, "D")
    else:
        _bucket(project).mkdir(parents=True)
    _age_tree(_bucket(project), HOUR)
    adapter = OmpAdapter()
    (orphan,) = adapter.find_orphans()
    assert orphan.kind is (OrphanKind.EMPTY if kind == "empty" else OrphanKind.DEAD_PROJECT)

    archive = _archive(project, "A1", in_bucket=True)
    _age_tree(_bucket(project), HOUR)
    result = adapter.delete_orphan(orphan)
    unprinted = adapter.delete_orphan(replace(orphan, fingerprints=None))

    assert result.failed and "changed since it was listed" in result.refused[orphan.paths[0]]
    assert unprinted.failed and "archived session" in unprinted.refused[orphan.paths[0]]
    assert archive.exists()


def test_a_dead_bucket_holding_a_backup_is_not_offered_whole(tmp_path):
    """A backup omp would restore is a session sx does not list."""
    gone = tmp_path / "sx-gone" / "proj"
    path = _session(gone, "D")
    backup = path.parent / f"{_name('R')}.0123456789abcdef.bak"
    backup.write_bytes(b"a session omp would restore")
    assert OmpAdapter().find_orphans() == []


def test_leftovers_never_take_an_archive(sandbox, old):
    """Offered without the bucket, and checked again at deletion time."""
    _session(old)
    archive = _archive(old, "A1", in_bucket=True)
    adapter = OmpAdapter()
    assert adapter.project_leftovers(str(old)) is None

    archive.unlink()
    session = _only(adapter)
    leftovers = adapter.project_leftovers(str(old))
    assert leftovers.paths == [session.primary_path.parent]
    adapter.delete(session)
    archive = _archive(old, "A1", in_bucket=True)

    result = adapter.delete_project_leftovers(leftovers)

    assert archive.exists() and "archived session" in result.refused[archive.parent]


def test_what_omp_session_unpack_sets_aside_is_kept(old):
    """``S.jsonl.unpack-<id>.bak`` and the directory ``S.unpack-<id>.bak``."""
    path = _session(old)
    bucket = path.parent
    set_aside = bucket / f"{path.stem}.unpack-0123456789abcdef.bak"
    (set_aside / "logs").mkdir(parents=True)
    _age_tree(set_aside, seconds=2 * DAY)
    adapter = OmpAdapter()
    assert adapter.find_orphans() == []

    adapter.delete(_only(adapter))
    (bucket / f"{path.name}.unpack-0123456789abcdef.bak").write_bytes(b"set aside")
    assert adapter.find_orphans() == []
    assert set_aside.is_dir()


def test_a_gc_compress_scratch_dir_counts_as_writing(old):
    """A fresh one means gc is swapping in compressed copies right now."""
    bucket = _bucket(old)
    scratch = bucket / ".omp-gc-compress-a1b2c3"
    scratch.mkdir(parents=True)
    (scratch / ".omp-gc-compress-owner").touch()
    adapter = OmpAdapter()
    assert adapter.find_orphans() == []

    _age_tree(scratch, seconds=HOUR)
    (orphan,) = adapter.find_orphans()
    assert (orphan.kind, orphan.paths) == (OrphanKind.EMPTY, [bucket])


def test_a_session_gc_also_archived_keeps_its_artifacts(sandbox, old, new):
    """The gc stopped before removing the original: both copies share ``S/``."""
    path = _session(old)
    artifacts = path.parent / path.stem
    (artifacts / "tool.log").parent.mkdir()
    (artifacts / "tool.log").write_text("output", encoding="utf-8")
    archive = _archive(old, "S1")
    adapter = OmpAdapter()
    session = _only(adapter)

    plan = adapter.plan_move([session], old, new)
    assert "archived copy" in plan.blocked[path] and not plan.relocations

    preview = adapter.delete(session, dry_run=True)
    assert artifacts not in preview.removed and "archived copy" in preview.skipped[artifacts]
    result = adapter.delete(session)

    assert result.removed == [path] and not path.exists()
    assert (artifacts / "tool.log").exists() and archive.exists()
    assert adapter.find_orphans() == []


# --- move ------------------------------------------------------------------------------------


def test_move_relocates_the_bucket_and_rewrites_only_the_header(sandbox, old, new):
    """Slot byte-identical; ``cwd`` re-pointed; the old path recorded; files travel."""
    path = _session(old, title="My title")
    artifacts = path.parent / path.stem
    artifacts.mkdir()
    (artifacts / "log.txt").write_text("tool output", encoding="utf-8")
    backup = path.parent / f"{path.name}.0123456789abcdef.bak"
    backup.write_bytes(b"backup")
    original = _lines(path)
    adapter = OmpAdapter()

    _plan, result = _move(adapter, old, new)

    destination = _sessions(sandbox) / "-work-renamed"
    moved = destination / path.name
    lines = _lines(moved)
    assert lines[0] == original[0] and len(lines[0]) == SLOT_BYTES
    header = json.loads(lines[1])
    assert header["cwd"] == str(new)
    assert header["previousSessionFiles"] == [str(path)]
    assert lines[2:] == original[2:]
    assert (destination / path.stem / "log.txt").read_text(encoding="utf-8") == "tool output"
    assert (destination / backup.name).read_bytes() == b"backup"
    assert not path.parent.exists()
    assert result.fields_updated == 1 and result.refused == {}
    assert _only(adapter).project_path == str(new)


def test_a_bucket_holding_lock_files_moves_around_them(sandbox, old, new):
    """Every session file travels; omp's lock files stay where omp made them."""
    path = _session(old)
    artifacts = path.parent / path.stem
    artifacts.mkdir()
    locks = [path.parent / f".{path.name}.lock.os", path.parent / f"{path.name}.lock"]
    for lock in locks:
        lock.touch()
        _age(lock)
    adapter = OmpAdapter()

    _plan, result = _move(adapter, old, new)

    destination = _sessions(sandbox) / "-work-renamed"
    assert {entry.name for entry in destination.iterdir()} == {path.name, path.stem}
    assert all(lock.exists() for lock in locks) and not path.exists()
    assert result.refused == {}
    assert json.loads(_lines(destination / path.name)[1])["previousSessionFiles"] == [str(path)]


def test_a_dry_run_moves_nothing(old, new):
    """The plan is reported and the store left alone."""
    path = _session(old)
    before = path.read_bytes()
    _plan, result = _move(OmpAdapter(), old, new, dry_run=True)
    assert result.dry_run and result.moved and path.read_bytes() == before


def test_a_session_relocated_alone_takes_its_artifacts_and_backups(sandbox, old, new):
    """A bucket shared with a colliding project moves session by session."""
    mine = _session(old)
    (mine.parent / mine.stem).mkdir()
    backup = mine.parent / f"{mine.name}.0123456789abcdef.bak"
    backup.write_bytes(b"backup")
    theirs_project = Path(str(old).replace("work/app", "work-app"))
    theirs = _write(mine.parent / _name("T"), [_header("T", str(theirs_project))], slot=_slot())
    _age(theirs)
    adapter = OmpAdapter()

    _plan, result = _move(adapter, old, new)

    destination = _sessions(sandbox) / "-work-renamed"
    assert {path.name for path in destination.iterdir()} == {
        mine.name,
        mine.stem,
        backup.name,
    }
    assert theirs.exists() and not mine.exists() and not backup.exists()
    assert result.refused == {}


def test_a_busy_session_pins_its_bucket(sandbox, old, new):
    """The open session stays byte for byte; the other moves out alone."""
    chosen = _session(old, "S1")
    busy = _session(old, "S2")
    before = busy.read_bytes()
    adapter = OmpAdapter()

    with _flocked(_lease_dir(sandbox) / "S2.lock"):
        plan, result = _move(adapter, old, new)

    assert busy.read_bytes() == before and "has this session open" in plan.blocked[busy]
    assert (_sessions(sandbox) / "-work-renamed" / chosen.name).is_file()
    assert not chosen.exists()
    assert "stay put" in (result.note or "")


def test_a_collision_keeps_the_whole_session_in_place(sandbox, old, new):
    """A transcript that cannot move keeps its artifacts and backups with it."""
    path = _session(old)
    artifacts = path.parent / path.stem
    artifacts.mkdir()
    backup = path.parent / f"{path.name}.0123456789abcdef.bak"
    backup.write_bytes(b"backup")
    other = _session(old, "S2")
    squatter = _write(_sessions(sandbox) / "-work-renamed" / path.name, [_header("S1", str(new))])
    adapter = OmpAdapter()

    _plan, result = _move(adapter, old, new)

    assert path.exists() and artifacts.is_dir() and backup.exists()
    assert "already exists" in result.refused[path]
    assert "stays with the rest of its session" in result.refused[artifacts]
    assert "stays with the rest of its session" in result.refused[backup]
    assert (_sessions(sandbox) / "-work-renamed" / other.name).is_file()
    assert squatter.read_bytes().count(b"\n") == 1


def test_a_slot_of_the_wrong_size_refuses_the_move(sandbox, old, new):
    """Rewriting it could shift what omp overwrites in place."""
    path = _bucket(old) / _name("S1")
    _write(path, [_header("S1", str(old)), _user("u1", None, "hi")], slot=_slot(size=300))
    _age(path)
    before = path.read_bytes()
    adapter = OmpAdapter()

    plan, _result = _move(adapter, old, new)

    assert "not 256 bytes" in plan.blocked[path]
    assert path.read_bytes() == before and not plan.relocations


def test_a_header_sx_cannot_rewrite_faithfully_is_refused(old, new):
    """An unknown version or a field of an unexpected shape stays untouched."""
    first = _session(old, "S1", version=4)
    second = _session(old, "S2", previousSessionFiles="/one/path.jsonl")
    adapter = OmpAdapter()

    plan, _result = _move(adapter, old, new)

    assert "version 4" in plan.blocked[first]
    assert "previousSessionFiles" in plan.blocked[second]
    assert not plan.rewrites


def test_additional_directories_change_as_omps_move_to_changes_them():
    """The new ``cwd`` is dropped and an empty list goes; nothing else is re-pointed."""
    adapter = OmpAdapter()
    old, new = Path("/p/old"), Path("/p/new")
    header = _header(cwd="/p/old", additionalDirectories=["/p/old/lib", "/shared", "/p/new"])
    updated = adapter.header_update(header, old, new, moved_from=None)
    assert updated["cwd"] == "/p/new"
    assert updated["additionalDirectories"] == ["/p/old/lib", "/shared"]
    assert header["additionalDirectories"] == ["/p/old/lib", "/shared", "/p/new"]
    kept = _header(cwd="/p/old", additionalDirectories=["/p/old/lib"])
    assert adapter.header_update(kept, old, new, moved_from=None) == {**kept, "cwd": "/p/new"}

    for directories in (["/p/new"], []):
        header = _header(cwd="/p/old", additionalDirectories=directories)
        updated = adapter.header_update(header, old, new, moved_from=None)
        assert "additionalDirectories" not in updated and updated["cwd"] == "/p/new"
    assert adapter.header_update(_header(cwd="/q"), old, new, moved_from=None) is None
    untouched = _header(cwd="/q", additionalDirectories=[])
    assert adapter.header_update(untouched, old, new, moved_from=None) is None


def test_previous_session_files_are_appended_once():
    """Oh My Pi keeps a de-duplicated list, oldest first, of absolute paths."""
    adapter = OmpAdapter()
    header = _header(cwd="/p/old", previousSessionFiles=["/a.jsonl", "/a.jsonl", "/b.jsonl"])
    updated = adapter.header_update(
        header, Path("/p/old"), Path("/p/new"), moved_from=Path("/b.jsonl")
    )
    assert updated["previousSessionFiles"] == ["/a.jsonl", "/b.jsonl"]
    fresh = adapter.header_update(
        _header(cwd="/p/old"), Path("/p/old"), Path("/p/new"), moved_from=Path("/s/x.jsonl")
    )
    assert fresh["previousSessionFiles"] == ["/s/x.jsonl"]


def test_memory_folders_move_with_the_project(sandbox, old, new):
    """One per recorded directory, re-named for the re-pointed path."""
    _session(old, "S1")
    _session(old / "sub", "S2")
    memories = _memories(sandbox)
    top = memories / memory_folder_name(str(old))
    sub = memories / memory_folder_name(str(old / "sub"))
    for folder, document in ((top, "MEMORY.md"), (sub, "learned.md")):
        folder.mkdir(parents=True)
        (folder / document).write_text("remember", encoding="utf-8")
    unrelated = memories / memory_folder_name("/somewhere/else")
    unrelated.mkdir()
    _age_tree(memories, HOUR)
    adapter = OmpAdapter()

    plan, result = _move(adapter, old, new)

    new_top = memories / memory_folder_name(str(new))
    new_sub = memories / memory_folder_name(str(new / "sub"))
    assert (new_top / "MEMORY.md").is_file() and (new_sub / "learned.md").is_file()
    assert not top.exists() and not sub.exists() and unrelated.is_dir()
    assert (top, new_top) in result.moved and (sub, new_sub) in result.moved
    assert "memory folder" in (plan.note or "")


def test_memory_merges_into_an_existing_folder_and_refuses_collisions(sandbox, old, new):
    """What is already there is never overwritten; the rest moves in."""
    _session(old)
    memories = _memories(sandbox)
    source = memories / memory_folder_name(str(old))
    target = memories / memory_folder_name(str(new))
    source.mkdir(parents=True)
    target.mkdir()
    (source / "MEMORY.md").write_text("old index", encoding="utf-8")
    (source / "learned.md").write_text("lessons", encoding="utf-8")
    (target / "MEMORY.md").write_text("new index", encoding="utf-8")
    _age_tree(memories, HOUR)

    _plan, result = _move(OmpAdapter(), old, new)

    assert (target / "MEMORY.md").read_text(encoding="utf-8") == "new index"
    assert (target / "learned.md").read_text(encoding="utf-8") == "lessons"
    assert "already exists" in result.refused[source / "MEMORY.md"]
    assert (source / "MEMORY.md").exists()


def test_an_emptied_memory_folder_is_removed_after_a_merge(sandbox, old, new):
    """Every entry moved in, so the source folder goes."""
    _session(old)
    memories = _memories(sandbox)
    source = memories / memory_folder_name(str(old))
    target = memories / memory_folder_name(str(new))
    source.mkdir(parents=True)
    target.mkdir()
    (source / "learned.md").write_text("lessons", encoding="utf-8")
    _age_tree(memories, HOUR)
    _move(OmpAdapter(), old, new)
    assert not source.exists() and (target / "learned.md").is_file()


def test_memory_stays_while_a_session_for_its_directory_stays(sandbox, old, new):
    """Oh My Pi writes memory for a directory it has a session open in."""
    _session(old)
    folder = _memories(sandbox) / memory_folder_name(str(old))
    folder.mkdir(parents=True)
    with _flocked(_lease_dir(sandbox) / "S1.lock"):
        plan, _result = _move(OmpAdapter(), old, new)
    assert folder.is_dir() and "its memory stays" in plan.blocked[folder]


def test_memory_shared_with_a_colliding_directory_stays(sandbox, old, new):
    """``/x/a-b`` and ``/x/a/b`` share a memory folder; moving one must not take it."""
    project = old / "a" / "b"
    collider = old.parent / "app" / "a-b"
    _session(project, "S1")
    _session(collider, "S2")
    folder = _memories(sandbox) / memory_folder_name(str(project))
    assert folder.name == memory_folder_name(str(collider))
    folder.mkdir(parents=True)
    adapter = OmpAdapter()

    plan = adapter.plan_move(adapter.sessions_for_project(project), project, new)

    assert "encodes alike" in plan.blocked[folder]


def test_every_planned_path_stays_inside_the_store(sandbox, old, new):
    """The guard's invariant, applied to rewrites and relocations."""
    _session(old)
    _session(old / "sub", "SUB")
    (_memories(sandbox) / memory_folder_name(str(old))).mkdir(parents=True)
    _age_tree(_memories(sandbox), HOUR)
    adapter = OmpAdapter()

    plan = adapter.plan_move(adapter.sessions_for_project(old), old, new)

    roots = adapter.store_roots()
    assert len(plan.relocations) == 3
    assert all(is_within(s, roots) and is_within(d, roots) for s, d in plan.relocations)
    assert all(is_within(path, roots) for path in plan.rewrites)


def test_the_move_service_moves_omp_sessions_end_to_end(sandbox, old, new):
    """Through the service, as the TUI and CLI drive it."""
    path = _session(old)
    adapter = OmpAdapter()
    service = MoveService({adapter.name: adapter}, log_path=sandbox / "op.log")

    plans = service.plan(old, new)
    results = service.move(plans)

    assert list(plans) == ["omp"] and plans["omp"].live == []
    assert results["omp"].fields_updated == 1
    moved = _sessions(sandbox) / "-work-renamed" / path.name
    assert json.loads(_lines(moved)[1])["cwd"] == str(new)


# --- a store shared with Pi ------------------------------------------------------------


def test_a_shared_agent_dir_is_listed_once_under_oh_my_pi(sandbox, tmp_path, monkeypatch, old):
    """Both honor ``PI_CODING_AGENT_DIR``; Pi yields, Oh My Pi lists the sessions."""
    agent = tmp_path / "shared" / "agent"
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent))
    path = _session(old)
    pi, omp = PiAdapter(), OmpAdapter()

    assert path.parent.parent == agent / "sessions"
    assert pi.sessions_dir() == omp.sessions_dir() == agent / "sessions"
    assert pi.available() is False
    assert list(pi.discover()) == [] and pi.find_orphans() == []
    assert any("listed under Oh My Pi" in w for w in pi.store_warnings())
    assert omp.available() is True
    assert [s.session_id for s in omp.discover()] == ["S1"]
    assert omp.find_orphans() == []


# --- project leftovers ----------------------------------------------------------------------


def test_leftovers_offer_the_bucket_and_the_memory_folder(sandbox, old):
    """Memory is listed by its documents and removed only through the guard."""
    path = _session(old)
    memories = _memories(sandbox)
    memory = memories / memory_folder_name(str(old))
    (memory / "skills").mkdir(parents=True)
    for document in ("MEMORY.md", "memory_summary.md", "skills/review.md", "state.json"):
        (memory / document).write_text("x", encoding="utf-8")
    _age_tree(memory, seconds=HOUR)
    other = memories / memory_folder_name("/another/project")
    other.mkdir()
    adapter = OmpAdapter()
    session = _only(adapter)

    leftovers = adapter.project_leftovers(str(old))

    assert leftovers.paths == [path.parent, memory]
    assert leftovers.memory_files == [
        memory / "MEMORY.md",
        memory / "memory_summary.md",
        memory / "skills" / "review.md",
    ]
    adapter.delete(session)
    result = adapter.delete_project_leftovers(leftovers)
    assert result.removed == [path.parent, memory] and result.refused == {}
    assert other.is_dir() and memories.is_dir() and _sessions(sandbox).is_dir()


def test_leftover_buckets_are_emptied_around_lock_files(sandbox, old):
    """The last-session cleanup keeps omp's lock files too."""
    path = _session(old)
    lock = path.parent / f".{path.name}.lock.os"
    lock.touch()
    _age(lock)
    adapter = OmpAdapter()
    session = _only(adapter)
    leftovers = adapter.project_leftovers(str(old))
    assert leftovers.paths == [path.parent]

    adapter.delete(session)
    result = adapter.delete_project_leftovers(leftovers)

    assert result.refused == {} and lock.exists() and not path.exists()


def test_leftovers_stay_while_the_project_still_has_a_session(sandbox, old):
    """The offer is made before the delete; nothing goes if the session stays."""
    path = _session(old)
    memory = _memories(sandbox) / memory_folder_name(str(old))
    memory.mkdir(parents=True)
    _age(memory)
    adapter = OmpAdapter()
    leftovers = adapter.project_leftovers(str(old))

    result = adapter.delete_project_leftovers(leftovers)

    assert path.exists() and memory.is_dir()
    assert "still holds a session" in result.refused[path.parent]
    assert "still has a session" in result.refused[memory]


def test_a_bucket_omp_could_restore_a_session_from_is_not_offered(sandbox, old):
    """A backup of a deleted session is a session omp brings back."""
    path = _session(old)
    (path.parent / f"{_name('S8')}.0123456789abcdef.bak").write_bytes(b"restorable")
    assert OmpAdapter().project_leftovers(str(old)) is None


def test_memory_shared_with_a_colliding_project_is_not_offered(sandbox, old):
    """Its documents may be the other project's."""
    project = old / "a" / "b"
    _session(project, "S1")
    _session(old.parent / "app" / "a-b", "S2")
    (_memories(sandbox) / memory_folder_name(str(project))).mkdir(parents=True)
    leftovers = OmpAdapter().project_leftovers(str(project))
    assert leftovers is None or not leftovers.memory_files
    assert leftovers is None or all("memories" not in str(p) for p in leftovers.paths)
