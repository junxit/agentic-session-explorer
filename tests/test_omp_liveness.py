"""Oh My Pi liveness: the lease probe (D9), where the leases are (D5), and the caveat.

Every store is synthetic, under ``tmp_path``; ``omp`` is never run. A running
omp is played by this process or a child holding a ``flock`` or ``lockf`` on
the lease file, as omp does on macOS. The builders and the autouse sandbox
come from :mod:`tests.test_omp_adapter`.
"""

from __future__ import annotations

import fcntl
import os
import struct
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from sx.adapters import omp as omp_module
from sx.adapters.omp import LEASE_FREE, LEASE_HELD, LEASE_UNKNOWN, OmpAdapter
from tests import test_omp_adapter as base

sandbox = base.sandbox
old = base.old


# --- a rival omp -------------------------------------------------------------------

_HOLDER = r"""
import fcntl, os, sys
kind, path = sys.argv[1], sys.argv[2]
fd = os.open(path, os.O_RDWR)
if kind == "flock":
    fcntl.flock(fd, fcntl.LOCK_EX)
elif kind == "flock-shared":
    fcntl.flock(fd, fcntl.LOCK_SH)
else:
    fcntl.lockf(fd, fcntl.LOCK_EX)
print("ready", flush=True)
sys.stdin.read()
"""

_RIVAL = r"""
import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDWR)
print("ready", flush=True)
for _ in sys.stdin:
    outcome = []
    for take, drop in (
        (lambda: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB), lambda: fcntl.flock(fd, fcntl.LOCK_UN)),
        (lambda: fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB), lambda: fcntl.lockf(fd, fcntl.LOCK_UN)),
    ):
        try:
            take()
        except OSError:
            outcome.append("busy")
        else:
            drop()
            outcome.append("ok")
    print(" ".join(outcome), flush=True)
"""

_RACER = r"""
import fcntl, os, sys, time
fd = os.open(sys.argv[1], os.O_RDWR)
print("ready", flush=True)
sys.stdin.readline()
failures = 0
deadline = time.monotonic() + float(sys.argv[2])
while time.monotonic() < deadline:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        failures += 1
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
print(failures, flush=True)
"""


def _child(script: str, *args: str) -> subprocess.Popen:
    """Start a helper process and wait until it says it is ready."""
    child = subprocess.Popen(
        [sys.executable, "-I", "-c", script, *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert child.stdout is not None and child.stdout.readline().strip() == "ready"
    return child


def _stop(child: subprocess.Popen) -> None:
    """Close a helper's input, wait for it to exit, then close its output."""
    assert child.stdin is not None and child.stdout is not None
    child.stdin.close()
    child.wait(timeout=10)
    child.stdout.close()


class _Rival:
    """A second process that tries to take the lease once, as omp's claim does."""

    def __init__(self, lease: Path) -> None:
        """Start the rival on ``lease``."""
        self._child = _child(_RIVAL, str(lease))

    def attempt(self) -> str:
        """Try a non-blocking exclusive ``flock``, then a ``lockf``; report each."""
        assert self._child.stdin is not None and self._child.stdout is not None
        self._child.stdin.write("try\n")
        self._child.stdin.flush()
        return self._child.stdout.readline().strip()

    def close(self) -> None:
        """Stop the rival."""
        _stop(self._child)


class _Watched:
    """Stand-in for the ``fcntl`` module the omp adapter uses.

    After every ``flock``, ``lockf`` or ``fcntl`` call the probe makes — while
    the probe still holds whatever it took — the rival tries to take the
    lease, as omp would if it started writing the session at that moment.
    """

    def __init__(self, real, rival: _Rival, calls: list) -> None:
        """Wrap the real module."""
        self._real, self._rival, self._calls = real, rival, calls

    def __getattr__(self, name: str):
        """Return the real attribute, watching the three lock primitives."""
        value = getattr(self._real, name)
        if name not in ("flock", "lockf", "fcntl"):
            return value

        def watched(*args):
            """Make the call, then let the rival try."""
            outcome = value(*args)
            command = args[1] if len(args) > 1 else None
            self._calls.append((name, command, self._rival.attempt()))
            return outcome

        return watched


@pytest.fixture
def lease(sandbox: Path) -> Path:
    """Return session ``S1``'s lease file, existing and free."""
    path = base._lease_dir(sandbox) / "S1.lock"
    path.write_text("123:456\n", encoding="utf-8")
    return path


# --- D9: the probe asks, and never takes ---------------------------------------------


@pytest.mark.parametrize("kind", ["flock", "flock-shared", "lockf"])
def test_a_holder_in_another_process_reads_held(lease, kind):
    """omp's exclusive flock, a shared one, a POSIX lock: all seen; nobody holding reads free."""
    adapter = OmpAdapter()
    assert adapter.lease_state("S1") == LEASE_FREE
    holder = _child(_HOLDER, kind, str(lease))
    try:
        assert adapter.lease_state("S1") == LEASE_HELD
    finally:
        _stop(holder)
    assert adapter.lease_state("S1") == LEASE_FREE


def test_the_probe_never_makes_a_holders_acquire_fail(lease, monkeypatch):
    """omp claims in one attempt; a probe that held the lease even briefly would fork the session."""
    rival = _Rival(lease)
    calls: list = []
    monkeypatch.setattr(omp_module, "fcntl", _Watched(fcntl, rival, calls))
    try:
        assert OmpAdapter().lease_state("S1") == LEASE_FREE
    finally:
        rival.close()
    assert calls, "the probe asked the kernel nothing"
    assert [attempt for *_, attempt in calls] == ["ok ok"] * len(calls), calls


def test_the_probe_only_asks_with_f_getlk(lease, monkeypatch):
    """No ``flock`` and no ``lockf``: a single non-acquiring query."""
    rival = _Rival(lease)
    calls: list = []
    monkeypatch.setattr(omp_module, "fcntl", _Watched(fcntl, rival, calls))
    try:
        OmpAdapter().lease_state("S1")
    finally:
        rival.close()
    assert [(name, command) for name, command, _ in calls] == [("fcntl", fcntl.F_GETLK)]


def test_probing_while_omp_takes_the_lease_never_fails_omp(lease):
    """A rival hammers non-blocking claims while sx probes: not one claim fails."""
    racer = _child(_RACER, str(lease), "0.6")
    adapter = OmpAdapter()
    assert racer.stdin is not None and racer.stdout is not None
    racer.stdin.write("go\n")
    racer.stdin.flush()
    probes = 0
    while racer.poll() is None and probes < 20_000:
        adapter.lease_state("S1")
        probes += 1
    failures = int(racer.stdout.readline())
    _stop(racer)
    assert probes > 0 and failures == 0


@pytest.mark.skipif(sys.platform != "darwin", reason="the layout is verified on macOS only")
def test_struct_flock_layout_matches_this_kernel(sandbox, lease):
    """A known lock comes back at the offsets the layout gives, in exactly 24 bytes."""
    layout = omp_module._FLOCK_LAYOUT
    assert layout is not None and struct.calcsize(layout) == 24
    setter = (
        "import fcntl, os, struct, sys\n"
        "fd = os.open(sys.argv[1], os.O_RDWR)\n"
        "fcntl.fcntl(fd, fcntl.F_SETLK, struct.pack(sys.argv[2], 100, 7, 0, fcntl.F_WRLCK, 0))\n"
        "print('ready', flush=True)\n"
        "sys.stdin.read()\n"
    )
    holder = _child(setter, str(lease), layout)
    try:
        fd = os.open(lease, os.O_RDONLY)
        try:
            query = struct.pack(layout, 0, 0, 0, fcntl.F_WRLCK, os.SEEK_SET) + b"\xaa" * 40
            answer = fcntl.fcntl(fd, fcntl.F_GETLK, query)
        finally:
            os.close(fd)
    finally:
        _stop(holder)
    assert struct.unpack(layout, answer[:24]) == (100, 7, holder.pid, fcntl.F_WRLCK, os.SEEK_SET)
    assert answer[24:] == b"\xaa" * 40  # the kernel wrote 24 bytes and no more


def test_a_platform_without_a_known_layout_counts_every_lease_unknown(lease, monkeypatch):
    """Where ``struct flock`` was not verified, nothing is guessed."""
    monkeypatch.setattr(omp_module, "_FLOCK_LAYOUT", None)
    adapter = OmpAdapter()
    assert adapter.lease_state("S1") == LEASE_UNKNOWN
    assert any("cannot probe" in warning for warning in adapter.store_warnings())


# --- D5: every place omp may keep its leases ------------------------------------------


def test_a_lease_omp_took_before_xdg_state_home_existed_is_still_found(sandbox, tmp_path, monkeypatch):
    """omp fixes its lease directory at startup; ``omp config init-xdg`` later does not move it."""
    state = tmp_path / "xdg-state"
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
    lease = base._lease_dir(sandbox) / "LIVE.lock"
    with base._flocked(lease):
        (state / "omp").mkdir(parents=True)
        adapter = OmpAdapter()
        assert adapter.lease_dir() == state / "omp" / "run" / "session-owners"
        assert adapter.lease_state("LIVE") == LEASE_HELD


def test_a_missing_lease_directory_next_to_sessions_is_unknown(sandbox, old):
    """omp creates the directory before its first write: without it, sx is looking elsewhere."""
    path = base._session(old)
    base._lease_dir(sandbox).rmdir()
    adapter = OmpAdapter()
    session = base._only(adapter)

    result = adapter.delete(session)

    assert path.exists() and "cannot tell" in result.refused[path]
    assert adapter.lease_state("S1") == LEASE_UNKNOWN
    warning = next(w for w in adapter.store_warnings() if "session leases exists" in w)
    assert ".omp/run/session-owners" in warning and "none can be deleted" in warning


def test_no_missing_lease_warning_without_sessions(sandbox):
    """A store omp never wrote to has no lease directory, and nothing to warn about."""
    base._lease_dir(sandbox).rmdir()
    base._sessions(sandbox).mkdir(parents=True)
    assert OmpAdapter().store_warnings() == []


def test_xdg_state_home_from_dot_env_moves_the_leases(sandbox, old):
    """omp fills unset variables from ``~/.env`` at startup; sx reads them the same way."""
    state = sandbox / "custom-state"
    (sandbox / ".env").write_text(f"# omp\nexport XDG_STATE_HOME={state}  # moved\n", "utf-8")
    leases = state / "omp" / "run" / "session-owners"
    leases.mkdir(parents=True)
    leases.chmod(0o700)
    path = base._session(old)
    adapter = OmpAdapter()

    with base._flocked(leases / "S1.lock"):
        result = adapter.delete(base._only(adapter))
        assert adapter.lease_dir() == leases
        assert adapter.lease_state("S1") == LEASE_HELD

    assert path.exists() and "has this session open" in result.refused[path]
    assert any(
        note.startswith("XDG_STATE_HOME is set in") and ".env, which Oh My Pi loads" in note
        for note in adapter.store_notes()
    )
    assert adapter.memories_dir() == state / "omp" / "memories"


def test_pi_test_session_owners_dir_moves_the_leases(sandbox, old, tmp_path, monkeypatch):
    """Named for omp's tests, honored by every omp process whenever it is set."""
    owners = tmp_path / "owners"
    owners.mkdir(mode=0o700)
    monkeypatch.setenv("PI_TEST_SESSION_OWNERS_DIR", str(owners))
    path = base._session(old)
    adapter = OmpAdapter()

    with base._flocked(owners / "S1.lock"):
        result = adapter.delete(base._only(adapter))
        assert adapter.lease_dir() == owners
        assert adapter.lease_state("S1") == LEASE_HELD

    assert path.exists() and "has this session open" in result.refused[path]


def test_a_lease_dir_named_in_another_dot_env_is_probed_too(sandbox, old, tmp_path, monkeypatch):
    """sx's environment wins for sx, but an omp started without it would follow the file."""
    owners = tmp_path / "from-dotenv"
    owners.mkdir(mode=0o700)
    (sandbox / ".env").write_text(f"OMP_TEST_SESSION_OWNERS_DIR={owners}\n", "utf-8")
    elsewhere = tmp_path / "from-env"
    elsewhere.mkdir(mode=0o700)
    monkeypatch.setenv("PI_TEST_SESSION_OWNERS_DIR", str(elsewhere))
    base._session(old)
    adapter = OmpAdapter()

    with base._flocked(owners / "S1.lock"):
        assert adapter.lease_dir() == elsewhere
        assert adapter.lease_state("S1") == LEASE_HELD


def test_the_session_projects_own_dot_env_is_probed(sandbox, old, tmp_path):
    """``<cwd>/.env`` differs per omp process: the session's project is asked about too."""
    state = tmp_path / "project-state"
    leases = state / "omp" / "run" / "session-owners"
    leases.mkdir(parents=True)
    leases.chmod(0o700)
    (old / ".env").write_text(f'XDG_STATE_HOME="{state}"\n', "utf-8")
    path = base._session(old)
    adapter = OmpAdapter()
    session = base._only(adapter)

    with base._flocked(leases / "S1.lock"):
        result = adapter.delete(session)
        assert path.exists() and "has this session open" in result.refused[path]
        last = adapter.last_activity(session)
        assert last is not None and abs((datetime.now() - last).total_seconds()) < 60
        assert adapter.lease_state("S1") == LEASE_FREE  # without the project sx cannot know
        assert adapter.lease_state("S1", str(old)) == LEASE_HELD


@pytest.mark.parametrize(
    ("content", "warning"),
    [
        ("XDG_STATE_HOME=state\n", "XDG_STATE_HOME is a relative path"),
        ("PI_TEST_SESSION_OWNERS_DIR=owners\n", "PI_TEST_SESSION_OWNERS_DIR is a relative path"),
    ],
)
def test_a_relative_lease_location_in_a_dot_env_makes_liveness_unknown(sandbox, content, warning):
    """omp would resolve it against its own working directory; sx cannot place it."""
    (sandbox / ".env").write_text(content, "utf-8")
    adapter = OmpAdapter()
    assert adapter.lease_state("S1") == LEASE_UNKNOWN
    assert any(warning in w for w in adapter.store_warnings())


def test_an_empty_pi_test_session_owners_dir_means_omp_runs_without_leases(sandbox, monkeypatch):
    """omp's lease directory is then the empty path: it holds no lease at all."""
    monkeypatch.setenv("PI_TEST_SESSION_OWNERS_DIR", "")
    adapter = OmpAdapter()
    assert adapter.lease_dir() is None
    assert adapter.lease_state("S1") == LEASE_UNKNOWN
    assert any("set but empty" in w for w in adapter.store_warnings())


def test_the_conventional_xdg_state_directory_is_probed(sandbox, old):
    """An omp whose environment set ``XDG_STATE_HOME=~/.local/state`` keeps its leases there."""
    leases = sandbox / ".local" / "state" / "omp" / "run" / "session-owners"
    leases.mkdir(parents=True)
    leases.chmod(0o700)
    with base._flocked(leases / "S1.lock"):
        assert OmpAdapter().lease_state("S1") == LEASE_HELD


# --- the caveat ----------------------------------------------------------------------


def test_the_confirmations_say_to_close_the_session_in_omp_first():
    """omp holds its lease only while it writes, so an open session may look idle."""
    caveat = OmpAdapter.open_session_caveat
    assert caveat is not None
    for phrase in ("only while it writes", "close a session in Oh My Pi", "back whole",
                   "two places", "header-less file"):
        assert phrase in caveat
