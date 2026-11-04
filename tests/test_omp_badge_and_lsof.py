"""The live badge's cache, its time limits, and the names ``lsof`` prints.

The TUI asks :meth:`~sx.adapters.omp.OmpAdapter.last_activity` about every
Oh My Pi session it shows, on its own thread. A scan of the running omp
processes, the store and every ``.env`` file are reused for a few seconds
from when they were finished, and ``ps`` and ``lsof`` get a short time limit
there; a delete still looks afresh. ``lsof`` escapes some characters in the
names it prints, and those are undone.

``ps`` and ``lsof`` are never run: ``subprocess.run`` is replaced where a test
needs the listings' own code, and their output is what they printed on
macOS (Darwin 27) for the same names. ``omp`` is never run. The builders and
the autouse sandbox come from :mod:`tests.test_omp_adapter`.
"""

from __future__ import annotations

import os
import subprocess
import time

import pytest

from sx.adapters import omp as omp_module
from sx.adapters.omp import LEASE_HELD, LEASE_UNKNOWN, OmpAdapter
from tests import test_omp_adapter as base
from tests.test_omp_processes import OMP_PID, UID, _private

sandbox = base.sandbox
old = base.old
new = base.new

#: The real listings, taken before the autouse sandbox replaces ``ps``'s.
_LIST_PROCESSES = omp_module._list_processes
_LIST_OPEN_FILES = omp_module._list_open_files


# --- the badge's cache ------------------------------------------------------------------


@pytest.mark.parametrize("blind", [False, True], ids=["answered", "blind"])
def test_a_scan_slower_than_the_reuse_window_is_still_reused(sandbox, old, monkeypatch, blind):
    """The window starts when the scan finishes: a slow scan, or a blind one, is not stale at once.

    Scaled down from production, where the window is 5 s and ``ps`` and
    ``lsof`` may each run until their time limit.
    """
    for sid in ("S1", "S2", "S3", "S4"):
        base._session(old, sid)
    adapter = OmpAdapter()
    sessions = list(adapter.discover())
    listed: list[float] = []

    def slow_ps() -> list | None:
        """Answer the first time only after longer than the reuse window."""
        listed.append(time.monotonic())
        if len(listed) == 1:
            time.sleep(1.05)
        return None if blind else []

    monkeypatch.setattr(omp_module, "_SCAN_REUSE_SECONDS", 1.0)
    monkeypatch.setattr(omp_module, "_list_processes", slow_ps)

    states = [adapter.last_activity(session) for session in sessions]

    assert len(listed) == 1
    assert all(state is not None for state in states)


def test_the_badge_resolves_the_store_and_reads_each_dot_env_once_a_window(sandbox, monkeypatch):
    """Not every ``.env`` file again for each session shown: once, while the window lasts."""
    for index in range(12):
        base._session(sandbox / "work" / f"p{index % 4}", f"S{index}")
    (sandbox / ".env").write_text("OTHER=1\n", encoding="utf-8")
    resolved: list[object] = []
    read: list[object] = []
    resolve, read_dotenv = omp_module._resolve, omp_module._read_dotenv
    monkeypatch.setattr(omp_module, "_resolve", lambda home: resolved.append(home) or resolve(home))
    monkeypatch.setattr(
        omp_module, "_read_dotenv", lambda path: read.append(path) or read_dotenv(path)
    )
    adapter = OmpAdapter()
    sessions = list(adapter.discover())
    resolved.clear()
    read.clear()

    for session in sessions * 2:
        adapter.last_activity(session)

    projects = [path for path in read if path.parent.parent == sandbox / "work"]
    assert len(resolved) <= 1  # the store listing them resolved may still be young enough
    assert len(read) == len(set(read)) and len(projects) == 4


def test_a_delete_resolves_the_store_afresh_right_after_the_badge(sandbox, old, tmp_path):
    """A ``.env`` written a moment after the badge looked is what the delete goes by."""
    path = base._session(old)
    adapter = OmpAdapter()
    session = base._only(adapter)
    adapter.last_activity(session)
    owners = _private(tmp_path / "owners")
    (sandbox / ".env").write_text(f"PI_TEST_SESSION_OWNERS_DIR={owners}\n", encoding="utf-8")

    with base._flocked(owners / "S1.lock"):
        result = adapter.delete(session)

    assert path.exists() and "has this session open" in result.refused[path]


@pytest.mark.parametrize("how", ["delete", "move"])
def test_a_delete_or_move_reads_the_projects_dot_env_afresh_right_after_the_badge(
    sandbox, old, new, tmp_path, how
):
    """The badge read the project's ``.env`` a moment before it was written: that is not reused.

    The badge may keep a project's ``.env`` reading for a few seconds, as it
    keeps the store's; a delete or a move reads it again, and finds the lease
    directory it names.
    """
    path = base._session(old)
    before = path.read_bytes()
    adapter = OmpAdapter()
    session = base._only(adapter)
    adapter.last_activity(session)  # reads <project>/.env: there is none yet
    owners = _private(tmp_path / "owners")
    (old / ".env").write_text(f"PI_TEST_SESSION_OWNERS_DIR={owners}\n", encoding="utf-8")

    with base._flocked(owners / "S1.lock"):
        if how == "delete":
            result = adapter.delete(session)
        else:
            _plan, result = base._move(adapter, old, new)

    assert path.read_bytes() == before and "has this session open" in result.refused[path]


# --- the time limits, and how lsof runs ---------------------------------------------------


def _run_ps_and_lsof(monkeypatch: pytest.MonkeyPatch, lsof_output: bytes) -> list[tuple]:
    """Run the real listings over canned output: one omp, with ``lsof_output`` for it.

    Returns:
        Each ``subprocess.run`` call, as ``(args, keyword arguments)``.
    """
    calls: list[tuple] = []

    def run(args, **kwargs):
        """Stand in for ``subprocess.run``, never running anything."""
        calls.append((args, kwargs))
        if args[0] == omp_module._PS:
            return subprocess.CompletedProcess(args, 0, f"{OMP_PID} {UID} omp\n".encode(), b"")
        return subprocess.CompletedProcess(args, 0, lsof_output, b"")

    monkeypatch.setattr(omp_module, "_list_processes", _LIST_PROCESSES)
    monkeypatch.setattr(omp_module, "_list_open_files", _LIST_OPEN_FILES)
    monkeypatch.setattr(omp_module, "_alive", lambda pid: True)
    monkeypatch.setattr(omp_module.subprocess, "run", run)
    return calls


def test_the_badge_gives_ps_and_lsof_a_short_time_limit(sandbox, old, monkeypatch):
    """The TUI's thread waits a second at most for each listing; anything else, longer."""
    base._session(old)
    calls = _run_ps_and_lsof(monkeypatch, b"p%d\x00\n" % OMP_PID)
    adapter = OmpAdapter()
    session = base._only(adapter)

    adapter.last_activity(session)
    badge = [kwargs["timeout"] for _, kwargs in calls]
    calls.clear()
    adapter.lease_state("S1")
    fresh = [kwargs["timeout"] for _, kwargs in calls]

    assert len(badge) == 2 and max(badge) < omp_module._SCAN_TIMEOUT_SECONDS
    assert fresh == [omp_module._SCAN_TIMEOUT_SECONDS] * 2


def test_lsof_never_waits_on_a_file_system_and_prints_utf_8(sandbox, monkeypatch):
    """``-b`` skips calls that block on a hung mount; a UTF-8 locale prints names as they are."""
    calls = _run_ps_and_lsof(monkeypatch, b"p%d\x00\n" % OMP_PID)
    monkeypatch.setenv("LC_ALL", "C")

    OmpAdapter().lease_state("S1")

    args, kwargs = calls[-1]
    assert args[0] == omp_module._LSOF and "-b" in args
    assert kwargs["env"]["LC_ALL"].endswith("UTF-8")


def test_a_badge_listing_that_runs_out_of_time_reads_as_open(sandbox, old, monkeypatch):
    """Out of time is not knowing: the session counts as open, and a delete is refused."""
    path = base._session(old)
    _run_ps_and_lsof(monkeypatch, b"")

    def hung(args, **kwargs):
        """Stand in for a listing that does not answer in time."""
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(omp_module.subprocess, "run", hung)
    adapter = OmpAdapter()
    session = base._only(adapter)

    last = adapter.last_activity(session)

    assert last is not None and abs(time.time() - last.timestamp()) < 60
    assert adapter.lease_state("S1") == LEASE_UNKNOWN
    assert "cannot tell" in adapter.delete(session).refused[path] and path.exists()


# --- the names lsof prints ------------------------------------------------------------------

#: What ``lsof -nP -w -b -F n0 -p <pid>`` printed on Darwin 27 for a process
#: holding ``<dir>/S<case>.lock`` open, for each awkward directory name, in a
#: C locale and in a UTF-8 one (its other open files left out).
_PREFIX = b"/private/tmp/sx-fix3-adapters/lsofcap/"
_PRINTED = {
    "C": (
        b"p4395\x00\nfcwd\x00n/private/tmp/sx-fix3-adapters\x00"
        b"\nf3\x00n" + _PREFIX + b"we\\nird/Snewline.lock\x00"
        b"\nf4\x00n" + _PREFIX + b"tab\\tdir/Stab.lock\x00"
        b"\nf5\x00n" + _PREFIX + b"back\\\\slash/Sbackslash.lock\x00"
        b"\nf6\x00n" + _PREFIX + b"caf\\xc3\\xa9/Saccent.lock\x00"
        b"\nf7\x00n" + _PREFIX + b"car^Bet/Scaret.lock\x00"
        b"\nf8\x00n" + _PREFIX + b"c\\rr/Scr.lock\x00"
        b"\nf9\x00n" + _PREFIX + b"ct^Arl/Sctrl.lock\x00"
        b"\nf10\x00n" + _PREFIX + b"with space/Sspace.lock\x00\n"
    ),
    "UTF-8": (
        b"p4395\x00\nfcwd\x00n/private/tmp/sx-fix3-adapters\x00"
        b"\nf3\x00n" + _PREFIX + b"we\\nird/Snewline.lock\x00"
        b"\nf4\x00n" + _PREFIX + b"tab\\tdir/Stab.lock\x00"
        b"\nf5\x00n" + _PREFIX + b"back\\\\slash/Sbackslash.lock\x00"
        b"\nf6\x00n" + _PREFIX + b"caf\xc3\xa9/Saccent.lock\x00"
        b"\nf7\x00n" + _PREFIX + b"car^Bet/Scaret.lock\x00"
        b"\nf8\x00n" + _PREFIX + b"c\\rr/Scr.lock\x00"
        b"\nf9\x00n" + _PREFIX + b"ct^Arl/Sctrl.lock\x00"
        b"\nf10\x00n" + _PREFIX + b"with space/Sspace.lock\x00\n"
    ),
}

#: The directories those files were in, as created.
_DIRECTORIES = {
    "newline": "we\nird",
    "tab": "tab\tdir",
    "backslash": "back\\slash",
    "accent": "caf\u00e9",
    "caret": "car^Bet",
    "cr": "c\rr",
    "ctrl": "ct\x01rl",
    "space": "with space",
}

#: What ``lsof`` writes in place of a byte, as captured above.
_LSOF_WRITES = {b"\n": b"\\n", b"\t": b"\\t", b"\r": b"\\r", b"\\": b"\\\\"}


def _as_lsof_prints(name: str, *, utf8: bool) -> bytes:
    """Write ``name`` as ``lsof`` prints it (without the ``^X`` it uses for other controls)."""
    printed = bytearray()
    for byte in os.fsencode(name):
        char = bytes([byte])
        if char in _LSOF_WRITES:
            printed += _LSOF_WRITES[char]
        elif byte >= 0x80 and not utf8:
            printed += b"\\x%02x" % byte
        else:
            printed.append(byte)
    return bytes(printed)


def test_the_captured_output_is_what_this_escaping_predicts():
    """The fixture for the next tests and the real output agree."""
    for locale, output in _PRINTED.items():
        for case, directory in _DIRECTORIES.items():
            if case == "ctrl":
                continue  # printed as ^A, which cannot be told from "^A"
            name = os.fsdecode(_PREFIX) + f"{directory}/S{case}.lock"
            assert b"n" + _as_lsof_prints(name, utf8=locale == "UTF-8") + b"\x00" in output


@pytest.mark.parametrize("locale", ["C", "UTF-8"])
def test_lsof_escapes_in_the_names_it_prints_are_undone(locale):
    """``\\n``, ``\\t``, ``\\r``, ``\\\\`` and ``\\xNN`` are what they stand for.

    Another control character is printed as ``^`` and a letter, which a name
    can also hold: it is left as printed, and such a directory is not asked
    about (a lease file open in it is still held).
    """
    names = omp_module._parse_lsof(_PRINTED[locale])[4395]
    prefix = os.fsdecode(_PREFIX)
    expected = [f"{prefix}{directory}/S{case}.lock" for case, directory in _DIRECTORIES.items()]
    expected[6] = f"{prefix}ct^Arl/Sctrl.lock"
    assert names == ["/private/tmp/sx-fix3-adapters", *expected]


@pytest.mark.parametrize("utf8", [False, True], ids=["C", "UTF-8"])
@pytest.mark.parametrize("case", ["newline", "tab", "backslash", "accent", "cr"])
def test_a_lease_directory_whose_name_lsof_escapes_is_asked_about(
    sandbox, tmp_path, monkeypatch, case, utf8
):
    """omp holds S2's lease there; S1's lease in the same directory is held too."""
    leases = _private(tmp_path / _DIRECTORIES[case])
    open_lease = leases / "S2.lock"
    open_lease.touch()
    printed = _as_lsof_prints(str(open_lease), utf8=utf8)
    _run_ps_and_lsof(monkeypatch, b"p%d\x00\nf3\x00n%s\x00\n" % (OMP_PID, printed))
    adapter = OmpAdapter()

    with base._flocked(leases / "S1.lock"):
        assert adapter.lease_state("S1") == LEASE_HELD
    assert adapter._process_scan().dirs == (leases,)
