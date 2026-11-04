"""Shared pytest helpers for the sx test suite.

Every test operates on synthetic data under pytest's ``tmp_path`` fixture. No
test reads or writes the user's real stores — ``~/.claude`` (and Claude's
scratchpads under ``/private/tmp/claude-<uid>``), ``~/.codex``, ``~/.gemini``,
``~/.local/share/opencode``, ``~/.pi`` or ``~/.omp``. Adapters import ``home``
into their own module, so a test redirects that name in each adapter module it
uses; a test that builds the whole registry redirects every one of them, and
Claude's scratchpad root, first (``test_safety_regressions`` shows how), and
the TUI and CLI tests replace the registry with fakes.

The autouse fixtures below scrub ``CLAUDE_CONFIG_DIR`` and every Pi / Oh My Pi
store override for every test, to keep that true on a machine where the
developer has relocated their own stores, and point Claude's scratchpad root
into a sandbox for every test, because that root follows the user id rather
than ``HOME``: a fake home directory does not hide it.

One more turns a test that hangs into a failed run (:func:`_no_hangs`).
"""

from __future__ import annotations

import faulthandler
import json
import os
import threading
from pathlib import Path

import pytest

#: Seconds one test may take before it is treated as hung: the stack of every
#: thread is dumped to stderr and the run exits with status 1, where it used to
#: wait forever — a TUI test whose screen never settles awaits it endlessly.
#: Set ``SX_TEST_TIMEOUT`` to change it, or to ``0`` to turn the guard off
#: (for a debugger session, say).
TEST_TIMEOUT_SECONDS = float(os.environ.get("SX_TEST_TIMEOUT", "120"))

#: A copy of the real stderr, for the hang report (see :func:`pytest_configure`).
_report_fd: int | None = None


def pytest_configure(config: pytest.Config) -> None:
    """Keep a descriptor on the real stderr for the hang guard's report.

    While a test runs, output capture points file descriptor 2 at a file that
    pytest prints only once the test ends, which a run that exits on a hang
    never does. Configuration runs while nothing is being captured, so the
    copy taken here reaches the terminal.

    Args:
        config: The pytest configuration.
    """
    global _report_fd
    _report_fd = os.dup(2)


def pytest_unconfigure(config: pytest.Config) -> None:
    """Close the guard's copy of stderr.

    Args:
        config: The pytest configuration.
    """
    global _report_fd
    if _report_fd is not None:
        os.close(_report_fd)
        _report_fd = None


def _report_hang(test: str) -> None:
    """Name the test that hung, dump every thread's stack, and end the run.

    Args:
        test: The test's node id.
    """
    if _report_fd is None:
        return
    os.write(
        _report_fd,
        f"\n{test} has run for {TEST_TIMEOUT_SECONDS:g} s: taken for a hang\n".encode(),
    )
    faulthandler.dump_traceback(file=_report_fd, all_threads=True)
    os._exit(1)


@pytest.fixture(autouse=True)
def _no_hangs(request: pytest.FixtureRequest):
    """Fail the run, rather than hang, when one test takes far longer than any should.

    Armed before each test's other fixtures and cancelled after their
    teardown, so all of it is covered. A timer thread names the test and
    dumps every thread's stack — the stack alone would not name an async
    test, which hangs awaiting in the event loop. In case the hung code never
    lets that thread run, :mod:`faulthandler`'s own watchdog, which needs no
    interpreter lock, ends the run shortly after.

    Args:
        request: The test's fixture request.

    Yields:
        Nothing.
    """
    if TEST_TIMEOUT_SECONDS <= 0 or _report_fd is None:
        yield
        return
    timer = threading.Timer(TEST_TIMEOUT_SECONDS, _report_hang, args=(request.node.nodeid,))
    timer.daemon = True
    timer.start()
    faulthandler.dump_traceback_later(TEST_TIMEOUT_SECONDS + 30, exit=True, file=_report_fd)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
        timer.cancel()


@pytest.fixture(autouse=True)
def _no_claude_config_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scrub ``CLAUDE_CONFIG_DIR`` so the suite never follows a real one.

    ``sx`` reads this variable the way Claude Code does. Without the scrub, a
    developer who exports it gets the Claude adapter building its store roots —
    the delete allowlist — and its ``.claude.json`` path from their live
    configuration directory while the test believes it is under ``tmp_path``,
    at which point a move test rewrites their real config. Tests that exercise
    the override set the variable themselves; ``monkeypatch`` is function
    scoped, so their ``setenv`` simply wins.
    """
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)


@pytest.fixture(autouse=True)
def _no_real_scratchpads(monkeypatch: pytest.MonkeyPatch, tmp_path_factory) -> None:
    """Point Claude's scratchpad root into a sandbox for every test.

    Every other store follows ``HOME``, which tests redirect. This one is
    ``/private/tmp/claude-<uid>``, derived from the user id and shared by every
    Claude config on the account, so a test that forgot to redirect it reached
    the user's real scratchpads. A test that sets its own root simply
    overrides this one.
    """
    root = tmp_path_factory.mktemp("claude-scratch")
    monkeypatch.setattr("sx.util.scratchpad_root", lambda: root)
    monkeypatch.setattr("sx.adapters.claude.scratchpad_root", lambda: root)


#: Variables Pi and Oh My Pi read to relocate their stores. ``sx`` follows them
#: the way each harness does, so a developer who exports any of them would
#: otherwise point a test's store roots — the delete allowlist — at their real
#: sessions. ``XDG_DATA_HOME``/``XDG_STATE_HOME`` are included because Oh My Pi
#: moves its sessions and leases under them when ``<xdg>/omp`` exists.
_PI_STORE_ENV = (
    "PI_CODING_AGENT_DIR",
    "PI_CODING_AGENT_SESSION_DIR",
    "PI_CONFIG_DIR",
    "PI_PROFILE",
    "OMP_PROFILE",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
)


@pytest.fixture(autouse=True)
def _no_pi_store_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scrub every Pi / Oh My Pi store override, as for ``CLAUDE_CONFIG_DIR``."""
    for name in _PI_STORE_ENV:
        monkeypatch.delenv(name, raising=False)


def write_compact_jsonl(path: Path, records: list[dict]) -> Path:
    """Write records as compact JSON lines, the way every harness writes them.

    Tests that assert byte-for-byte behavior must start from the bytes a real
    harness produces; Python's default separators (``", "``, ``": "``) are not
    those bytes.

    Args:
        path: Destination file (parent dirs are created if needed).
        records: Objects to serialize, one JSON object per line.

    Returns:
        The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in records),
        encoding="utf-8",
    )
    return path


def write_jsonl(path: Path, records: list[dict]) -> Path:
    """Write a list of dicts as a JSONL file and return the path.

    Args:
        path: Destination file (parent dirs are created if needed).
        records: Objects to serialize, one JSON object per line.

    Returns:
        The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in records),
        encoding="utf-8",
    )
    return path
