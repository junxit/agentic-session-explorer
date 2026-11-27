"""The shared lock probe and process listings (:mod:`sx.locks`)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from sx import locks
from tests.agy_store import Holder


@pytest.mark.skipif(sys.platform != "darwin", reason="F_GETLK sees flock only on macOS")
def test_a_flock_held_elsewhere_reads_held_and_nothing_is_taken(tmp_path: Path):
    """Another process's flock is seen; once it is gone the lock reads free."""
    lock = tmp_path / "x.lock"
    lock.write_bytes(b"")
    assert locks.flock_state(lock) == locks.FREE
    with Holder(lock):
        assert locks.flock_state(lock) == locks.HELD
    assert locks.flock_state(lock) == locks.FREE


def test_odd_targets_are_absent_or_unknown(tmp_path: Path):
    """Nothing there is absent; a directory, a link or an unknown layout is unknown."""
    assert locks.flock_state(tmp_path / "missing.lock") == locks.ABSENT
    assert locks.flock_state(tmp_path) == locks.UNKNOWN
    target = tmp_path / "real.lock"
    target.write_bytes(b"")
    link = tmp_path / "link.lock"
    link.symlink_to(target)
    assert locks.flock_state(link) == locks.UNKNOWN
    assert locks.flock_state(target, layout=None) == locks.UNKNOWN


def test_ps_and_lsof_output_is_parsed():
    """``ps`` lines and ``lsof -F n0`` fields, escapes included."""
    assert locks.parse_ps(b"  12   501 /bin/agy -c\n  bad line\n   7 0\n") == [
        (12, 501, "/bin/agy -c"),
        (7, 0, ""),
    ]
    output = b"p12\0n/a\\x20b.db\0\np13\0n/tab\\there\0"
    assert locks.parse_lsof(output) == {12: ["/a b.db"], 13: ["/tab\there"]}


def test_this_process_is_listed():
    """The real ``ps`` finds the test run itself."""
    processes = locks.list_processes()
    assert processes is not None
    assert os.getpid() in {pid for pid, _, _ in processes}
