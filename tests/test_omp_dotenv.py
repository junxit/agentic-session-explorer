"""Oh My Pi's ``.env`` files, read as omp reads them — or not at all.

omp fills unset variables from ``.env`` files at startup with ``util.parseEnv``,
and those variables can move its sessions and its leases. ``sx`` parses them
as Bun's ``parseEnv`` does (:mod:`tests.test_omp_bun_dotenv`) and checks that
reading against Node 26's ``parseEnv``, and counts a file it cannot read the
way omp would as unknown: the leases then go unknown, and nothing is deleted
or moved. Every expected value below is what Node 26.13's ``util.parseEnv``
returned for the same text. The builders and the autouse sandbox come from
:mod:`tests.test_omp_adapter`; ``omp`` is never run.
"""

from __future__ import annotations

import os

import pytest

from sx.adapters.omp import (
    LEASE_FREE,
    LEASE_HELD,
    LEASE_UNKNOWN,
    OmpAdapter,
    _DotEnv,
    _node_parse_env,
    _omp_values,
)
from sx.adapters.omp import _read_dotenv as read
from tests import test_omp_adapter as base

sandbox = base.sandbox
old = base.old

MiB = 1024 * 1024


def parse(text: str) -> dict[str, str]:
    """Return what omp keeps of Node's reading of ``text``."""
    return _omp_values(_node_parse_env(text))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # The three cases sx used to read differently.
        (
            "XDG_STATE_HOME=/a/state\rPI_CONFIG_DIR=.cfg\r",
            {"XDG_STATE_HOME": "/a/statePI_CONFIG_DIR=.cfg"},
        ),
        ("XDG_STATE_HOME=/a/state \n", {"XDG_STATE_HOME": "/a/state "}),
        ("export\tXDG_STATE_HOME=/a/state\n", {}),
        # Node's other quirks.
        ("A=   \nB=2", {"A": "B=2"}),
        ("A=\nB=2", {"A": "", "B": "2"}),
        ("export A=\nB=2", {"B": "2"}),
        ("export  A=1\n", {"A": "1"}),
        ("=A=1\n", {"A": "1"}),
        ('A="x\nB=2\n', {"A": '"x', "B": "2"}),
        ("A='a\\nb'\n", {"A": "a\\nb"}),
        ('A="a\\nb"\n', {"A": "a\nb"}),
        ('A="1\nB=2" C=3\nD=4', {"A": "1\nB=2", "D": "4"}),
        ('A="x\\"y"\n', {"A": "x\\"}),
        ("A=x#y #z\n", {"A": "x"}),
        ("A=\fx\f\n", {"A": "\fx\f"}),
        ("﻿A=1\nB=2\n", {"B": "2"}),
        ("A=1 B=2\n", {"A": "1 B=2"}),
        ("A=1\x00x\nB=2\n", {"B": "2"}),
        ("A=1\nA=2\n", {"A": "2"}),
    ],
)
def test_dot_env_text_is_parsed_as_node_parses_it(text, expected):
    """Every quirk of ``node_dotenv.cc`` is kept: the variables omp sees are these."""
    assert parse(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A=x\n" * 150_000, {"A": "x"}),
        ("A='x\n" + "B=y\n" * 150_000, {"A": "'x", "B": "y"}),
        ('A="x\n' * 150_000, {"A": "x\nA="}),
    ],
    ids=["plain", "one-unclosed-quote", "quotes-closing-across-lines"],
)
def test_a_large_file_is_parsed_whole_and_quickly(text, expected):
    """150,000 lines: walked by position, so the cost grows with the file, not its square."""
    assert parse(text) == expected


# --- files sx cannot read the way omp does --------------------------------------------------


def _held_elsewhere(sandbox, tmp_path):
    """Return a private lease directory sx would find only through a ``.env``."""
    owners = tmp_path / "owners"
    owners.mkdir(mode=0o700)
    return owners


@pytest.mark.parametrize("kind", ["oversize", "fifo", "unreadable"])
def test_a_dot_env_sx_cannot_read_makes_every_lease_unknown(sandbox, old, tmp_path, kind):
    """It could name the lease directory omp writes to: nothing is deleted, and sx says why."""
    owners = _held_elsewhere(sandbox, tmp_path)
    dotenv = sandbox / ".env"
    if kind == "oversize":
        dotenv.write_text("#" * MiB + f"\nPI_TEST_SESSION_OWNERS_DIR={owners}\n", "utf-8")
    elif kind == "fifo":
        os.mkfifo(dotenv)
    else:
        dotenv.write_text(f"PI_TEST_SESSION_OWNERS_DIR={owners}\n", "utf-8")
        dotenv.chmod(0)
    path = base._session(old)
    try:
        with base._flocked(owners / "S1.lock"):
            adapter = OmpAdapter()
            assert read(dotenv) is None
            assert adapter.lease_state("S1") == LEASE_UNKNOWN
            result = adapter.delete(base._only(adapter))
        assert path.exists() and "cannot tell" in result.refused[path]
        assert any(
            ".env cannot be read the way Oh My Pi reads it" in w for w in adapter.store_warnings()
        )
    finally:
        if kind == "unreadable":
            dotenv.chmod(0o600)


def test_a_value_cut_off_at_the_limit_is_never_read_as_a_shorter_one(sandbox, tmp_path):
    """The old 1 MiB cut read ``/owners`` as ``/own``; now the file is unknown."""
    head = "#" * (MiB - 1 - len("PI_TEST_SESSION_OWNERS_DIR=/own")) + "\n"
    dotenv = sandbox / ".env"
    dotenv.write_text(head + "PI_TEST_SESSION_OWNERS_DIR=/owners\n", "utf-8")
    assert read(dotenv) is None
    assert OmpAdapter().lease_dir() != tmp_path / "own"


def test_a_directory_or_no_file_sets_nothing(sandbox):
    """omp's read fails on both: it finds nothing, and neither does sx."""
    (sandbox / ".env").mkdir()
    assert read(sandbox / ".env") == _DotEnv({}) == read(sandbox / "missing" / ".env")
    assert OmpAdapter().lease_state("S1") == LEASE_FREE


def test_reading_a_directory_leaves_no_descriptor_open(sandbox):
    """``os.fdopen`` refuses a directory's descriptor and does not close it; sx does."""
    (sandbox / ".env").mkdir()
    before = len(os.listdir("/dev/fd"))
    for _ in range(50):
        read(sandbox / ".env")
    assert len(os.listdir("/dev/fd")) <= before


def test_a_projects_dot_env_sx_cannot_read_makes_only_its_sessions_unknown(sandbox, old, tmp_path):
    """``<cwd>/.env`` is loaded only by an omp started there."""
    (old / ".env").write_text("#" * (MiB + 1), "utf-8")
    other = sandbox / "work" / "other"
    other.mkdir()
    adapter = OmpAdapter()

    assert adapter.lease_state("S1", str(old)) == LEASE_UNKNOWN
    assert adapter.lease_state("S1", str(other)) == LEASE_FREE
    with base._flocked(base._lease_dir(sandbox) / "S1.lock"):
        assert adapter.lease_state("S1", str(old)) == LEASE_HELD
