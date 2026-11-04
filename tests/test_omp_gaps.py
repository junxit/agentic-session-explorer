"""Oh My Pi guards no other test held to account.

A session in a format ``sx`` does not know stays where it is when its bucket
moves, and keeps a dead project's bucket even when it appears between Pi's
re-check and omp's own. A ``PI_TEST_RUNTIME=1`` that ``sx``'s own environment
overrides still sends some omp to the test runtime's lease directory. And a
``.env`` file larger than 1 MiB when it is opened, or that grows past it while
it is read, is never taken for what was read of it.

Stores are synthetic, under ``tmp_path``; the builders and the autouse sandbox
come from :mod:`tests.test_omp_adapter`. ``omp`` is never run.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sx.adapters import omp as omp_module
from sx.adapters.omp import LEASE_UNKNOWN, OmpAdapter
from sx.model import OrphanKind
from tests import test_omp_adapter as base

sandbox = base.sandbox
old = base.old
new = base.new

UID = os.getuid()


# --- a format sx does not know --------------------------------------------------------------


@pytest.mark.parametrize("problem", ["version", "slot-size"])
def test_a_session_in_a_format_sx_does_not_know_stays_when_its_bucket_moves(old, new, problem):
    """V4 is refused, and must not ride along when the bucket is renamed: S1 moves on its own."""
    kept = base._bucket(old) / base._name("V4")
    if problem == "version":
        base._write(kept, [base._header("V4", str(old), version=4)], slot=base._slot())
    else:
        base._write(kept, [base._header("V4", str(old))], slot=base._slot(size=300))
    base._age(kept)
    moving = base._session(old, "S1")
    before = kept.read_bytes()

    plan, result = base._move(OmpAdapter(), old, new)

    assert kept in plan.blocked and kept.read_bytes() == before
    assert kept in result.refused
    assert (base._bucket(new) / moving.name).exists() and not moving.exists()


def test_a_dead_projects_bucket_that_gains_an_unknown_format_after_pis_check_stays(
    sandbox, monkeypatch
):
    """V4 is written between Pi's re-check of the bucket and omp's own: omp's still refuses."""
    gone = sandbox / "gone" / "proj"
    base._session(gone, "S1")
    base._age_tree(base._sessions(sandbox), base.DAY)
    adapter = OmpAdapter()
    (orphan,) = [o for o in adapter.find_orphans() if o.kind is OrphanKind.DEAD_PROJECT]
    late = base._bucket(gone) / base._name("V4")
    pis_check = OmpAdapter._dead_folder_refusal

    def then_v4_appears(self, folder: Path) -> str | None:
        """Run Pi's re-check, then let a newer omp write a version 4 session."""
        reason = pis_check(self, folder)
        if not late.exists():
            base._write(late, [base._header("V4", str(gone), version=4)], slot=base._slot())
            base._age(late, seconds=base.DAY)
        return reason

    monkeypatch.setattr(OmpAdapter, "_dead_folder_refusal", then_v4_appears)

    result = adapter.delete_orphan(orphan)

    assert late.exists() and "version 4" in result.refused[orphan.paths[0]]


# --- a test runtime only another omp would see ------------------------------------------------


def test_a_test_runtime_switch_sxs_environment_overrides_still_counts(sandbox, monkeypatch):
    """``PI_TEST_RUNTIME=0`` here; an omp started without it reads ``~/.env``'s ``1``.

    That omp keeps its leases in the test runtime's directory, which is then
    asked about in full: one that is not private reads unknown, as for omp.
    """
    monkeypatch.setenv("PI_TEST_RUNTIME", "0")
    (sandbox / ".env").write_text("PI_TEST_RUNTIME=1\n", encoding="utf-8")
    test_leases = Path(os.environ["TMPDIR"]) / f"omp-test-session-owners-{UID}"
    test_leases.mkdir(mode=0o755)
    test_leases.chmod(0o755)
    adapter = OmpAdapter()

    dirs, unknowable = adapter.lease_dirs()

    assert adapter.lease_dir() == base._lease_dir(sandbox)  # sx's own omp would not use it
    assert test_leases in dirs and not unknowable
    assert adapter.lease_state("S1") == LEASE_UNKNOWN


# --- the size of a .env file ----------------------------------------------------------------------


def _same_file(info: os.stat_result, path: Path) -> bool:
    """Return True if ``info`` describes the file at ``path``."""
    other = os.stat(path)
    return (info.st_dev, info.st_ino) == (other.st_dev, other.st_ino)


def test_a_dot_env_larger_than_the_limit_when_opened_is_unknown_however_little_is_read(
    sandbox, monkeypatch
):
    """Its size when opened decides: a file cut short while ``sx`` reads it was being rewritten."""
    dotenv = sandbox / ".env"
    dotenv.write_text("PI_TEST_SESSION_OWNERS_DIR=/owners\n", encoding="utf-8")
    real_fstat = os.fstat

    def fstat(fd: int) -> os.stat_result:
        """Report ``~/.env`` as over the limit, as it was when it was opened."""
        info = real_fstat(fd)
        if not _same_file(info, dotenv):
            return info
        fields = list(info[:10])
        fields[6] = omp_module._DOTENV_LIMIT + 1  # st_size
        return os.stat_result(fields)

    monkeypatch.setattr(omp_module.os, "fstat", fstat)

    assert OmpAdapter().lease_state("S1") == LEASE_UNKNOWN


def test_a_dot_env_that_grows_past_the_limit_while_it_is_read_is_unknown(sandbox, monkeypatch):
    """Under the limit when opened, over it by the time it is read: what was read is not all."""
    dotenv = sandbox / ".env"
    dotenv.write_text("PI_TEST_SESSION_OWNERS_DIR=/owners\n", encoding="utf-8")
    real_fstat = os.fstat

    def fstat(fd: int) -> os.stat_result:
        """Let ``~/.env`` grow right after ``sx`` looks at its size."""
        info = real_fstat(fd)
        if _same_file(info, dotenv) and info.st_size <= omp_module._DOTENV_LIMIT:
            with dotenv.open("a", encoding="utf-8") as fh:
                fh.write("#" * omp_module._DOTENV_LIMIT)
        return info

    monkeypatch.setattr(omp_module.os, "fstat", fstat)

    assert OmpAdapter().lease_state("S1") == LEASE_UNKNOWN
