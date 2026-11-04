"""Oh My Pi liveness from the lease files running omp processes hold open (D5).

The variables that place omp's leases can live in omp's own environment alone
— a shell's ``XDG_STATE_HOME``, ``PI_TEST_RUNTIME``, a ``PI_CONFIG_DIR`` with a
shared agent directory, a ``.env`` in the directory omp was started from —
where ``sx`` cannot read them. So ``sx`` also looks at what the running omp
processes have open. These tests never run ``omp`` and never list real
processes: they replace the process listing (``ps``) and the open-file listing
(``lsof``) with fakes. The builders and the autouse sandbox (which lists no
processes at all) come from :mod:`tests.test_omp_adapter`.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from sx.adapters import omp as omp_module
from sx.adapters.omp import LEASE_FREE, LEASE_HELD, LEASE_UNKNOWN, OmpAdapter
from sx.service import DeleteService
from tests import test_omp_adapter as base

sandbox = base.sandbox
old = base.old

#: A process id no test process has; the fakes stand behind it.
OMP_PID = 999_001
UID = os.getuid()


class _Processes:
    """Fake process and open-file listings, counting how often they are asked."""

    def __init__(self) -> None:
        """Start with no process running."""
        self.processes: list[tuple[int, int, str]] | None = []
        self.open_files: dict[int, list[str]] | None = {}
        self.alive: set[int] = set()
        self.listed = 0

    def install(self, monkeypatch: pytest.MonkeyPatch) -> _Processes:
        """Put the fakes in place of ``ps``, ``lsof`` and the liveness check."""
        monkeypatch.setattr(omp_module, "_list_processes", self.list_processes, raising=False)
        monkeypatch.setattr(omp_module, "_list_open_files", self.list_open_files, raising=False)
        monkeypatch.setattr(omp_module, "_alive", lambda pid: pid in self.alive, raising=False)
        return self

    def list_processes(self) -> list[tuple[int, int, str]] | None:
        """Stand in for ``ps``."""
        self.listed += 1
        return self.processes

    def list_open_files(self, pids: list[int]) -> dict[int, list[str]] | None:
        """Stand in for ``lsof``: only the processes asked about."""
        if self.open_files is None:
            return None
        return {pid: files for pid, files in self.open_files.items() if pid in pids}

    def omp_holds(self, *leases: Path, command: str = "/Users/me/.local/bin/omp") -> None:
        """Run one omp process holding ``leases`` open."""
        self.processes = [(OMP_PID, UID, command)]
        self.open_files = {OMP_PID: ["/Users/me", *(str(lease) for lease in leases)]}
        self.alive = {OMP_PID}


@pytest.fixture
def processes(monkeypatch: pytest.MonkeyPatch) -> _Processes:
    """Replace the process and open-file listings with fakes."""
    return _Processes().install(monkeypatch)


def _private(directory: Path) -> Path:
    """Create a lease directory, private as omp makes it."""
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    return directory


# --- D5: leases placed by omp's own environment ---------------------------------------


@pytest.mark.parametrize(
    "where",
    [
        "home/custom-state/omp/run/session-owners",  # XDG_STATE_HOME in omp's shell only
        "home/work/.env-owners",  # PI_TEST_SESSION_OWNERS_DIR from omp's launch dir's .env
        "home/.omp-work/run/session-owners",  # PI_CONFIG_DIR with a shared agent directory
        "systmp/omp-test-session-owners-{uid}",  # PI_TEST_RUNTIME=1 in omp's environment
    ],
    ids=["xdg-in-omps-shell", "launch-dir-dotenv", "config-dir", "test-runtime"],
)
def test_a_lease_only_omps_own_environment_places_is_found(
    sandbox, old, tmp_path, processes, where
):
    """The verifier's four cases: sx's own environment names none of these directories.

    Nothing holds a lock on the lease file here, so only the scan can find it.
    """
    path = base._session(old)
    lease = _private(tmp_path / where.format(uid=UID)) / "S1.lock"
    lease.write_text(f"{OMP_PID}:1\n", encoding="utf-8")
    processes.omp_holds(lease)
    adapter = OmpAdapter()
    (session,) = adapter.discover()

    result = DeleteService({"omp": adapter}, log_path=sandbox / "ops.log").delete(session)

    assert path.exists() and "has this session open" in result.refused[path]
    assert adapter.lease_state("S1") == LEASE_HELD


def test_a_lease_file_held_open_is_held_wherever_it_is(sandbox, old, tmp_path, processes):
    """Even in a directory sx would never look in, and with nothing holding a lock on it."""
    lease = _private(tmp_path / "anywhere") / "S1.lock"
    lease.touch()
    processes.omp_holds(lease)
    adapter = OmpAdapter()

    assert adapter.lease_state("S1") == LEASE_HELD
    assert adapter.lease_state("S2") == LEASE_FREE


@pytest.mark.parametrize(
    "command",
    [
        "omp",
        "/opt/homebrew/bin/omp --mode rpc",
        "/Users/John Doe/.local/bin/omp",
        "bun /Users/me/.bun/bin/omp",
        "node /usr/local/lib/node_modules/@oh-my-pi/pi-coding-agent/dist/cli.js",
        "bun run /src/oh-my-pi/packages/coding-agent/src/cli.ts",
    ],
)
def test_omp_is_recognized_however_it_runs(sandbox, old, tmp_path, processes, command):
    """The Bun binary named omp, and Bun or Node running omp's package."""
    lease = _private(tmp_path / "leases") / "S1.lock"
    lease.touch()
    processes.omp_holds(lease, command=command)
    assert OmpAdapter().lease_state("S1") == LEASE_HELD


@pytest.mark.parametrize(
    "command",
    ["zsh", "vim notes.txt", "/usr/bin/ompd", "compiler --opt", "python3 -m sx"],
)
def test_other_processes_are_not_looked_at(sandbox, monkeypatch, processes, command):
    """Only command lines that may be omp's have their open files listed."""
    processes.processes = [(OMP_PID, UID, command)]
    asked: list[list[int]] = []
    monkeypatch.setattr(omp_module, "_list_open_files", lambda pids: asked.append(pids) or {})
    assert OmpAdapter().lease_state("S1") == LEASE_FREE
    assert asked == []


def test_another_users_processes_and_sx_itself_are_not_omp(sandbox, tmp_path, processes):
    """A lease file open in another user's omp is that user's; sx is never omp."""
    lease = _private(tmp_path / "leases") / "S1.lock"
    lease.touch()
    processes.processes = [(OMP_PID, UID + 1, "omp"), (os.getpid(), UID, "omp")]
    processes.open_files = {OMP_PID: [str(lease)], os.getpid(): [str(lease)]}
    processes.alive = {OMP_PID, os.getpid()}
    assert OmpAdapter().lease_state("S1") == LEASE_FREE


# --- not knowing ------------------------------------------------------------------------


def test_an_omp_whose_open_files_cannot_be_listed_makes_free_leases_unknown(
    sandbox, old, processes
):
    """lsof could not list a running omp: any session might be its."""
    path = base._session(old)
    processes.processes = [(OMP_PID, UID, "omp")]
    processes.open_files = {}  # lsof listed nothing for it
    processes.alive = {OMP_PID}
    adapter = OmpAdapter()

    assert adapter.lease_state("S1") == LEASE_UNKNOWN
    assert "cannot tell" in adapter.delete(base._only(adapter)).refused[path]
    assert path.exists()


def test_an_omp_that_exited_before_lsof_looked_is_no_reason_to_doubt(sandbox, processes):
    """A process gone between the two listings holds nothing."""
    processes.processes = [(OMP_PID, UID, "omp")]
    processes.open_files = {}
    processes.alive = set()
    assert OmpAdapter().lease_state("S1") == LEASE_FREE


@pytest.mark.parametrize("broken", ["ps", "lsof"])
def test_a_listing_that_fails_makes_free_leases_unknown(sandbox, processes, broken):
    """No process list, or no open files for a running omp: not knowing counts as open."""
    if broken == "ps":
        processes.processes = None
    else:
        processes.processes = [(OMP_PID, UID, "omp")]
        processes.open_files = None
        processes.alive = {OMP_PID}
    assert OmpAdapter().lease_state("S1") == LEASE_UNKNOWN


def test_a_held_lease_stays_held_even_when_the_scan_is_blind(sandbox, processes):
    """Not knowing never turns a lease the kernel reports held into anything else."""
    processes.processes = None
    with base._flocked(base._lease_dir(sandbox) / "S1.lock"):
        assert OmpAdapter().lease_state("S1") == LEASE_HELD


# --- the lease directories running omp processes use ---------------------------------------


def test_a_lease_directory_in_use_is_asked_about_every_session(sandbox, tmp_path, processes):
    """omp holds S2's lease in a directory sx did not know; S1's lease there is held too.

    Whoever holds S1's lease — an omp the process listing did not recognize —
    is found by asking the kernel about that directory.
    """
    leases = _private(tmp_path / "elsewhere")
    s2 = leases / "S2.lock"
    s2.touch()
    processes.omp_holds(s2)
    with base._flocked(leases / "S1.lock"):
        assert OmpAdapter().lease_state("S1") == LEASE_HELD


def test_only_private_directories_become_lease_directories(sandbox, tmp_path, processes):
    """A ``.lock`` in a shared directory is no lease directory; omp refuses those."""
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)
    lock = shared / "S3.lock"
    lock.touch()
    processes.omp_holds(lock, base._bucket("/tmp/x") / "2026-05-01T10-00-00-000Z_S1.jsonl.lock")
    adapter = OmpAdapter()

    assert adapter.lease_state("S3") == LEASE_HELD  # it is open, so held all the same
    assert adapter.lease_state("S1") == LEASE_FREE  # a transcript's lock file is no lease
    assert shared not in adapter._process_scan().dirs


def test_no_known_lease_directory_but_one_in_use_reads_free(sandbox, old, tmp_path, processes):
    """sx found no lease directory of its own; the one omp is using says S1 is free."""
    base._session(old)
    base._lease_dir(sandbox).rmdir()
    adapter = OmpAdapter()
    assert adapter.lease_state("S1") == LEASE_UNKNOWN

    s2 = _private(tmp_path / "in-use") / "S2.lock"
    s2.touch()
    processes.omp_holds(s2)
    assert adapter.lease_state("S1") == LEASE_FREE


# --- how often the processes are listed ------------------------------------------------------


def test_the_live_badge_reuses_a_recent_scan(sandbox, old, processes):
    """The TUI asks about every session it shows; the processes are listed once in a while."""
    base._session(old, "S1")
    base._session(old, "S2")
    adapter = OmpAdapter()
    sessions = list(adapter.discover())

    for session in sessions * 3:
        adapter.last_activity(session)
    assert processes.listed == 1

    adapter.__dict__["_sx_scan"] = replace(adapter.__dict__["_sx_scan"], taken=-1e9)
    adapter.last_activity(sessions[0])
    assert processes.listed == 2


def test_a_delete_always_lists_the_processes_afresh_and_once(sandbox, old, tmp_path, processes):
    """A scan the live badge took a moment ago is never what a delete decides on."""
    path = base._session(old)
    adapter = OmpAdapter()
    (session,) = adapter.discover()
    adapter.last_activity(session)
    assert processes.listed == 1
    lease = _private(tmp_path / "late") / "S1.lock"
    lease.touch()
    processes.omp_holds(lease)  # omp started writing S1 just now

    assert adapter.last_activity(session) is not None and processes.listed == 1  # reused
    result = adapter.delete(session)

    assert path.exists() and "has this session open" in result.refused[path]
    assert processes.listed == 2  # one new scan for the whole delete


def test_each_unit_of_a_move_lists_the_processes_afresh(sandbox, old, processes):
    """Planning looks once; each unit looks again right before it runs."""
    base._session(old, "S1")
    other = base._session(old / "sub", "S2")
    adapter = OmpAdapter()
    plan = adapter.plan_move(adapter.sessions_for_project(old), old, old.parent / "renamed")
    assert processes.listed == 1 and len(plan.units) == 2

    result = adapter.move(plan)

    assert len(result.moved) == 2 and not other.exists()
    assert processes.listed == 3


def test_an_orphan_delete_lists_the_processes_afresh(sandbox, old, tmp_path, processes):
    """omp writes a header-less file named for S1, holding S1's lease while it does."""
    fragment = base._write(base._bucket(old) / base._name("S1"), [base._user("u1", None, "x")])
    base._age_tree(fragment.parent, base.DAY)
    adapter = OmpAdapter()
    (orphan,) = adapter.find_orphans()
    lease = _private(tmp_path / "late") / "S1.lock"
    lease.touch()
    processes.omp_holds(lease)

    result = adapter.delete_orphan(orphan)

    assert "has this session open" in result.refused[fragment] and fragment.exists()


def test_a_leftover_delete_lists_the_processes_afresh(sandbox, old, tmp_path, processes):
    """The last session is deleted; omp's writer brings it back as debris, holding its lease.

    The bucket was offered before the delete. By the time its removal runs,
    omp holds the lease of the session named by what is in it.
    """
    path = base._session(old)
    adapter = OmpAdapter()
    offer = adapter.project_leftovers(str(old))
    assert offer is not None and offer.paths == [path.parent]
    assert adapter.delete(base._only(adapter)).removed
    fragment = base._write(path, [base._user("u9", "u8", "written after the delete")], slot=None)
    base._age(fragment, seconds=120)
    lease = _private(tmp_path / "late") / "S1.lock"
    lease.touch()
    processes.omp_holds(lease)

    result = adapter.delete_project_leftovers(offer)

    assert "has this session open" in result.refused[path.parent] and fragment.exists()


# --- omp's test runtime (its ys) ------------------------------------------------------------


def test_pi_test_runtime_in_sxs_environment_moves_the_lease_directory(sandbox, monkeypatch):
    """With ``PI_TEST_RUNTIME=1`` and no lease directory set, omp uses the temp directory."""
    expected = Path(os.environ["TMPDIR"]) / f"omp-test-session-owners-{UID}"
    monkeypatch.setenv("PI_TEST_RUNTIME", "1")
    assert OmpAdapter().lease_dir() == expected

    monkeypatch.setenv("PI_TEST_SESSION_OWNERS_DIR", "")  # set but empty: still the temp dir
    adapter = OmpAdapter()
    assert adapter.lease_dir() == expected
    assert not any("set but empty" in warning for warning in adapter.store_warnings())

    monkeypatch.setenv("PI_TEST_SESSION_OWNERS_DIR", str(sandbox / "owners"))
    assert OmpAdapter().lease_dir() == sandbox / "owners"  # the variable wins


def test_pi_test_runtime_from_a_dot_env_moves_the_lease_directory(sandbox):
    """omp reads the switch after it loads its ``.env`` files."""
    (sandbox / ".env").write_text("PI_TEST_RUNTIME=1\n", encoding="utf-8")
    expected = Path(os.environ["TMPDIR"]) / f"omp-test-session-owners-{UID}"
    assert OmpAdapter().lease_dir() == expected


def test_the_test_runtimes_lease_directory_is_always_asked(sandbox, old):
    """An omp whose own environment set ``PI_TEST_RUNTIME=1`` is found without any scan."""
    leases = _private(Path(os.environ["TMPDIR"]) / f"omp-test-session-owners-{UID}")
    with base._flocked(leases / "S1.lock"):
        assert OmpAdapter().lease_state("S1") == LEASE_HELD


def test_a_left_over_test_runtime_directory_never_vouches_that_a_lease_is_free(sandbox, old):
    """It is only a guess at where some omp keeps leases: it may add a held lease, never a free one.

    With no lease directory sx knows of, a stale one from an old test run must
    not turn "cannot tell" into "free".
    """
    base._session(old)
    base._lease_dir(sandbox).rmdir()
    _private(Path(os.environ["TMPDIR"]) / f"omp-test-session-owners-{UID}")
    assert OmpAdapter().lease_state("S1") == LEASE_UNKNOWN


# --- the listings themselves --------------------------------------------------------------


def test_ps_output_is_read_line_by_line():
    """``pid uid command`` per line; a command keeps its spaces; junk is skipped."""
    output = (
        b"    1     0 /sbin/launchd\n"
        b"  412   501 /Users/John Doe/.local/bin/omp --mode rpc\n"
        b"  413   501\n"
        b"garbage line\n"
    )
    assert omp_module._parse_ps(output) == [
        (1, 0, "/sbin/launchd"),
        (412, 501, "/Users/John Doe/.local/bin/omp --mode rpc"),
        (413, 501, ""),
    ]


def test_lsof_output_is_read_field_by_field():
    """``-F n0``: NUL after every field, a newline after every set."""
    output = (
        b"p412\x00\nfcwd\x00n/Users/me/work\x00\nf12\x00n/Users/me/.omp/run/session-owners/S1.lock\x00"
        b"\nf13\x00n/odd\nname.lock\x00\np413\x00\n"
    )
    assert omp_module._parse_lsof(output) == {
        412: ["/Users/me/work", "/Users/me/.omp/run/session-owners/S1.lock", "/odd\nname.lock"],
        413: [],
    }


@pytest.mark.parametrize(
    ("name", "session_id"),
    [
        ("/x/session-owners/S1.lock", "S1"),
        ("/x/omp-test-session-owners-501/abc-DEF_9.lock", "abc-DEF_9"),
        ("/x/h-0123456789abcdef.lock", "h-0123456789abcdef"),
        ("/x/2026-05-01T10-00-00-000Z_S1.jsonl.lock", None),
        ("/x/.S1.jsonl.lock.os", None),
        ("relative/S1.lock", None),
        ("/x/S1.lockfile", None),
    ],
)
def test_only_lease_file_names_count(name, session_id):
    """``<id>.lock``, the id as omp uses it as a file name; a transcript's lock files are not."""
    assert omp_module._lease_name(name) == session_id
