"""Where a ``.env`` line only Bun's parser reads sends Oh My Pi's leases and store.

omp runs on Bun, and Bun's ``util.parseEnv`` reads some lines Node's does not —
``KEY: value``, ``KEY`` then ``=value`` on the next line, a byte-order mark, a
tab after ``export``, a carriage return as a line break — and one Node reads
that it skips (``=KEY=value``). ``sx`` follows Bun's reading, and a variable it
follows that the two read differently makes every lease unknown, with a
warning (:mod:`tests.test_omp_bun_dotenv` tests the parser itself). The
builders and the autouse sandbox come from :mod:`tests.test_omp_adapter`;
``omp`` is never run.
"""

from __future__ import annotations

import pytest

from sx.adapters.omp import LEASE_FREE, LEASE_HELD, LEASE_UNKNOWN, OmpAdapter
from sx.util import short_path
from tests import test_omp_adapter as base

sandbox = base.sandbox
old = base.old


def _lease_dir(root):
    """Create the lease directory omp uses under an ``XDG_STATE_HOME``, private as omp makes it."""
    leases = root / "omp" / "run" / "session-owners"
    leases.mkdir(parents=True)
    leases.chmod(0o700)
    return leases


@pytest.mark.parametrize(
    "text",
    [
        "XDG_STATE_HOME: {state}\n",
        "XDG_STATE_HOME\n={state}\n",
        "\ufeffXDG_STATE_HOME={state}\n",
        "export\tXDG_STATE_HOME={state}\n",
        "OTHER=1\rXDG_STATE_HOME={state}\r",
        "XDG_STATE_HOME={state}\f\n",
        'XDG_STATE_HOME=\n"{state}"\n',
    ],
    ids=[
        "colon",
        "key-then-newline",
        "byte-order-mark",
        "export-tab",
        "carriage-return",
        "form-feed",
        "quote-on-next-line",
    ],
)
def test_a_lease_directory_only_buns_reading_names_is_asked(sandbox, old, tmp_path, text):
    """omp, on Bun, keeps its leases where this line says; Node's reading would miss them.

    Asked there, a lease omp holds is held, and nothing is deleted. Unheld it
    is still not free: which reading omp's own Bun makes is not certain.
    """
    leases = _lease_dir(tmp_path / "state")
    (sandbox / ".env").write_text(text.format(state=tmp_path / "state"), encoding="utf-8")
    path = base._session(old)
    adapter = OmpAdapter()

    with base._flocked(leases / "S1.lock"):
        assert adapter.lease_state("S1") == LEASE_HELD
        result = adapter.delete(base._only(adapter))

    assert path.exists() and "has this session open" in result.refused[path]
    assert adapter.lease_state("S1") == LEASE_UNKNOWN
    assert any("XDG_STATE_HOME" in w and "Bun" in w for w in adapter.store_warnings())


def test_a_variable_only_nodes_reading_sets_makes_the_leases_unknown(sandbox, old, tmp_path):
    """``=KEY=v`` sets KEY for Node and nothing for Bun: omp keeps its leases at home."""
    _lease_dir(tmp_path / "state")
    (sandbox / ".env").write_text(f"=XDG_STATE_HOME={tmp_path / 'state'}\n", encoding="utf-8")
    path = base._session(old)
    adapter = OmpAdapter()

    assert adapter.lease_dir() == base._lease_dir(sandbox)
    assert adapter.lease_state("S1") == LEASE_UNKNOWN
    assert "cannot tell" in adapter.delete(base._only(adapter)).refused[path]
    assert path.exists()


@pytest.mark.parametrize("name", ["PI_CONFIG_DIR", "OMP_CONFIG_DIR"])
def test_the_store_is_where_buns_reading_puts_it(sandbox, name):
    """omp reads ``PI_CONFIG_DIR: alt-omp`` as set, so its sessions are under ``~/alt-omp``."""
    (sandbox / ".env").write_text(f"{name}: alt-omp\n", encoding="utf-8")
    adapter = OmpAdapter()

    assert adapter.sessions_dir() == sandbox / "alt-omp" / "agent" / "sessions"
    assert adapter.lease_state("S1") == LEASE_UNKNOWN
    warning = next(w for w in adapter.store_warnings() if "Bun" in w)
    assert f".env sets {name} in a form" in warning


def test_a_projects_dot_env_only_bun_reads_moves_its_sessions_leases(sandbox, old, tmp_path):
    """``<cwd>/.env`` is loaded by an omp started there: its sessions only are affected."""
    leases = _lease_dir(tmp_path / "state")
    (old / ".env").write_text(f"XDG_STATE_HOME: {tmp_path / 'state'}\n", encoding="utf-8")
    path = base._session(old)
    adapter = OmpAdapter()
    session = base._only(adapter)

    with base._flocked(leases / "S1.lock"):
        assert adapter.lease_state("S1", str(old)) == LEASE_HELD
        result = adapter.delete(session)

    assert path.exists() and "has this session open" in result.refused[path]
    assert adapter.lease_state("S1", str(old)) == LEASE_UNKNOWN
    assert adapter.lease_state("S1") == LEASE_FREE


def test_lines_both_parsers_read_alike_change_nothing(sandbox, old):
    """A file of ordinary lines, and forms sx does not follow, leave every lease as it was."""
    (sandbox / ".env").write_text(
        "# tools\nexport EDITOR=vim\nOTHER: x\nPATH_EXTRA='a b' # c\n", encoding="utf-8"
    )
    adapter = OmpAdapter()
    assert adapter.lease_state("S1") == LEASE_FREE
    assert adapter.store_warnings() == []


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("XDG_STATE_HOME: {state}\n", "sets XDG_STATE_HOME in a form that Bun's .env parser"),
        ("#" * (1024 * 1024 + 1), "cannot be read the way Oh My Pi reads it at startup"),
        ("XDG_STATE_HOME=relative/state\n", "sets XDG_STATE_HOME to a relative path"),
    ],
    ids=["read-differently", "unreadable", "relative"],
)
def test_a_session_its_projects_dot_env_keeps_unknown_names_the_file(
    sandbox, old, tmp_path, text, problem
):
    """The refusal names ``<cwd>/.env`` and what is wrong with it, not only "cannot tell".

    Store-level ``.env`` files are named in ``sx harnesses``; a project's own
    file affects only that project's sessions, so their refusal is where it is
    named.
    """
    (old / ".env").write_text(text.format(state=tmp_path / "state"), encoding="utf-8")
    path = base._session(old)
    adapter = OmpAdapter()
    session = base._only(adapter)

    result = adapter.delete(session)

    assert path.exists()
    reason = result.refused[path]
    assert reason.startswith(f"{short_path(old / '.env')} {problem}")
    assert reason.endswith(", so sx cannot tell whether Oh My Pi has this session open (refused)")
    assert adapter.lease_state("S1", str(old)) == LEASE_UNKNOWN


def test_a_projects_dot_env_that_reads_alike_is_not_blamed(sandbox, old):
    """An ordinary project ``.env`` neither keeps the session nor appears in a reason."""
    (old / ".env").write_text("EDITOR=vim\n", encoding="utf-8")
    path = base._session(old)
    adapter = OmpAdapter()

    result = adapter.delete(base._only(adapter))

    assert not path.exists() and path not in result.refused
