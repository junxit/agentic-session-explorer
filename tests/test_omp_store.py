"""Where Oh My Pi's store is: a shared agent directory (M10) and ``.env`` files (D5).

Every store is synthetic, under ``tmp_path``; ``omp`` and ``pi`` are never run.
The builders and the autouse sandbox come from :mod:`tests.test_omp_adapter`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest

from sx import cli
from sx.adapters.omp import OmpAdapter
from sx.adapters.pi import PiAdapter, encode_session_folder
from sx.model import Session
from tests import test_omp_adapter as base

sandbox = base.sandbox
old = base.old
new = base.new


def _pi_session(agent: Path, project: Path, sid: str = "P1") -> Path:
    """Write a session exactly as Pi writes one: header first, Pi's folder name."""
    path = agent / "sessions" / encode_session_folder(str(project)) / base._name(sid)
    base._write(path, [base._header(sid, str(project)), base._user("u1", None, "hello pi")], slot=None)
    base._age(path)
    return path


def _harnesses(capsys: pytest.CaptureFixture, adapters: list) -> str:
    """Return what ``sx harnesses`` prints for these adapters."""
    assert cli.cmd_harnesses(argparse.Namespace(registry=(adapters, []))) == 0
    return capsys.readouterr().out


# --- M10: an agent directory both harnesses read -----------------------------------------


def test_a_pi_store_named_by_pi_coding_agent_dir_is_left_to_pi(sandbox, monkeypatch, old, new):
    """Without Oh My Pi's marks it is Pi's: Oh My Pi lists, moves and deletes nothing there."""
    agent = sandbox / "pi-agent"
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent))
    path = _pi_session(agent, old)
    pi, omp = PiAdapter(), OmpAdapter()
    stranger = Session(harness="omp", session_id="P1", project_path=str(old), paths=[path])
    before = path.read_bytes()

    assert omp.sessions_dir() == pi.sessions_dir() == agent / "sessions"
    assert omp.available() is False
    assert list(omp.discover()) == [] and omp.find_orphans() == []
    deleted = omp.delete(stranger)
    plan = omp.plan_move([stranger], old, new)
    moved = omp.move(plan)
    assert "Pi's data" in deleted.refused[path] and path.read_bytes() == before
    assert "Pi's data" in plan.blocked[path] and not plan.relocations and not plan.rewrites
    assert moved.failed or not moved.moved
    assert "Pi's data" in omp.delete_paths([path]).refused[path] and path.exists()
    assert omp.project_leftovers(str(old)) is None
    assert any("takes it for Pi's store" in w for w in omp.store_warnings())

    assert pi.available() is True and [s.session_id for s in pi.discover()] == ["P1"]
    assert not any("Oh My Pi" in w for w in pi.store_warnings())


def test_oh_my_pis_marks_make_the_named_store_its_own(sandbox, monkeypatch, old):
    """An ``agent.db`` (or a transcript or backup that starts with a title slot) says Oh My Pi used it."""
    agent = sandbox / "shared-agent"
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent))
    path = _pi_session(agent, old)
    (agent / "agent.db").write_bytes(b"")
    pi, omp = PiAdapter(), OmpAdapter()

    assert omp.available() is True and [s.paths for s in omp.discover()] == [[path]]
    assert pi.available() is False and list(pi.discover()) == []
    assert not any("Pi's store" in w for w in omp.store_warnings())


def test_the_default_store_is_omps_without_any_marks(sandbox, old):
    """``~/.omp/agent`` (or a profile's) is Oh My Pi's by definition: nothing is sampled."""
    path = base._session(old, slot=False)
    assert not (sandbox / ".omp" / "agent" / "agent.db").exists()
    adapter = OmpAdapter()
    assert adapter.available() is True and [s.paths for s in adapter.discover()] == [[path]]


def test_an_override_equal_to_the_default_is_the_default(sandbox, monkeypatch, old):
    """omp drops ``PI_CODING_AGENT_DIR`` when it names the default agent directory."""
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(sandbox / ".omp" / "agent"))
    path = base._session(old, slot=False)
    adapter = OmpAdapter()
    assert adapter.available() is True and [s.paths for s in adapter.discover()] == [[path]]


def test_sx_harnesses_says_whose_store_it_is(sandbox, monkeypatch, old, capsys):
    """The shared store shows up once, under the harness that owns it, and the other says why."""
    agent = sandbox / "pi-agent"
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent))
    _pi_session(agent, old)

    out = _harnesses(capsys, [PiAdapter(), OmpAdapter()])

    assert "✓ Pi" in out
    assert "  · Oh My Pi       store belongs to Pi" in out.splitlines()
    assert "⚠ Oh My Pi: PI_CODING_AGENT_DIR points Oh My Pi at" in out
    assert "takes it for Pi's store" in out

    (agent / "terminal-sessions").mkdir()
    out = _harnesses(capsys, [PiAdapter(), OmpAdapter()])

    assert "  · Pi             store belongs to Oh My Pi" in out.splitlines()
    assert "✓ Oh My Pi" in out
    assert "⚠ Pi:" in out and "holds Oh My Pi's data (terminal-sessions)" in out
    assert "takes it for Pi's store" not in out


def test_each_adapter_says_whose_store_it_left(sandbox, monkeypatch, old):
    """``store_owner`` names the harness a shared store was handed to, and nothing else."""
    agent = sandbox / "pi-agent"
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent))
    _pi_session(agent, old)
    assert (PiAdapter().store_owner(), OmpAdapter().store_owner()) == (None, "Pi")

    (agent / "agent.db").write_bytes(b"")
    assert (PiAdapter().store_owner(), OmpAdapter().store_owner()) == ("Oh My Pi", None)

    monkeypatch.delenv("PI_CODING_AGENT_DIR")
    assert OmpAdapter().store_owner() is None  # its default store is its own


# --- .env files decide the store as they decide omp's ------------------------------------


def _xdg_data(root: Path) -> Path:
    """Create ``<root>/omp`` so omp would move its sessions under ``root``."""
    (root / "omp").mkdir(parents=True)
    return root


def test_dot_env_files_fill_unset_variables_in_omps_order(sandbox, tmp_path, monkeypatch):
    """``<agent dir>/.env`` beats ``<config root>/.env`` beats ``~/.env``; the environment beats all."""
    home_file, config_file = sandbox / ".env", sandbox / ".omp" / ".env"
    agent_file = sandbox / ".omp" / "agent" / ".env"
    agent_file.parent.mkdir(parents=True)
    for name, path in (("home", home_file), ("config", config_file), ("agent", agent_file)):
        path.write_text(f"XDG_DATA_HOME={_xdg_data(tmp_path / name)}\n", encoding="utf-8")

    assert OmpAdapter().sessions_dir() == tmp_path / "agent" / "omp" / "sessions"
    agent_file.unlink()
    assert OmpAdapter().sessions_dir() == tmp_path / "config" / "omp" / "sessions"
    config_file.unlink()
    assert OmpAdapter().sessions_dir() == tmp_path / "home" / "omp" / "sessions"
    monkeypatch.setenv("XDG_DATA_HOME", "")  # empty counts as unset, as in omp
    assert OmpAdapter().sessions_dir() == tmp_path / "home" / "omp" / "sessions"
    monkeypatch.setenv("XDG_DATA_HOME", str(_xdg_data(tmp_path / "env")))
    assert OmpAdapter().sessions_dir() == tmp_path / "env" / "omp" / "sessions"


def test_a_profile_from_dot_env_selects_its_store(sandbox):
    """``OMP_PROFILE`` in ``~/.env`` is applied when omp starts, before it opens a store."""
    (sandbox / ".env").write_text("OMP_PROFILE=work\n", encoding="utf-8")
    adapter = OmpAdapter()
    assert adapter.sessions_dir() == sandbox / ".omp" / "profiles" / "work" / "agent" / "sessions"
    notes = adapter.store_notes()
    assert any("OMP_PROFILE selects Oh My Pi profile 'work'" in note for note in notes)
    assert any(note.startswith("OMP_PROFILE is set in") for note in notes)


def test_omp_prefixed_keys_in_dot_env_set_the_pi_keys(sandbox):
    """omp copies ``OMP_X`` to ``PI_X`` in its ``.env`` files: ``OMP_CONFIG_DIR`` renames ``.omp``."""
    (sandbox / ".env").write_text("OMP_CONFIG_DIR=alt-omp\n", encoding="utf-8")
    adapter = OmpAdapter()
    assert adapter.sessions_dir() == sandbox / "alt-omp" / "agent" / "sessions"
    assert adapter.lease_dir() == sandbox / "alt-omp" / "run" / "session-owners"


def test_the_config_roots_dot_env_follows_the_profile_omp_started_with(sandbox, tmp_path, monkeypatch):
    """With a profile set, the config root's ``.env`` is ``~/.omp/profiles/<p>/.env``."""
    monkeypatch.setenv("OMP_PROFILE", "work")
    profile_root = sandbox / ".omp" / "profiles" / "work"
    profile_root.mkdir(parents=True)
    (profile_root / ".env").write_text(
        f"XDG_DATA_HOME={_xdg_data(tmp_path / 'data')}\n", encoding="utf-8"
    )
    (tmp_path / "data" / "omp" / "profiles" / "work").mkdir(parents=True)
    assert OmpAdapter().sessions_dir() == tmp_path / "data" / "omp" / "profiles" / "work" / "sessions"


def test_pi_coding_agent_dir_in_dot_env_is_not_followed(sandbox, tmp_path):
    """omp reads ``PI_CODING_AGENT_DIR`` before it loads any ``.env`` file."""
    (sandbox / ".env").write_text(f"PI_CODING_AGENT_DIR={tmp_path / 'agent'}\n", encoding="utf-8")
    assert OmpAdapter().sessions_dir() == base._sessions(sandbox)


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('XDG_DATA_HOME="{root}"\n', "quoted"),
        ("XDG_DATA_HOME='{root}'  # single quotes\n", "quoted"),
        ("  export XDG_DATA_HOME={root} # trailing comment\n", "exported"),
        ("# XDG_DATA_HOME={root}\n", None),
        ("XDG_DATA_HOME\n", None),
    ],
)
def test_dot_env_lines_are_read_as_omp_reads_them(sandbox, tmp_path, content, expected):
    """Quotes, ``export`` and comments, as Bun's ``parseEnv`` takes them (and Node's)."""
    root = _xdg_data(tmp_path / "data")
    (sandbox / ".env").write_text(content.format(root=root), encoding="utf-8")
    sessions = OmpAdapter().sessions_dir()
    assert sessions == (root / "omp" / "sessions" if expected else base._sessions(sandbox))


def test_a_dot_env_that_is_not_a_regular_file_is_ignored(sandbox, tmp_path):
    """A pipe named ``.env`` would block the read: it is never opened."""
    os.mkfifo(sandbox / ".env")
    assert OmpAdapter().sessions_dir() == base._sessions(sandbox)
