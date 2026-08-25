"""Tests for the full delete cascade.

Deleting a session must take everything the harness wrote *for that session* —
and nothing that belongs to the project or to the harness itself. All fixtures
are synthetic under ``tmp_path``; each adapter's own ``home`` is monkeypatched,
never the real stores.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sx.adapters import claude as claude_mod
from sx.adapters import codex as codex_mod
from sx.adapters import gemini as gemini_mod
from sx.adapters.claude import ClaudeAdapter, encode_project_dir
from sx.adapters.codex import CodexAdapter
from sx.adapters.gemini import GeminiAdapter
from sx.model import OrphanKind
from sx.util import is_within

PROJECT = "/work/proj"
SID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
OTHER = "99999999-9999-9999-9999-999999999999"


def _claude_home(tmp_path: Path, monkeypatch, *, sessions=(SID,)) -> ClaudeAdapter:
    """Build a Claude store with every session-keyed artifact class present."""
    monkeypatch.setattr(claude_mod, "home", lambda: tmp_path)
    scratch = tmp_path / "scratch"
    monkeypatch.setattr(claude_mod, "scratchpad_root", lambda: scratch)

    base = tmp_path / ".claude"
    folder = base / "projects" / encode_project_dir(PROJECT)
    folder.mkdir(parents=True)
    for sid in sessions:
        (folder / f"{sid}.jsonl").write_text(
            json.dumps({"type": "user", "cwd": PROJECT, "message": {"content": "hi"}}) + "\n"
        )
        (folder / sid / "tool-results").mkdir(parents=True)
        for sub in ("file-history", "session-env", "tasks"):
            (base / sub / sid).mkdir(parents=True)
            (base / sub / sid / "data").write_text("x")
        (base / "security").mkdir(exist_ok=True)
        (base / "security" / f"security_warnings_state_{sid}.json").write_text("{}")
        (base / "security" / f"security_warnings_state_{sid}.lock").write_text("")
        (scratch / encode_project_dir(PROJECT) / sid).mkdir(parents=True)
        (scratch / encode_project_dir(PROJECT) / sid / "note.txt").write_text("scratch")

    # Project-scoped state that must survive a session delete.
    (folder / "memory").mkdir()
    (folder / "memory" / "MEMORY.md").write_text("# Memory Index\n")
    (folder / "memory" / "a-fact.md").write_text("---\nname: a-fact\n---\nbody\n")

    # Harness bulk that must never be reachable.
    (base / "security" / "agent-sdk-venv" / "lib").mkdir(parents=True)
    (base / "plugins").mkdir()
    (base / "settings.json").write_text("{}")
    (base / ".credentials.json").write_text("{}")
    (base / "history.jsonl").write_text(
        "".join(
            json.dumps({"display": f"p{i}", "project": PROJECT, "sessionId": sid}) + "\n"
            for i, sid in enumerate(list(sessions) + [OTHER])
        )
    )
    (tmp_path / ".claude.json").write_text(
        json.dumps({"projects": {PROJECT: {"allowedTools": ["Bash"]}}})
    )
    return ClaudeAdapter()


def _session(adapter: ClaudeAdapter, sid: str = SID):
    """Return the discovered session with this id."""
    return next(s for s in adapter.discover() if s.session_id == sid)


# --- the cascade reaches everything session-keyed --------------------------


def test_delete_takes_every_session_keyed_artifact(tmp_path, monkeypatch):
    """One delete removes the transcript, sidecar, state dirs, security and scratch."""
    adapter = _claude_home(tmp_path, monkeypatch)
    base = tmp_path / ".claude"

    adapter.delete(_session(adapter))

    for sub in ("file-history", "session-env", "tasks"):
        assert not (base / sub / SID).exists(), sub
    assert not (base / "security" / f"security_warnings_state_{SID}.json").exists()
    assert not (base / "security" / f"security_warnings_state_{SID}.lock").exists()
    assert not (tmp_path / "scratch" / encode_project_dir(PROJECT) / SID).exists()
    folder = base / "projects" / encode_project_dir(PROJECT)
    assert not (folder / f"{SID}.jsonl").exists()
    assert not (folder / SID).exists()


def test_delete_removes_only_this_session_s_security_state(tmp_path, monkeypatch):
    """The id is embedded mid-name; a neighbour's state must survive."""
    adapter = _claude_home(tmp_path, monkeypatch)
    keep = tmp_path / ".claude" / "security" / f"security_warnings_state_{OTHER}.json"
    keep.write_text("{}")

    adapter.delete(_session(adapter))

    assert keep.exists()


def test_delete_removes_this_session_s_history_rows_only(tmp_path, monkeypatch):
    """Prompt history is rewritten in place, keeping every other session's rows."""
    adapter = _claude_home(tmp_path, monkeypatch)
    history = tmp_path / ".claude" / "history.jsonl"

    result = adapter.delete(_session(adapter))

    remaining = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
    assert [row["sessionId"] for row in remaining] == [OTHER]
    assert result.note is not None and "prompt-history" in result.note


def test_codex_delete_removes_its_history_rows(tmp_path, monkeypatch):
    """Codex tags prompt history with session_id, in its own file."""
    monkeypatch.setattr(codex_mod, "home", lambda: tmp_path)
    rollout = tmp_path / ".codex" / "sessions" / "2026" / "01" / "01" / "rollout-x.jsonl"
    rollout.parent.mkdir(parents=True)
    rollout.write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "cx-1", "cwd": PROJECT}}) + "\n"
    )
    history = tmp_path / ".codex" / "history.jsonl"
    history.write_text(
        json.dumps({"session_id": "cx-1", "text": "gone"}) + "\n"
        + json.dumps({"session_id": "cx-2", "text": "kept"}) + "\n"
    )

    adapter = CodexAdapter()
    adapter.delete(next(iter(adapter.discover())))

    rows = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
    assert [r["session_id"] for r in rows] == ["cx-2"]


# --- and nothing that is not ------------------------------------------------


def test_every_cascade_path_carries_the_session_id(tmp_path, monkeypatch):
    """The containment assertion: no target may be unrelated to the session."""
    adapter = _claude_home(tmp_path, monkeypatch)
    session = _session(adapter)
    paths = adapter.correlated_paths(session)

    assert paths
    for path in paths:
        assert SID in path.name, path


def test_every_cascade_path_is_inside_a_declared_store_root(tmp_path, monkeypatch):
    """A target the guard would refuse is a target that silently never goes."""
    adapter = _claude_home(tmp_path, monkeypatch)
    roots = adapter.store_roots()

    for path in adapter.correlated_paths(_session(adapter)):
        assert is_within(path, roots), path


def test_a_non_uuid_session_id_triggers_no_cascade(tmp_path, monkeypatch):
    """A hand-placed or malformed transcript must never sweep shared state."""
    adapter = _claude_home(tmp_path, monkeypatch)
    folder = tmp_path / ".claude" / "projects" / encode_project_dir(PROJECT)
    (folder / "a.jsonl").write_text(json.dumps({"type": "user", "cwd": PROJECT}) + "\n")

    stray = next(s for s in adapter.discover() if s.session_id == "a")
    assert adapter.correlated_paths(stray) == []


@pytest.mark.parametrize(
    "relative",
    ["settings.json", ".credentials.json", "plugins", "history.jsonl", "plans"],
)
def test_harness_config_is_outside_every_store_root(tmp_path, monkeypatch, relative):
    """~/.claude must never become a store root: it holds all of this."""
    adapter = _claude_home(tmp_path, monkeypatch)
    assert not is_within(tmp_path / ".claude" / relative, adapter.store_roots())


def test_the_vendored_venv_is_refused_even_inside_its_root(tmp_path, monkeypatch):
    """security/ has to be a root; the 282 MB venv inside it must still be safe."""
    adapter = _claude_home(tmp_path, monkeypatch)
    venv = tmp_path / ".claude" / "security" / "agent-sdk-venv"

    result = adapter.delete_paths([venv], dry_run=False)

    assert venv.exists()
    assert "protected" in next(iter(result.refused.values()))


# --- memory survives a session delete --------------------------------------


def test_deleting_a_session_never_touches_project_memory(tmp_path, monkeypatch):
    """Memory belongs to the directory; a conversation ending must not take it."""
    adapter = _claude_home(tmp_path, monkeypatch, sessions=(SID, OTHER))
    memory = tmp_path / ".claude" / "projects" / encode_project_dir(PROJECT) / "memory"

    adapter.delete(_session(adapter))

    assert (memory / "a-fact.md").read_text() == "---\nname: a-fact\n---\nbody\n"
    assert (memory / "MEMORY.md").exists()


def test_project_leftovers_reports_memory_and_settings(tmp_path, monkeypatch):
    """What the confirmation shows: memory files, paths, and config entries."""
    adapter = _claude_home(tmp_path, monkeypatch)

    leftovers = adapter.project_leftovers(PROJECT)

    assert len(leftovers.memory_files) == 2
    assert any("settings entry" in note for note in leftovers.config_notes)
    assert any("prompt-history" in note for note in leftovers.config_notes)
    assert not leftovers.empty


def test_project_cleanup_removes_memory_folder_and_settings(tmp_path, monkeypatch):
    """The opt-in bundle: only reached through an explicit, unticked checkbox."""
    adapter = _claude_home(tmp_path, monkeypatch)
    adapter.delete(_session(adapter))

    result = adapter.delete_project_leftovers(adapter.project_leftovers(PROJECT))

    folder = tmp_path / ".claude" / "projects" / encode_project_dir(PROJECT)
    assert not folder.exists()
    assert not (tmp_path / "scratch" / encode_project_dir(PROJECT)).exists()
    config = json.loads((tmp_path / ".claude.json").read_text())
    assert config["projects"] == {}
    history = (tmp_path / ".claude" / "history.jsonl").read_text()
    assert PROJECT not in history
    assert result.note is not None


def test_gemini_folder_trust_key_is_matched_case_insensitively(tmp_path, monkeypatch):
    """trustedFolders.json stores paths lower-cased; an exact match finds nothing."""
    monkeypatch.setattr(gemini_mod, "home", lambda: tmp_path)
    base = tmp_path / ".gemini"
    (base / "history" / "proj" / "chats").mkdir(parents=True)
    (base / "history" / "proj" / ".project_root").write_text("/Work/Proj\n")
    (base / "trustedFolders.json").write_text(json.dumps({"/work/proj": "TRUST_FOLDER"}))
    (base / "projects.json").write_text(json.dumps({"projects": {"/Work/Proj": "proj"}}))

    adapter = GeminiAdapter()
    leftovers = adapter.project_leftovers("/Work/Proj")
    assert leftovers is not None
    assert any("trust" in note for note in leftovers.config_notes)

    adapter.delete_project_leftovers(leftovers)
    assert json.loads((base / "trustedFolders.json").read_text()) == {}
    assert not (base / "history" / "proj").exists()


# --- the backlog surfaces in the orphan screen ------------------------------


def test_stale_artifacts_are_grouped_into_one_orphan_per_class(tmp_path, monkeypatch):
    """350 separate rows would bury every other finding, so each class is one row."""
    adapter = _claude_home(tmp_path, monkeypatch)
    base = tmp_path / ".claude"
    for dead in (OTHER, "12345678-1234-1234-1234-123456789abc"):
        (base / "session-env" / dead).mkdir(parents=True)

    stale = [o for o in adapter.find_orphans() if o.kind is OrphanKind.STALE_SESSION]
    env = next(o for o in stale if "session-env" in o.reason)

    assert len(env.paths) == 2
    assert "2 entry(s)" in env.reason
    for path in env.paths:
        assert is_within(path, adapter.store_roots())


def test_a_live_session_s_artifacts_are_never_called_stale(tmp_path, monkeypatch):
    """The session exists, so its state is in use — not backlog."""
    adapter = _claude_home(tmp_path, monkeypatch)

    stale = [o for o in adapter.find_orphans() if o.kind is OrphanKind.STALE_SESSION]
    assert not any(SID in path.name for o in stale for path in o.paths)
