"""Delete-time safety for orphans, and dead-project detection, in the shared core.

An orphan is listed at one moment and deleted at another, possibly minutes
later, and deletion is permanent. These tests pin that the delete checks again:
a folder listed as empty that has since gained a session, a dead project's
folder whose project came back, anything written in the last 90 seconds — all
refused. And a project ``sx`` cannot see (an untraversable parent) is never
mistaken for one that is gone.

Only behavior is tested, through adapters and services that predate these
fixes, so each test can also be run against older code. Everything is synthetic
under ``tmp_path``; each adapter module's own ``home`` is redirected.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import pytest

from sx.adapters import claude as claude_mod
from sx.adapters import codex as codex_mod
from sx.adapters import gemini as gemini_mod
from sx.adapters import opencode as opencode_mod
from sx.adapters.base import HarnessAdapter
from sx.adapters.claude import ClaudeAdapter, encode_project_dir
from sx.adapters.codex import CodexAdapter
from sx.adapters.gemini import GeminiAdapter
from sx.adapters.opencode import OpencodeAdapter
from sx.model import Capability, Message, Orphan, OrphanKind, Role, Session
from sx.service import DeleteService
from tests.conftest import write_compact_jsonl

SID = "aaaaaaaa-bbbb-cccc-dddd-000000000001"
OTHER = "aaaaaaaa-bbbb-cccc-dddd-000000000002"
HOUR = 3600

needs_permissions = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root traverses directories whatever their mode",
)


def _age(*paths: Path, seconds: float = HOUR) -> None:
    """Backdate files and directories so nothing counts as being written."""
    past = time.time() - seconds
    for path in paths:
        os.utime(path, (past, past), follow_symlinks=False)


def _age_tree(root: Path, seconds: float = HOUR) -> None:
    """Backdate a whole tree, deepest entries first."""
    for top, dirs, files in os.walk(root, topdown=False):
        _age(*(Path(top, name) for name in (*files, *dirs)), seconds=seconds)
    _age(root, seconds=seconds)


def _claude(tmp_path: Path, monkeypatch) -> tuple[ClaudeAdapter, Path]:
    """Point Claude at a sandbox and return the adapter and its projects folder."""
    monkeypatch.setattr(claude_mod, "home", lambda: tmp_path)
    monkeypatch.setattr(claude_mod, "scratchpad_root", lambda: tmp_path / "scratch")
    projects = tmp_path / ".claude" / "projects"
    projects.mkdir(parents=True)
    return ClaudeAdapter(), projects


def _transcript(folder: Path, sid: str, cwd: str) -> Path:
    """Write an aged one-turn Claude transcript recording ``cwd``."""
    path = write_compact_jsonl(
        folder / f"{sid}.jsonl",
        [{"type": "user", "cwd": cwd, "sessionId": sid, "message": {"content": "hi"}}],
    )
    _age(path)
    return path


def _only(orphans: list[Orphan], path: Path) -> Orphan:
    """Return the one orphan whose paths are exactly ``[path]``."""
    found = [orphan for orphan in orphans if orphan.paths == [path]]
    assert len(found) == 1, orphans
    return found[0]


# --- D1: an orphan is checked again when it is deleted -----------------------------


def test_an_empty_folder_that_gains_a_session_is_not_deleted(tmp_path, monkeypatch):
    """The scan saw an empty folder; the harness then wrote a session into it."""
    adapter, projects = _claude(tmp_path, monkeypatch)
    project = tmp_path / "work" / "app"
    project.mkdir(parents=True)
    folder = projects / encode_project_dir(str(project))
    folder.mkdir()
    _age(folder)
    orphan = _only(adapter.find_orphans(), folder)
    assert orphan.kind is OrphanKind.EMPTY

    transcript = _transcript(folder, SID, str(project))
    _age(folder)  # even with the folder's own time put back
    service = DeleteService({"claude": adapter}, log_path=tmp_path / "ops.log")

    preview = service.preview_orphan(orphan)
    result = service.delete_orphan(orphan)

    assert transcript.exists()
    assert folder not in result.removed and folder in result.refused
    assert result.failed
    assert preview.removed == [] and folder in preview.refused


def test_an_orphan_written_in_the_last_90_seconds_waits(tmp_path, monkeypatch):
    """Unchanged since the scan, but just made — a harness may be creating it."""
    adapter, projects = _claude(tmp_path, monkeypatch)
    folder = projects / "-fresh-folder"
    folder.mkdir()
    orphan = _only(adapter.find_orphans(), folder)

    result = adapter.delete_orphan(orphan)

    assert folder.is_dir() and "90 seconds" in result.refused[folder]

    _age(folder)
    aged = _only(adapter.find_orphans(), folder)
    assert adapter.delete_orphan(aged).removed == [folder]
    assert not folder.exists()


def test_a_dead_projects_folder_whose_project_returns_is_not_deleted(tmp_path, monkeypatch):
    """Nothing in the folder changed; its project directory was re-cloned."""
    adapter, projects = _claude(tmp_path, monkeypatch)
    project = tmp_path / "work" / "gone"
    folder = projects / encode_project_dir(str(project))
    transcript = _transcript(folder, SID, str(project))
    _age_tree(folder)
    orphan = _only(adapter.find_orphans(), folder)
    assert orphan.kind is OrphanKind.DEAD_PROJECT

    project.mkdir(parents=True)
    result = adapter.delete_orphan(orphan)

    assert transcript.exists() and folder in result.refused


def test_a_stale_artifact_whose_session_returns_is_not_deleted(tmp_path, monkeypatch):
    """``session-env/<id>`` looked stale; then the session's transcript appeared."""
    adapter, projects = _claude(tmp_path, monkeypatch)
    env = tmp_path / ".claude" / "session-env" / SID
    env.mkdir(parents=True)
    (env / "data").write_text("x", encoding="utf-8")
    _age_tree(env)
    (stale,) = [o for o in adapter.find_orphans() if o.kind is OrphanKind.STALE_SESSION]
    assert stale.paths == [env]

    _transcript(projects / "-work-app", SID, "/work/app")
    result = adapter.delete_orphan(stale)

    assert env.is_dir() and env in result.refused


def test_an_orphan_swapped_for_a_link_out_of_the_store_is_not_deleted(tmp_path, monkeypatch):
    """The listed folder is now a symbolic link to something outside the store."""
    adapter, projects = _claude(tmp_path, monkeypatch)
    folder = projects / "-old-folder"
    folder.mkdir()
    _age(folder)
    orphan = _only(adapter.find_orphans(), folder)
    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    folder.rmdir()
    folder.symlink_to(outside)

    result = adapter.delete_orphan(orphan)

    assert (outside / "keep.txt").exists() and folder.is_symlink()
    assert folder in result.refused


def test_every_scan_fingerprints_what_it_lists(tmp_path, monkeypatch):
    """Claude, Gemini and opencode orphans all carry a fingerprint per path."""
    adapter, projects = _claude(tmp_path, monkeypatch)
    (projects / "-empty").mkdir()
    monkeypatch.setattr(gemini_mod, "home", lambda: tmp_path)
    (tmp_path / ".gemini").mkdir()
    (tmp_path / ".gemini" / "projects.json.1.tmp").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(opencode_mod, "home", lambda: tmp_path)
    store = tmp_path / ".local" / "share" / "opencode"
    (store / "storage" / "session_diff").mkdir(parents=True)
    con = sqlite3.connect(store / "opencode.db")
    con.execute("CREATE TABLE session (id text PRIMARY KEY)")
    con.commit()
    con.close()
    (store / "storage" / "session_diff" / "ghost.json").write_text("[]", encoding="utf-8")

    orphans = [
        *adapter.find_orphans(),
        *GeminiAdapter().find_orphans(),
        *OpencodeAdapter().find_orphans(),
    ]

    assert {o.harness for o in orphans} == {"claude", "gemini", "opencode"}
    for orphan in orphans:
        assert orphan.fingerprints is not None
        assert set(orphan.fingerprints) == set(orphan.paths)


class _Synthetic(HarnessAdapter):
    """A bare adapter whose orphans are built outside any scan of a real store."""

    name = "synthetic"
    display = "Synthetic"
    capabilities = Capability.BROWSE | Capability.ORPHANS | Capability.DELETE

    def __init__(self, root: Path) -> None:
        """Use ``root`` as the only store root."""
        self._root = root

    def store_roots(self) -> list[Path]:
        """Return the one root."""
        return [self._root]

    def discover(self):
        """No sessions."""
        return iter(())

    def load(self, session: Session) -> list[Message]:
        """No transcript."""
        return [Message(Role.USER, text="hi", timestamp=datetime(2026, 1, 1))]

    def find_orphans(self) -> list[Orphan]:
        """List every file in the root."""
        return [
            Orphan(self.name, OrphanKind.STRAY_TEMP, [path], "stray", 1)
            for path in sorted(self._root.iterdir())
        ]


def test_an_adapter_that_does_not_fingerprint_keeps_the_store_root_guards(tmp_path):
    """Only adapters whose orphans are real store data opt into the timing checks."""
    root = tmp_path / "store"
    root.mkdir()
    fresh = root / "fresh.tmp"
    fresh.write_text("x", encoding="utf-8")
    adapter = _Synthetic(root)
    (orphan,) = adapter.find_orphans()
    assert orphan.fingerprints is None

    assert adapter.delete_orphan(orphan).removed == [fresh]


# --- D4: a project sx cannot see is never taken for one that is gone ----------------


@pytest.fixture()
def locked(tmp_path: Path):
    """Yield a live project inside a parent ``sx`` may not traverse."""
    guarded = tmp_path / "guarded"
    project = guarded / "app"
    project.mkdir(parents=True)
    (project / "main.py").write_text("print('live')\n", encoding="utf-8")
    os.chmod(guarded, 0o000)
    try:
        yield project
    finally:
        os.chmod(guarded, 0o755)


@needs_permissions
def test_claude_never_calls_an_unreadable_project_dead(tmp_path, monkeypatch, locked):
    """``Path.exists()`` says False behind an untraversable parent; it is not gone."""
    adapter, projects = _claude(tmp_path, monkeypatch)
    folder = projects / encode_project_dir(str(locked))
    transcript = _transcript(folder, SID, str(locked))
    _age_tree(folder)

    orphans = adapter.find_orphans()
    sessions = list(adapter.discover())
    for orphan in orphans:
        DeleteService({"claude": adapter}, log_path=tmp_path / "ops.log").delete_orphan(orphan)

    assert transcript.exists()
    assert not any(folder in orphan.paths for orphan in orphans)
    assert [s.is_orphan for s in sessions] == [False]


def test_claude_never_calls_a_folder_dead_while_one_project_in_it_lives(tmp_path, monkeypatch):
    """``/w/a/b`` and ``/w/a-b`` share a folder; one is gone, the other is not."""
    adapter, projects = _claude(tmp_path, monkeypatch)
    gone = tmp_path / "w" / "a" / "b"
    alive = tmp_path / "w" / "a-b"
    alive.mkdir(parents=True)
    folder = projects / encode_project_dir(str(gone))
    assert folder.name == encode_project_dir(str(alive))
    _transcript(folder, SID, str(gone))
    keep = _transcript(folder, OTHER, str(alive))
    _age_tree(folder)

    orphans = adapter.find_orphans()

    assert not any(folder in orphan.paths for orphan in orphans)
    assert keep.exists()


@needs_permissions
def test_gemini_never_calls_an_unreadable_project_dead(tmp_path, monkeypatch, locked):
    """The marker names a live project behind an untraversable parent."""
    monkeypatch.setattr(gemini_mod, "home", lambda: tmp_path)
    hash_dir = tmp_path / ".gemini" / "tmp" / "abc123"
    (hash_dir / "chats").mkdir(parents=True)
    (hash_dir / ".project_root").write_text(str(locked) + "\n", encoding="utf-8")
    write_compact_jsonl(hash_dir / "chats" / "session-1.jsonl", [{"sessionId": "g-1"}])

    adapter = GeminiAdapter()
    orphans = adapter.find_orphans()
    sessions = list(adapter.discover())

    assert not any(hash_dir in orphan.paths for orphan in orphans)
    assert [s.is_orphan for s in sessions] == [False]


@needs_permissions
def test_gemini_never_calls_a_marker_it_cannot_read_missing(tmp_path, monkeypatch):
    """An unreadable ``.project_root`` is doubt, not an absent marker."""
    monkeypatch.setattr(gemini_mod, "home", lambda: tmp_path)
    hash_dir = tmp_path / ".gemini" / "tmp" / "abc123"
    (hash_dir / "chats").mkdir(parents=True)
    marker = hash_dir / ".project_root"
    marker.write_text("/somewhere/alive\n", encoding="utf-8")
    os.chmod(marker, 0o000)
    try:
        orphans = GeminiAdapter().find_orphans()
    finally:
        os.chmod(marker, 0o644)
    assert not any(hash_dir in orphan.paths for orphan in orphans)


@needs_permissions
def test_codex_never_calls_an_unreadable_project_an_orphan(tmp_path, monkeypatch, locked):
    """Codex sessions take their orphan flag from the shared base."""
    monkeypatch.setattr(codex_mod, "home", lambda: tmp_path)
    rollout = tmp_path / ".codex" / "sessions" / "2026" / "01" / "01" / "rollout-x.jsonl"
    write_compact_jsonl(rollout, [{"type": "session_meta", "payload": {"id": "c", "cwd": str(locked)}}])
    assert [s.is_orphan for s in CodexAdapter().discover()] == [False]


@needs_permissions
def test_opencode_never_calls_an_unreadable_project_an_orphan(tmp_path, monkeypatch, locked):
    """A session row whose directory sx cannot see is not offered as a dead project's."""
    monkeypatch.setattr(opencode_mod, "home", lambda: tmp_path)
    store = tmp_path / ".local" / "share" / "opencode"
    store.mkdir(parents=True)
    con = sqlite3.connect(store / "opencode.db")
    con.execute(
        "CREATE TABLE session (id text PRIMARY KEY, parent_id text, directory text, "
        "title text, time_created integer, time_updated integer)"
    )
    con.execute(
        "INSERT INTO session VALUES ('ses_1', NULL, ?, 't', 1780000000000, 1780000000000)",
        (str(locked),),
    )
    con.commit()
    con.close()
    assert [s.is_orphan for s in OpencodeAdapter().discover()] == [False]


def test_a_project_on_an_unmounted_volume_is_still_not_dead(tmp_path, monkeypatch):
    """The volume check survives the switch to ``project_state``."""
    adapter, projects = _claude(tmp_path, monkeypatch)
    project = "/Volumes/DefinitelyNotMounted42/proj"
    folder = projects / encode_project_dir(project)
    _transcript(folder, SID, project)
    _age_tree(folder)
    assert not any(folder in orphan.paths for orphan in adapter.find_orphans())


# --- a session store sx cannot read is not an empty one ----------------------------


def _artifacts(tmp_path: Path, sid: str) -> list[Path]:
    """Give ``sid`` an aged entry in every class of session-keyed state Claude keeps."""
    base = tmp_path / ".claude"
    made = [base / sub / sid for sub in ("file-history", "session-env", "tasks")]
    for path in made:
        path.mkdir(parents=True)
        (path / "data").write_text("x", encoding="utf-8")
    security = base / "security" / f"security_warnings_state_{sid}.json"
    security.parent.mkdir(parents=True)
    security.write_text("{}", encoding="utf-8")
    made.append(security)
    for path in made:
        if path.is_dir():
            _age_tree(path)
        else:
            _age(path)
    return made


def _stale_paths(adapter: ClaudeAdapter) -> list[Path]:
    """Every path the scan offers as a gone session's state."""
    return [
        path
        for orphan in adapter.find_orphans()
        if orphan.kind is OrphanKind.STALE_SESSION
        for path in orphan.paths
    ]


def _onto_a_volume(tmp_path: Path, path: Path) -> Path:
    """Move ``path`` onto a "volume" and leave a symbolic link to it in its place.

    Returns:
        Where it lives now; renaming that away "unmounts" the volume.
    """
    volume = tmp_path / "Volumes" / "ext"
    volume.mkdir(parents=True, exist_ok=True)
    target = volume / path.name
    path.rename(target)
    path.symlink_to(target)
    return target


def _project_folder(tmp_path: Path, projects: Path) -> tuple[Path, str]:
    """Return the store folder of a project directory that exists, and the directory."""
    project = tmp_path / "work" / "app"
    project.mkdir(parents=True, exist_ok=True)
    return projects / encode_project_dir(str(project)), str(project)


def _a_live_session(tmp_path: Path, monkeypatch) -> tuple[ClaudeAdapter, Path, list[Path]]:
    """A store holding one session, ``SID``, with its state in every class.

    Returns:
        ``(adapter, the session's project folder, its state)``.
    """
    adapter, projects = _claude(tmp_path, monkeypatch)
    folder, project = _project_folder(tmp_path, projects)
    _transcript(folder, SID, project)
    _age_tree(projects)
    return adapter, folder, _artifacts(tmp_path, SID)


def test_a_store_behind_a_link_to_an_unmounted_volume_hides_its_sessions(tmp_path, monkeypatch):
    """``projects`` links to a volume that is not mounted: its sessions are not gone."""
    adapter, folder, made = _a_live_session(tmp_path, monkeypatch)
    volume = _onto_a_volume(tmp_path, folder.parent)
    assert _stale_paths(adapter) == [], "premise: through a working link, nothing is stale"

    volume.rename(tmp_path / "unplugged")

    assert _stale_paths(adapter) == []
    assert all(path.exists() for path in made)


@pytest.mark.parametrize("target", ["unmounted", "a file"])
def test_a_project_folder_linked_to_no_directory_hides_its_sessions(
    tmp_path, monkeypatch, target
):
    """One project folder is a link whose target is not a directory now."""
    adapter, folder, _ = _a_live_session(tmp_path, monkeypatch)
    volume = _onto_a_volume(tmp_path, folder)
    assert _stale_paths(adapter) == [], "premise: through a working link, nothing is stale"

    volume.rename(tmp_path / "unplugged")
    if target == "a file":
        (tmp_path / "unplugged.txt").write_text("not a folder", encoding="utf-8")
        folder.unlink()
        folder.symlink_to(tmp_path / "unplugged.txt")

    assert _stale_paths(adapter) == []


@needs_permissions
def test_a_projects_store_that_cannot_be_listed_hides_its_sessions(tmp_path, monkeypatch):
    """``projects`` itself is unreadable: every session in it is out of sight."""
    adapter, folder, _ = _a_live_session(tmp_path, monkeypatch)
    projects = folder.parent
    os.chmod(projects, 0o000)
    try:
        offered = _stale_paths(adapter)
    finally:
        os.chmod(projects, 0o755)

    assert offered == []


def test_with_no_projects_store_at_all_every_session_is_gone(tmp_path, monkeypatch):
    """Only a ``projects`` that is definitely absent means there are no sessions."""
    adapter, projects = _claude(tmp_path, monkeypatch)
    projects.rmdir()
    made = _artifacts(tmp_path, SID)

    assert sorted(_stale_paths(adapter)) == sorted(made)


def _store_out_of_sight(tmp_path: Path, folder: Path, how: str) -> Path | None:
    """Put the session store out of sight in one of the ways a scan must treat as doubt.

    Args:
        tmp_path: The test's sandbox.
        folder: A project folder in the store.
        how: Which part goes out of sight, and how.

    Returns:
        The directory made unreadable, for the caller to make readable again.
    """
    if how == "store link dangles":
        _onto_a_volume(tmp_path, folder.parent).rename(tmp_path / "unplugged")
        return None
    if how == "folder link dangles":
        _onto_a_volume(tmp_path, folder).rename(tmp_path / "unplugged")
        return None
    locked = folder.parent if how == "store unreadable" else folder
    os.chmod(locked, 0o000)
    return locked


@pytest.mark.parametrize(
    "how",
    [
        "store link dangles",
        "folder link dangles",
        pytest.param("store unreadable", marks=needs_permissions),
        pytest.param("folder unreadable", marks=needs_permissions),
    ],
)
def test_stale_state_is_not_deleted_while_the_store_is_out_of_sight(tmp_path, monkeypatch, how):
    """The delete asks the scan's question again, and doubt refuses it.

    The scan saw the state of a session that was gone. The session is back by
    the time of the delete, but where ``sx`` cannot see it.
    """
    adapter, projects = _claude(tmp_path, monkeypatch)
    folder, project = _project_folder(tmp_path, projects)
    _transcript(folder, OTHER, project)
    _age_tree(projects)
    made = _artifacts(tmp_path, SID)
    orphans = [o for o in adapter.find_orphans() if o.kind is OrphanKind.STALE_SESSION]
    assert sorted(p for o in orphans for p in o.paths) == sorted(made), "premise: offered"

    _transcript(folder, SID, project)
    _age(folder)
    service = DeleteService({"claude": adapter}, log_path=tmp_path / "ops.log")
    locked = _store_out_of_sight(tmp_path, folder, how)
    try:
        previews = [service.preview_orphan(orphan) for orphan in orphans]
        results = [service.delete_orphan(orphan) for orphan in orphans]
    finally:
        if locked is not None:
            os.chmod(locked, 0o755)

    assert all(path.exists() for path in made)
    for result in (*previews, *results):
        assert result.removed == [] and result.failed
        assert all("may still exist" in reason for reason in result.refused.values())


# --- G1: a delete that fails is reported as failed ---------------------------------


@needs_permissions
def test_a_file_that_cannot_be_unlinked_is_refused_not_removed(tmp_path):
    """``EACCES`` on unlink: the path stays, and the result says so."""
    root = tmp_path / "store"
    sealed = root / "sealed"
    sealed.mkdir(parents=True)
    victim = sealed / "victim.tmp"
    victim.write_text("x", encoding="utf-8")
    adapter = _Synthetic(root)
    os.chmod(sealed, 0o555)
    try:
        result = adapter.delete_paths([victim])
    finally:
        os.chmod(sealed, 0o755)
    assert victim.exists()
    assert victim not in result.removed and victim in result.refused
    assert result.failed and result.freed_bytes == 0


@needs_permissions
def test_a_folder_that_cannot_be_removed_is_refused_not_removed(tmp_path):
    """``rmtree`` fails part-way: the folder is reported, never claimed removed."""
    root = tmp_path / "store"
    folder = root / "folder"
    (folder / "sealed").mkdir(parents=True)
    (folder / "sealed" / "keep.txt").write_text("x", encoding="utf-8")
    adapter = _Synthetic(root)
    os.chmod(folder / "sealed", 0o555)
    try:
        result = adapter.delete_paths([folder])
    finally:
        os.chmod(folder / "sealed", 0o755)
    assert (folder / "sealed" / "keep.txt").exists()
    assert folder not in result.removed and folder in result.refused
    assert result.failed


@needs_permissions
def test_a_path_that_cannot_be_checked_is_refused_not_called_absent(tmp_path):
    """A parent sx cannot traverse hides the path; that is not "does not exist"."""
    root = tmp_path / "store"
    guarded = root / "guarded"
    guarded.mkdir(parents=True)
    hidden = guarded / "victim.tmp"
    hidden.write_text("x", encoding="utf-8")
    adapter = _Synthetic(root)
    os.chmod(guarded, 0o000)
    try:
        result = adapter.delete_paths([hidden])
    finally:
        os.chmod(guarded, 0o755)
    assert hidden.exists()
    assert hidden in result.refused and result.failed


def test_harnesses_say_nothing_about_open_sessions_by_default():
    """``open_session_caveat`` is opt-in: no adapter here keeps silent sessions."""
    for adapter in (ClaudeAdapter(), CodexAdapter(), GeminiAdapter(), OpencodeAdapter()):
        assert getattr(adapter, "open_session_caveat", None) is None


def test_orphan_delete_results_are_logged_as_before(tmp_path, monkeypatch):
    """The op-log record of an orphan delete keeps its shape."""
    adapter, projects = _claude(tmp_path, monkeypatch)
    folder = projects / "-old-folder"
    folder.mkdir()
    _age(folder)
    orphan = _only(adapter.find_orphans(), folder)
    log = tmp_path / "ops.log"

    DeleteService({"claude": adapter}, log_path=log).delete_orphan(orphan)

    entry = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert entry["action"] == "delete_orphan" and entry["removed"] == [str(folder)]


def test_a_change_since_the_scan_is_refused_even_when_backdated(tmp_path, monkeypatch):
    """Only the fingerprint can see this: a file added and dated hours ago."""
    adapter, projects = _claude(tmp_path, monkeypatch)
    project = tmp_path / "work" / "gone"
    folder = projects / encode_project_dir(str(project))
    _transcript(folder, SID, str(project))
    _age_tree(folder)
    orphan = _only(adapter.find_orphans(), folder)

    note = folder / "memory" / "MEMORY.md"
    note.parent.mkdir()
    note.write_text("# written after the scan\n", encoding="utf-8")
    _age_tree(folder, seconds=5 * HOUR)

    result = adapter.delete_orphan(orphan)

    assert note.exists() and "changed since it was listed" in result.refused[folder]
