"""Tests for the SQLite-backed opencode adapter.

Every test builds a synthetic ``opencode.db`` under ``tmp_path`` and points the
adapter at it by monkeypatching :func:`sx.util.home`. The user's real
``~/.local/share/opencode`` is never read or written. ``sqlite3`` is stdlib, so
no extra dependency is required.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import sx.adapters.opencode as opencode_mod
from sx.adapters.opencode import OpencodeAdapter
from sx.model import OrphanKind, Role


# --- synthetic database builder -------------------------------------------

def _opencode_root(home_dir: Path) -> Path:
    """Return (and create) the opencode store under a fake home."""
    root = home_dir / ".local" / "share" / "opencode"
    (root / "storage" / "session_diff").mkdir(parents=True, exist_ok=True)
    return root


def _init_schema(con: sqlite3.Connection) -> None:
    """Create the minimal opencode schema the adapter reads."""
    con.executescript(
        """
        CREATE TABLE session(
            id TEXT, project_id TEXT, parent_id TEXT, slug TEXT,
            directory TEXT, title TEXT, time_created INTEGER,
            time_updated INTEGER, time_archived INTEGER
        );
        CREATE TABLE message(
            id TEXT, session_id TEXT, time_created INTEGER,
            time_updated INTEGER, data TEXT
        );
        CREATE TABLE part(
            id TEXT, message_id TEXT, session_id TEXT,
            time_created INTEGER, time_updated INTEGER, data TEXT
        );
        CREATE TABLE todo(id TEXT, session_id TEXT, data TEXT);
        CREATE TABLE session_share(session_id TEXT, id TEXT);
        CREATE TABLE session_message(session_id TEXT, id TEXT);
        """
    )


def _add_session(
    con: sqlite3.Connection,
    root: Path,
    sid: str,
    *,
    directory: str,
    title: str,
    parent_id: str | None = None,
    turns: list[tuple[str, list[tuple[str, str]]]] | None = None,
    t0: int = 1_780_000_000_000,
    with_sidecar: bool = True,
    extra_rows: bool = True,
) -> Path:
    """Insert a session plus its messages/parts and write the sidecar file.

    Args:
        con: Open writable connection.
        root: The opencode store root (for the sidecar file).
        sid: Session id.
        directory: Project cwd recorded on the session.
        title: Session title.
        parent_id: Non-None marks a sub-agent (child) session.
        turns: List of ``(role, [(part_type, text), ...])`` describing messages.
        t0: Base millisecond timestamp; parts increment from here.
        with_sidecar: Whether to create the ``session_diff/<id>.json`` file.
        extra_rows: Whether to add a todo/share/session_message row (cascade test).

    Returns:
        The sidecar path (whether or not it was created).
    """
    con.execute(
        "INSERT INTO session(id, parent_id, directory, title, time_created, time_updated) "
        "VALUES (?,?,?,?,?,?)",
        (sid, parent_id, directory, title, t0, t0 + 100),
    )
    clock = t0
    for m_index, (role, parts) in enumerate(turns or []):
        mid = f"{sid}_msg{m_index}"
        con.execute(
            "INSERT INTO message(id, session_id, time_created, data) VALUES (?,?,?,?)",
            (mid, sid, clock, json.dumps({"role": role})),
        )
        for p_index, (ptype, text) in enumerate(parts):
            clock += 1
            pid = f"{mid}_part{p_index}"
            data = {"type": ptype}
            if text is not None:
                data["text"] = text
            con.execute(
                "INSERT INTO part(id, message_id, session_id, time_created, data) "
                "VALUES (?,?,?,?,?)",
                (pid, mid, sid, clock, json.dumps(data)),
            )
    if extra_rows:
        con.execute("INSERT INTO todo(id, session_id, data) VALUES (?,?,?)",
                    (f"{sid}_todo", sid, "{}"))
        con.execute("INSERT INTO session_share(session_id, id) VALUES (?,?)",
                    (sid, f"{sid}_share"))
        con.execute("INSERT INTO session_message(session_id, id) VALUES (?,?)",
                    (sid, f"{sid}_sm"))

    sidecar = root / "storage" / "session_diff" / f"{sid}.json"
    if with_sidecar:
        sidecar.write_text("[]", encoding="utf-8")
    return sidecar


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    """An OpencodeAdapter pointed at a fresh synthetic store under tmp_path.

    The adapter does ``from ..util import home``, binding ``home`` into its own
    module namespace, so we patch it there (patching ``sx.util.home`` would not
    affect the already-imported reference).
    """
    monkeypatch.setattr(opencode_mod, "home", lambda: tmp_path)
    _opencode_root(tmp_path)
    return OpencodeAdapter()


def _connect(adapter: OpencodeAdapter) -> sqlite3.Connection:
    """Open a writable connection to the adapter's synthetic db, init schema."""
    db = adapter._db_path()
    con = sqlite3.connect(str(db))
    _init_schema(con)
    return con


# --- tests -----------------------------------------------------------------

def test_available_requires_db(adapter):
    """available() is False until the db file exists, True once created."""
    assert adapter.available() is False
    _connect(adapter).close()
    assert adapter.available() is True


def test_discovery_lists_top_level_skips_children(adapter):
    """Top-level sessions are listed; sub-agent (parent_id) sessions are hidden."""
    con = _connect(adapter)
    _add_session(con, adapter._store(), "ses_top", directory="/proj/a",
                 title="Top session", turns=[("user", [("text", "hi")])])
    _add_session(con, adapter._store(), "ses_child", directory="/proj/a",
                 title="Child", parent_id="ses_top",
                 turns=[("assistant", [("text", "sub")])])
    con.commit()
    con.close()

    sessions = list(adapter.discover())
    assert [s.session_id for s in sessions] == ["ses_top"]
    s = sessions[0]
    assert s.title == "Top session"
    assert s.project_path == "/proj/a"
    assert s.created is not None and s.created.year == 2026
    assert s.primary_path == adapter._sidecar_path("ses_top")


def test_load_reconstructs_transcript_in_order(adapter):
    """Transcript is user text, assistant reasoning(thinking), assistant text — in order."""
    con = _connect(adapter)
    _add_session(
        con, adapter._store(), "ses_x", directory="/proj/b", title="T",
        turns=[
            ("user", [("text", "What is software?")]),
            ("assistant", [
                ("step-start", None),
                ("reasoning", "let me think"),
                ("text", "Software is instructions."),
                ("step-finish", None),
            ]),
        ],
    )
    con.commit()
    con.close()

    session = next(iter(adapter.discover()))
    msgs = adapter.load(session)
    assert len(msgs) == 3  # step-start/step-finish skipped
    assert msgs[0].role is Role.USER and msgs[0].text == "What is software?"
    assert msgs[1].role is Role.ASSISTANT and msgs[1].thinking == "let me think"
    assert msgs[1].text == ""
    assert msgs[2].role is Role.ASSISTANT and msgs[2].text == "Software is instructions."


def test_tool_part_maps_to_tool_role_and_others_skip(adapter):
    """A `tool` part becomes Role.TOOL; snapshot/file parts are skipped."""
    con = _connect(adapter)
    con.execute(
        "INSERT INTO session(id, parent_id, directory, title, time_created, time_updated) "
        "VALUES ('ses_t', NULL, '/p', 'Tools', 1780000000000, 1780000000100)"
    )
    con.execute(
        "INSERT INTO message(id, session_id, time_created, data) "
        "VALUES ('m', 'ses_t', 1780000000000, ?)", (json.dumps({"role": "assistant"}),)
    )
    parts = [
        ("p1", json.dumps({"type": "tool", "tool": "bash",
                           "state": {"output": "hello from shell"}})),
        ("p2", json.dumps({"type": "snapshot"})),
        ("p3", json.dumps({"type": "file", "text": "ignored"})),
    ]
    for i, (pid, data) in enumerate(parts):
        con.execute(
            "INSERT INTO part(id, message_id, session_id, time_created, data) "
            "VALUES (?, 'm', 'ses_t', ?, ?)", (pid, 1780000000001 + i, data)
        )
    con.commit()
    con.close()

    session = next(iter(adapter.discover()))
    msgs = adapter.load(session)
    assert len(msgs) == 1
    assert msgs[0].role is Role.TOOL
    assert msgs[0].tool_summary == "bash"
    assert "hello from shell" in msgs[0].text


def test_cascade_delete_removes_only_target_session(adapter):
    """Deleting D removes all its rows + sidecar; S is fully intact; db survives."""
    con = _connect(adapter)
    sc_d = _add_session(con, adapter._store(), "ses_D", directory="/p/d", title="Del",
                        turns=[("user", [("text", "q")]),
                               ("assistant", [("text", "a")])])
    sc_s = _add_session(con, adapter._store(), "ses_S", directory="/p/s", title="Keep",
                        turns=[("user", [("text", "keep")])])
    con.commit()
    con.close()

    session_d = next(s for s in adapter.discover() if s.session_id == "ses_D")
    result = adapter.delete(session_d, dry_run=False)

    # Re-open to inspect post-delete state.
    con = sqlite3.connect(str(adapter._db_path()))
    rows = lambda q, a=(): con.execute(q, a).fetchone()[0]
    assert rows("SELECT COUNT(*) FROM session WHERE id='ses_D'") == 0
    assert rows("SELECT COUNT(*) FROM message WHERE session_id='ses_D'") == 0
    assert rows("SELECT COUNT(*) FROM part WHERE session_id='ses_D'") == 0
    assert rows("SELECT COUNT(*) FROM todo WHERE session_id='ses_D'") == 0
    assert rows("SELECT COUNT(*) FROM session_share WHERE session_id='ses_D'") == 0
    assert rows("SELECT COUNT(*) FROM session_message WHERE session_id='ses_D'") == 0
    # S untouched.
    assert rows("SELECT COUNT(*) FROM session WHERE id='ses_S'") == 1
    assert rows("SELECT COUNT(*) FROM part WHERE session_id='ses_S'") == 1
    assert rows("SELECT COUNT(*) FROM todo WHERE session_id='ses_S'") == 1
    con.close()

    assert adapter._db_path().exists()      # db file NOT removed
    assert not sc_d.exists()                # D sidecar unlinked
    assert sc_s.exists()                    # S sidecar remains
    assert sc_d in result.removed
    assert adapter._db_path() not in result.removed  # db never reported removed
    assert result.note is not None and "db row" in result.note


def test_dry_run_changes_nothing(adapter):
    """A dry-run lists the sidecar + a row-count note but mutates nothing."""
    con = _connect(adapter)
    sc = _add_session(con, adapter._store(), "ses_p", directory="/p", title="Prev",
                      turns=[("user", [("text", "x")])])
    con.commit()
    con.close()

    session = next(iter(adapter.discover()))
    result = adapter.delete(session, dry_run=True)
    assert result.dry_run is True
    assert sc in result.removed
    assert result.note and "would delete" in result.note

    # Nothing actually changed.
    con = sqlite3.connect(str(adapter._db_path()))
    assert con.execute("SELECT COUNT(*) FROM session WHERE id='ses_p'").fetchone()[0] == 1
    con.close()
    assert sc.exists()


def test_find_orphans_flags_stray_sidecar(adapter):
    """A sidecar with no matching session row is a STRAY_TEMP orphan; delete is file-only."""
    con = _connect(adapter)
    _add_session(con, adapter._store(), "ses_live", directory="/p", title="Live",
                 turns=[("user", [("text", "x")])])
    con.commit()
    con.close()
    ghost = adapter._store() / "storage" / "session_diff" / "ghost.json"
    ghost.write_text("[]", encoding="utf-8")

    orphans = adapter.find_orphans()
    assert len(orphans) == 1
    o = orphans[0]
    assert o.kind is OrphanKind.STRAY_TEMP
    assert o.paths == [ghost]

    result = adapter.delete_orphan(o, dry_run=False)
    assert ghost in result.removed
    assert not ghost.exists()
    # The live session's sidecar is untouched.
    assert adapter._sidecar_path("ses_live").exists()


def test_dead_project_sets_is_orphan(adapter):
    """A session whose directory no longer exists is flagged is_orphan."""
    con = _connect(adapter)
    _add_session(con, adapter._store(), "ses_dead", directory="/nonexistent/xyz",
                 title="Dead", turns=[("user", [("text", "x")])])
    con.commit()
    con.close()

    session = next(iter(adapter.discover()))
    assert session.is_orphan is True


def test_missing_db_is_tolerated(adapter):
    """No db at all: discover/load/find_orphans return empty, never raise."""
    # adapter fixture created the dir but no db file.
    assert adapter.available() is False
    assert list(adapter.discover()) == []
    from sx.model import Session

    fake = Session(harness="opencode", session_id="nope")
    assert adapter.load(fake) == []
    assert adapter.find_orphans() == []


def test_wal_mode_db_is_readable(adapter):
    """A WAL-mode db (opencode's real mode) is discoverable read-only."""
    con = _connect(adapter)
    con.execute("PRAGMA journal_mode=WAL")
    _add_session(con, adapter._store(), "ses_wal", directory="/p", title="Wal",
                 turns=[("user", [("text", "hi")])])
    con.commit()
    con.close()
    sessions = list(adapter.discover())
    assert [s.session_id for s in sessions] == ["ses_wal"]


def test_sidecar_within_store_root(adapter):
    """The sidecar path resolves inside store_roots (delete guard allows it)."""
    from sx.util import is_within

    sc = adapter._sidecar_path("ses_any")
    assert is_within(sc, adapter.store_roots()) is True
    assert is_within(Path("/etc/hosts"), adapter.store_roots()) is False
