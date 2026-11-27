"""Antigravity CLI deletes: what goes, what stays, what refuses, and orphans.

Every store is synthetic (see :mod:`tests.agy_store`); the process scan is a fake
that finds no Antigravity process unless a test says otherwise.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from sx import locks
from sx.adapters import antigravity as agy_module
from sx.adapters.antigravity import AntigravityCliAdapter
from sx.model import OrphanKind
from sx.service import DeleteService, MoveService
from tests import agy_store as agy
from tests.agy_store import EMPTY, IDE, A, B, C, Holder


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake home directory, with no Antigravity process running."""
    root = tmp_path / "home"
    root.mkdir()
    monkeypatch.setattr(agy_module, "home", lambda: root)
    monkeypatch.setattr(locks, "list_processes", lambda **kw: [])
    return root


@pytest.fixture
def store(home: Path) -> Path:
    """An empty ``~/.gemini/antigravity-cli``."""
    return agy.store_dir(home)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A project directory that exists."""
    path = tmp_path / "work" / "proj"
    path.mkdir(parents=True)
    return path


def _aged(*paths: Path) -> None:
    """Backdate paths an hour, out of the orphan quiet window."""
    past = time.time() - 3600
    for path in paths:
        for item in [path, *path.rglob("*")] if path.is_dir() else [path]:
            os.utime(item, (past, past), follow_symlinks=False)


def _populated(store: Path, project: Path) -> None:
    """Conversation ``A`` (with the index, history and caches that mention it), and ``C``."""
    agy.conversation(
        store, A, [agy.user("hello", workspace=str(project)), agy.reply("hi")], title="Hello"
    )
    agy.conversation(store, C, [agy.user("other")], title="Other")
    agy.index(
        store,
        [
            {"conversation_id": A, "workspace_uris": agy.workspaces(str(project))},
            {"conversation_id": C},
            {"conversation_id": IDE, "app_data_dir": "antigravity"},
        ],
        fts=[(A, "hello"), (A, "hi"), (C, "other")],
        parents=[(A, C)],
    )
    history = [
        {"display": "/model", "timestamp": 1, "workspace": str(project), "type": "slash_command"},
        {"display": "hello", "timestamp": 2, "workspace": str(project), "conversationId": A},
        {"display": "other", "timestamp": 3, "workspace": "/elsewhere", "conversationId": C},
    ]
    (store / "history.jsonl").write_text(
        "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in history), encoding="utf-8"
    )
    (store / "cache" / "conversation_metadata.json").write_text(
        agy.go_json(
            {
                "conversations": {
                    A: {"summary": {"ID": A, "Title": "Hello <A&B>"}, "is_internal": False},
                    C: {"summary": {"ID": C, "Title": "Other"}, "is_internal": False},
                }
            },
            newline=False,
        ),
        encoding="utf-8",
    )
    (store / "cache" / "last_conversations.json").write_text(
        agy.go_json({str(project): A, "/elsewhere": C}, newline=False), encoding="utf-8"
    )
    (store / "jetbox_summaries_proto.pb").write_bytes(
        agy.f(1, agy.f(1, A) + agy.f(2, b"\x0a\x02hi"))
        + agy.f(1, agy.f(1, C) + agy.f(2, b"\x0a\x01o"))
    )
    (store / "settings.json").write_text(
        agy.go_json(
            {"model": "m", "trustedWorkspaces": [str(project), "/elsewhere"]}, newline=True
        ),
        encoding="utf-8",
    )


def _session(adapter: AntigravityCliAdapter, cid: str):
    """Return the listed session ``cid``."""
    return next(s for s in adapter.discover() if s.session_id == cid)


# --- what a delete removes ------------------------------------------------------------


def test_a_delete_removes_the_conversation_and_forgets_it_everywhere(store, project):
    """Files, then index rows, prompt history and cache entries; nothing of ``C`` or the IDE."""
    _populated(store, project)
    adapter = AntigravityCliAdapter()
    db = store / "conversations" / f"{A}.db"
    result = adapter.delete(_session(adapter, A))
    assert not result.refused, result.skipped
    for path in (
        db,
        store / "brain" / A,
        store / "annotations" / f"{A}.pbtxt",
        store / "presence" / f"{A}.lock",
    ):
        assert not path.exists(), path
        assert path in result.removed
    assert (store / "conversations" / f"{C}.db").exists()
    index = store / "conversation_summaries.db"
    assert agy.rows_of(index, "SELECT conversation_id FROM conversation_summaries ORDER BY 1") == [
        (IDE,),
        (C,),
    ]
    assert agy.rows_of(index, "SELECT conversation_id FROM conversations_fts") == [(C,)]
    assert agy.rows_of(index, "SELECT * FROM conversation_parents") == []
    history = [json.loads(line) for line in (store / "history.jsonl").read_text().splitlines()]
    assert [row["display"] for row in history] == ["/model", "other"]
    metadata = (store / "cache" / "conversation_metadata.json").read_text(encoding="utf-8")
    assert A not in metadata and metadata == agy.go_json(
        {"conversations": {C: {"summary": {"ID": C, "Title": "Other"}, "is_internal": False}}},
        newline=False,
    )
    assert json.loads((store / "cache" / "last_conversations.json").read_text()) == {
        "/elsewhere": C
    }
    assert (store / "jetbox_summaries_proto.pb").read_bytes() == agy.f(
        1, agy.f(1, C) + agy.f(2, b"\x0a\x01o")
    )
    assert "removed 4 index row(s)" in result.note
    assert "removed 1 prompt-history row(s)" in result.note
    assert "removed 3 cache entries" in result.note


def test_the_caches_keep_the_way_agy_writes_them(store, project):
    """Two-space JSON, Go's HTML escapes, no trailing newline when the file had none."""
    _populated(store, project)
    adapter = AntigravityCliAdapter()
    adapter.delete(_session(adapter, C))
    text = (store / "cache" / "conversation_metadata.json").read_text(encoding="utf-8")
    assert '"Title": "Hello \\u003cA\\u0026B\\u003e"' in text
    assert text.startswith('{\n  "conversations": {\n') and not text.endswith("\n")


def test_a_dry_run_changes_nothing_and_says_what_would_go(store, project):
    """The preview lists the files and counts the index edits."""
    _populated(store, project)
    before = {p: p.stat().st_mtime_ns for p in store.rglob("*") if p.is_file()}
    adapter = AntigravityCliAdapter()
    result = adapter.delete(_session(adapter, A), dry_run=True)
    assert store / "conversations" / f"{A}.db" in result.removed
    assert "would remove 4 index row(s)" in result.note
    assert "would remove 3 cache entries" in result.note
    after = {p: p.stat().st_mtime_ns for p in store.rglob("*") if p.is_file()}
    assert after == before


def test_subagents_go_first_and_the_parent_last(store):
    """The family is what the index lists as subagents, down the tree; never a fork."""
    agy.conversation(store, A, [agy.user("go"), agy.launch("worker", "do it", B)])
    agy.conversation(store, B, [agy.user("do it")])
    agy.conversation(store, C, [agy.user("sub of sub")])
    fork = "f0f0f0f0-0000-4000-8000-0000000000f0"
    agy.conversation(store, fork, [agy.user("forked")])
    agy.index(
        store,
        [
            {"conversation_id": A},
            {"conversation_id": B, "parent_conversation_id": A, "nesting_depth": 1},
            {"conversation_id": C, "parent_conversation_id": B, "nesting_depth": 2},
            {"conversation_id": fork, "raw_summary": agy.summary_blob(fork_parent=A)},
        ],
    )
    adapter = AntigravityCliAdapter()
    result = adapter.delete(_session(adapter, A))
    dbs = [p.name for p in result.removed if p.suffix == ".db"]
    assert dbs == [f"{C}.db", f"{B}.db", f"{A}.db"]
    assert (store / "conversations" / f"{fork}.db").exists()
    assert "deleted 2 subagent conversation(s) with it" in result.note
    assert "1 conversation(s) forked from this one stay" in result.note


def test_a_subagent_held_open_refuses_the_whole_delete(store):
    """Nothing goes if any conversation the delete would take is open."""
    agy.conversation(store, A, [agy.user("go"), agy.launch("worker", "do it", B)])
    agy.conversation(store, B, [agy.user("do it")])
    agy.index(store, [{"conversation_id": A}, {"conversation_id": B, "parent_conversation_id": A}])
    adapter = AntigravityCliAdapter()
    session = _session(adapter, A)
    with Holder(store / "presence" / f"{B}.lock"):
        result = adapter.delete(session)
    assert result.removed == []
    assert B in next(iter(result.refused.values()))
    assert (store / "conversations" / f"{A}.db").exists() and (
        store / "conversations" / f"{B}.db"
    ).exists()


def test_a_database_that_will_not_go_keeps_the_rest(store, monkeypatch):
    """The first conversation whose database cannot be removed stops the delete, whole."""
    agy.conversation(store, A, [agy.user("go"), agy.launch("worker", "do it", B)])
    agy.conversation(store, B, [agy.user("do it")])
    agy.index(store, [{"conversation_id": A}, {"conversation_id": B, "parent_conversation_id": A}])
    adapter = AntigravityCliAdapter()
    session = _session(adapter, A)
    stuck = store / "conversations" / f"{B}.db"
    real_unlink = Path.unlink

    def unlink(self, *args, **kwargs):
        if self == stuck:
            raise PermissionError("busy")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    result = adapter.delete(session)
    assert result.removed == []
    assert stuck in result.refused
    assert all(
        p.exists() for p in (stuck, store / "brain" / B, store / "conversations" / f"{A}.db")
    )
    assert "could not be removed" in result.skipped[store / "conversations" / f"{A}.db"]
    assert agy.rows_of(
        store / "conversation_summaries.db", "SELECT COUNT(*) FROM conversation_summaries"
    ) == [(2,)]


# --- refusals --------------------------------------------------------------------------


def test_worktrees_refuse_the_delete(store, project):
    """Its own worktree folder, or a step that used the shared one: left to agy's own delete."""
    _populated(store, project)
    worktree = store / "brain" / A / ".system_generated" / "worktrees" / "fix-bug"
    worktree.mkdir(parents=True)
    (worktree / ".git").write_text("gitdir: /repo/.git/worktrees/fix-bug\n", encoding="utf-8")
    adapter = AntigravityCliAdapter()
    reason = adapter.destructive_refusal(_session(adapter, A), rewriting=False)
    assert "holds Git worktrees" in reason and "F4" in reason
    agy.conversation(
        store, B, [agy.user("x"), agy.tool("run_command", "{}", f"{store}/worktrees/fix-bug ready")]
    )
    reason = adapter.destructive_refusal(_session(adapter, B), rewriting=False)
    assert "worktrees folder" in reason


@pytest.mark.parametrize(
    ("schema", "cascade", "expected"),
    [
        (agy.CONVERSATION_SCHEMA.replace("`cascade_id` text,", ""), None, "no cascade_id column"),
        ("CREATE TABLE notes (x);", None, "has no steps table"),
        (agy.CONVERSATION_SCHEMA, B, "records another conversation's id"),
    ],
)
def test_a_database_sx_does_not_recognize_is_refused(store, schema, cascade, expected):
    """The tripwire: the tables and the id sx checked, or no delete."""
    agy.conversation(
        store, A, [] if "notes" in schema else [agy.user("x")], schema=schema, cascade_id=cascade
    )
    agy.index(store, [{"conversation_id": A}])
    adapter = AntigravityCliAdapter()
    from sx.model import Session

    session = Session(
        harness="antigravity", session_id=A, paths=[store / "conversations" / f"{A}.db"]
    )
    reason = adapter.destructive_refusal(session, rewriting=False)
    assert expected in reason and reason.endswith("(refused)")


def test_an_index_with_another_layout_refuses_the_delete(store, project):
    """sx edits conversation_summaries.db only in the layout it knows."""
    _populated(store, project)
    import sqlite3

    con = sqlite3.connect(store / "conversation_summaries.db")
    con.execute("DROP TABLE conversation_parents")
    con.commit()
    con.close()
    adapter = AntigravityCliAdapter()
    result = adapter.delete(_session(adapter, A))
    assert result.removed == []
    assert "no conversation_parents table" in next(iter(result.refused.values()))


def test_an_index_that_cannot_be_read_refuses_the_delete(store, project):
    """A corrupt index is not taken for an empty one."""
    _populated(store, project)
    adapter = AntigravityCliAdapter()
    session = _session(adapter, A)
    (store / "conversation_summaries.db").write_bytes(b"not a database at all" * 100)
    result = adapter.delete(session)
    assert result.removed == []
    assert (store / "conversations" / f"{A}.db").exists()


def test_the_service_logs_the_delete(store, project, tmp_path):
    """Through DeleteService, like every harness: the op-log records it."""
    _populated(store, project)
    adapter = AntigravityCliAdapter()
    log = tmp_path / "ops.log"
    service = DeleteService({"antigravity": adapter}, log_path=log)
    result = service.delete(_session(adapter, A))
    assert not result.refused
    entry = json.loads(log.read_text().splitlines()[-1])
    assert (
        entry["action"] == "delete_session"
        and entry["harness"] == "antigravity"
        and entry["id"] == A
    )


# --- orphans -----------------------------------------------------------------------------


def test_an_empty_conversation_is_offered_and_its_delete_forgets_it(store):
    """No steps, nothing holding it: offered whole; deleting it also drops its index entries."""
    agy.conversation(store, EMPTY, [])
    agy.index(store, [{"conversation_id": EMPTY, "step_count": 0}], fts=[(EMPTY, "")])
    _aged(*store.rglob("*"))
    adapter = AntigravityCliAdapter()
    [orphan] = adapter.find_orphans()
    assert orphan.kind is OrphanKind.EMPTY and orphan.fingerprints is not None
    assert {p.name for p in orphan.paths} == {f"{EMPTY}.db", EMPTY, f"{EMPTY}.lock"}
    result = adapter.delete_orphan(orphan)
    assert not result.refused, result.skipped
    assert not (store / "conversations" / f"{EMPTY}.db").exists()
    assert agy.rows_of(
        store / "conversation_summaries.db", "SELECT COUNT(*) FROM conversation_summaries"
    ) == [(0,)]
    assert "removed 2 index row(s)" in result.note


def test_an_empty_conversation_a_cli_has_open_is_not_offered(store):
    """agy creates the conversation at startup, before the first prompt: never offered then."""
    agy.conversation(store, EMPTY, [])
    _aged(*store.rglob("*"))
    with Holder(store / "presence" / f"{EMPTY}.lock"):
        assert AntigravityCliAdapter().find_orphans() == []


def test_an_empty_conversation_that_gains_a_step_is_refused(store):
    """Checked again at delete time: a prompt sent since the scan keeps it."""
    agy.conversation(store, EMPTY, [])
    _aged(*store.rglob("*"))
    adapter = AntigravityCliAdapter()
    [orphan] = adapter.find_orphans()
    import sqlite3

    con = sqlite3.connect(store / "conversations" / f"{EMPTY}.db")
    con.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (0, 14, ?)", (agy.user("hi"),)
    )
    con.commit()
    con.close()
    _aged(*store.rglob("*"))
    result = adapter.delete_orphan(orphan)
    assert result.removed == []
    assert (store / "conversations" / f"{EMPTY}.db").exists()


def test_a_database_with_no_tables_is_offered(store):
    """What older CLIs re-created after a delete: an empty, schema-less database."""
    (store / "conversations" / f"{EMPTY}.db").write_bytes(b"")
    _aged(*store.rglob("*"))
    [orphan] = AntigravityCliAdapter().find_orphans()
    assert orphan.kind is OrphanKind.EMPTY and "holding nothing" in orphan.reason


def test_stale_annotations_and_locks_are_offered_but_not_artifacts(store):
    """Tiny files keyed to a gone conversation go; an artifacts folder keeps its id's files."""
    (store / "annotations" / f"{A}.pbtxt").write_text('title:"gone"\n', encoding="utf-8")
    (store / "presence" / f"{A}.lock").write_bytes(b"")
    (store / "annotations" / f"{B}.pbtxt").write_text('title:"gone too"\n', encoding="utf-8")
    (store / "brain" / B).mkdir()
    _aged(*store.rglob("*"))
    [orphan] = AntigravityCliAdapter().find_orphans()
    assert orphan.kind is OrphanKind.STALE_SESSION
    assert {p.name for p in orphan.paths} == {f"{A}.pbtxt", f"{A}.lock"}


def test_a_database_of_another_layout_is_not_offered(store):
    """Tables sx does not know: not taken for an empty conversation."""
    agy.conversation(store, EMPTY, [], schema="CREATE TABLE notes (x);")
    _aged(*store.rglob("*"))
    assert AntigravityCliAdapter().find_orphans() == []


# --- project-scoped state ---------------------------------------------------------------


def test_the_last_conversation_offers_the_projects_settings(store, project):
    """Trusted folder, the -c pointer and the folder's prompt history; written back as agy does."""
    _populated(store, project)
    adapter = AntigravityCliAdapter()
    leftovers = adapter.project_leftovers(str(project))
    assert leftovers.config_notes == [
        "Antigravity CLI trusted-folder entry",
        "Antigravity CLI last-conversation pointer",
        "2 Antigravity CLI prompt-history row(s)",
    ]
    result = adapter.delete_project_leftovers(leftovers)
    assert not result.refused
    settings = (store / "settings.json").read_text(encoding="utf-8")
    assert settings == agy.go_json(
        {"model": "m", "trustedWorkspaces": ["/elsewhere"]}, newline=True
    )
    assert json.loads((store / "cache" / "last_conversations.json").read_text()) == {
        "/elsewhere": C
    }
    assert [
        json.loads(r)["display"] for r in (store / "history.jsonl").read_text().splitlines()
    ] == ["other"]
    assert adapter.project_leftovers(str(project)) is None


# --- moving a project ---------------------------------------------------------------------


def test_a_move_reports_the_conversations_it_cannot_re_point(store, project, tmp_path):
    """Moving is not supported: each conversation is listed as refused, nothing is touched."""
    _populated(store, project)
    adapter = AntigravityCliAdapter()
    service = MoveService({"antigravity": adapter}, log_path=tmp_path / "ops.log")
    new = tmp_path / "work" / "moved"
    plans = service.plan(project, new)
    assert set(plans) == {"antigravity"}
    plan = plans["antigravity"]
    assert [s.session_id for s in plan.sessions] == [A] and plan.live == []
    [(path, reason)] = plan.blocked.items()
    assert path == store / "conversations" / f"{A}.db"
    assert "cannot be re-pointed by sx" in reason and reason.endswith("(refused)")
    before = {p: p.stat().st_mtime_ns for p in store.rglob("*") if p.is_file()}
    results = service.move(plans)
    assert results["antigravity"].refused == plan.blocked
    assert {p: p.stat().st_mtime_ns for p in store.rglob("*") if p.is_file()} == before


def test_a_row_another_app_owns_survives_the_clis_copy_being_deleted(store):
    """An IDE conversation imported into the CLI: the CLI's files go, the IDE's index row stays."""
    agy.conversation(store, IDE, [agy.user("imported")])
    agy.index(store, [{"conversation_id": IDE, "app_data_dir": "antigravity"}])
    adapter = AntigravityCliAdapter()
    result = adapter.delete(_session(adapter, IDE))
    assert store / "conversations" / f"{IDE}.db" in result.removed
    rows = agy.rows_of(
        store / "conversation_summaries.db",
        "SELECT conversation_id, app_data_dir FROM conversation_summaries",
    )
    assert rows == [(IDE, "antigravity")]


def test_stale_files_are_not_offered_when_their_conversation_cannot_be_ruled_out(store):
    """An id the IDE owns in the index, or an index that cannot be read: nothing is offered."""
    (store / "annotations" / f"{IDE}.pbtxt").write_text('title:"ide"\n', encoding="utf-8")
    agy.index(store, [{"conversation_id": IDE, "app_data_dir": "antigravity"}])
    _aged(*store.rglob("*"))
    assert AntigravityCliAdapter().find_orphans() == []
    (store / "annotations" / f"{A}.pbtxt").write_text('title:"gone"\n', encoding="utf-8")
    (store / "conversation_summaries.db").write_bytes(b"not a database" * 100)
    _aged(*store.rglob("*"))
    assert AntigravityCliAdapter().find_orphans() == []


# --- review regressions ---------------------------------------------------------------


def test_deleting_a_fork_keeps_the_originals_subagents(store):
    """A fork carries a copy of the original's launch steps; the subagents they name stay."""
    launch = agy.launch("worker", "do it", B)
    agy.conversation(store, A, [agy.user("go"), launch])
    agy.conversation(store, B, [agy.user("do it")])
    agy.conversation(store, C, [agy.user("go"), launch, agy.user("only the fork has this")])
    agy.index(
        store,
        [
            {"conversation_id": A},
            {"conversation_id": B, "parent_conversation_id": A, "nesting_depth": 1},
            {"conversation_id": C, "raw_summary": agy.summary_blob(fork_parent=A)},
        ],
    )
    adapter = AntigravityCliAdapter()
    result = adapter.delete(_session(adapter, C))
    assert not (store / "conversations" / f"{C}.db").exists()
    assert (store / "conversations" / f"{A}.db").exists()
    assert (store / "conversations" / f"{B}.db").exists() and (store / "brain" / B).exists()
    assert "subagent" not in (result.note or "")


def test_a_launch_the_index_does_not_confirm_is_kept_and_named(store):
    """A launched conversation with no parent in the index stays; the preview says so."""
    agy.conversation(store, A, [agy.user("go"), agy.launch("worker", "do it", B)])
    agy.conversation(store, B, [agy.user("do it")])
    agy.index(store, [{"conversation_id": A}])
    adapter = AntigravityCliAdapter()
    preview = adapter.delete(_session(adapter, A), dry_run=True)
    assert "1 conversation(s) it launched stay" in preview.note
    adapter.delete(_session(adapter, A))
    assert (store / "conversations" / f"{B}.db").exists()


def _stuck(monkeypatch, path: Path) -> None:
    """Make unlinking ``path`` fail, as a busy file on another platform would."""
    real_unlink = Path.unlink

    def unlink(self, *args, **kwargs):
        if self == path:
            raise PermissionError("busy")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)


def test_a_subagent_that_did_not_go_is_not_reported_deleted(store, monkeypatch):
    """Nothing removed means no "deleted N subagents" note, and the delete counts as failed."""
    agy.conversation(store, A, [agy.user("go")])
    agy.conversation(store, B, [agy.user("do it")])
    agy.index(store, [{"conversation_id": A}, {"conversation_id": B, "parent_conversation_id": A}])
    adapter = AntigravityCliAdapter()
    session = _session(adapter, A)
    _stuck(monkeypatch, store / "conversations" / f"{B}.db")
    result = adapter.delete(session)
    assert result.removed == [] and result.note is None and result.failed


def test_a_partial_delete_counts_only_the_subagents_that_went(store, monkeypatch):
    """Of two subagents, the one whose database will not go is not counted."""
    agy.conversation(store, A, [agy.user("go")])
    agy.conversation(store, B, [agy.user("x")])
    agy.conversation(store, C, [agy.user("y")])
    agy.index(
        store,
        [
            {"conversation_id": A},
            {"conversation_id": B, "parent_conversation_id": A},
            {"conversation_id": C, "parent_conversation_id": A},
        ],
    )
    adapter = AntigravityCliAdapter()
    session = _session(adapter, A)
    _stuck(monkeypatch, store / "conversations" / f"{C}.db")
    result = adapter.delete(session)
    assert not (store / "conversations" / f"{B}.db").exists()
    assert (store / "conversations" / f"{C}.db").exists()
    assert (store / "conversations" / f"{A}.db").exists()
    assert result.note.startswith("deleted 1 subagent conversation(s) with it")


def test_another_apps_search_rows_and_parent_link_are_kept(store):
    """The CLI's copy of an IDE conversation goes; every index row of the IDE's id stays."""
    agy.conversation(store, IDE, [agy.user("imported")])
    agy.index(
        store,
        [{"conversation_id": IDE, "app_data_dir": "antigravity"}],
        fts=[(IDE, "ide text")],
        parents=[(IDE, A)],
    )
    (store / "cache" / "conversation_metadata.json").write_text(
        agy.go_json(
            {"conversations": {IDE: {"summary": {"ID": IDE, "AppDataDir": "antigravity"}}}},
            newline=False,
        ),
        encoding="utf-8",
    )
    adapter = AntigravityCliAdapter()
    result = adapter.delete(_session(adapter, IDE))
    assert store / "conversations" / f"{IDE}.db" in result.removed
    index = store / "conversation_summaries.db"
    assert agy.rows_of(index, "SELECT COUNT(*) FROM conversations_fts") == [(1,)]
    assert agy.rows_of(index, "SELECT COUNT(*) FROM conversation_parents") == [(1,)]
    assert IDE in (store / "cache" / "conversation_metadata.json").read_text()


def test_a_conversation_opened_while_it_is_deleted_keeps_its_database(store, monkeypatch):
    """The presence lock is asked again right before each database goes."""
    agy.conversation(store, A, [agy.user("go")])
    agy.conversation(store, B, [agy.user("x")])
    agy.index(store, [{"conversation_id": A}, {"conversation_id": B, "parent_conversation_id": A}])
    adapter = AntigravityCliAdapter()
    session = _session(adapter, A)
    asked: list[str] = []
    real = AntigravityCliAdapter._presence

    def presence(self, cid):
        asked.append(cid)
        # Free while the delete checks, held from the moment it starts removing A.
        return locks.HELD if asked.count(A) >= 2 else real(self, cid)

    monkeypatch.setattr(AntigravityCliAdapter, "_presence", presence)
    result = adapter.delete(session)
    assert not (store / "conversations" / f"{B}.db").exists()
    assert (store / "conversations" / f"{A}.db").exists() and (store / "brain" / A).exists()
    assert "opened conversation" in result.skipped[store / "conversations" / f"{A}.db"]


def test_the_orphan_scan_lists_processes_once(store, monkeypatch):
    """One open candidate does not make every other candidate scan again."""
    ids = [f"e0e0e0e0-0000-4000-8000-{n:012d}" for n in range(12)]
    for cid in ids:
        agy.conversation(store, cid, [])
    _aged(*store.rglob("*"))
    calls = {"ps": 0, "lsof": 0}
    real = os.path.realpath(store)

    def ps(**kw):
        calls["ps"] += 1
        return [(424242, os.getuid(), "/opt/homebrew/Caskroom/antigravity-cli/1/antigravity")]

    def lsof(pids, **kw):
        calls["lsof"] += 1
        return {424242: [f"{real}/conversations/{ids[0]}.db"]}

    monkeypatch.setattr(locks, "list_processes", ps)
    monkeypatch.setattr(locks, "list_open_files", lsof)
    orphans = AntigravityCliAdapter().find_orphans()
    assert calls == {"ps": 1, "lsof": 1}
    assert len(orphans) == len(ids) - 1


def test_a_conversation_is_found_by_any_of_its_workspaces(store, tmp_path):
    """A move of a second workspace, or the last-session count, sees the conversation."""
    first, second = tmp_path / "one", tmp_path / "two"
    first.mkdir()
    second.mkdir()
    agy.conversation(store, A, [agy.user("go")])
    agy.index(
        store, [{"conversation_id": A, "workspace_uris": agy.workspaces(str(first), str(second))}]
    )
    adapter = AntigravityCliAdapter()
    assert [s.session_id for s in adapter.sessions_for_project(second)] == [A]
    plans = MoveService({"antigravity": adapter}, log_path=tmp_path / "ops.log").plan(
        second, tmp_path / "moved"
    )
    assert [s.session_id for s in plans["antigravity"].sessions] == [A]


def test_rewritten_files_keep_their_owner_only_mode(store, project, monkeypatch):
    """The jetbox cache is owner-only, and so is the copy sx writes, from its creation."""
    _populated(store, project)
    jetbox = store / "jetbox_summaries_proto.pb"
    os.chmod(jetbox, 0o600)
    seen: list[int] = []
    real_chmod = os.chmod

    def chmod(path, mode, *args, **kwargs):
        if Path(path).name.startswith(".sx-jetbox"):
            seen.append(os.stat(path).st_mode & 0o777)  # as created, before the chmod
        return real_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(agy_module.os, "chmod", chmod)
    adapter = AntigravityCliAdapter()
    adapter.delete(_session(adapter, A))
    assert seen == [0o600]
    assert jetbox.stat().st_mode & 0o777 == 0o600


def test_an_index_that_vanished_is_not_re_created(store, project, monkeypatch):
    """The index is opened read-write only if it is there: never made anew, empty."""
    _populated(store, project)
    adapter = AntigravityCliAdapter()
    index = store / "conversation_summaries.db"
    index.unlink()
    monkeypatch.setattr(agy_module.os.path, "lexists", lambda path: True)
    rows, error = adapter._drop_index_rows([A], dry_run=False)
    assert rows == 0 and error is not None
    assert not index.exists()
