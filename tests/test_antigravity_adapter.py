"""Antigravity CLI adapter: discovery, transcripts, liveness and what reading leaves behind.

Every store is synthetic, under ``tmp_path`` (see :mod:`tests.agy_store`); ``agy``
is never run. A CLI with a conversation open is played by a child process
holding the presence lock, and the process scan is replaced by a fake.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import pytest

from sx import locks
from sx.adapters import antigravity as agy_module
from sx.adapters.antigravity import AntigravityCliAdapter, uri_to_path
from sx.model import Capability, Role
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
    """A project directory that exists, with a space in its name."""
    path = tmp_path / "work" / "My Project"
    path.mkdir(parents=True)
    return path


def _basic(store: Path, project: Path) -> None:
    """One conversation, ``A``, in ``project``, listed in the index."""
    agy.conversation(
        store,
        A,
        [
            agy.user("Summarize this project", workspace=str(project)),
            agy.reply(
                thinking="Looking around first.",
                calls=(("run_command", '{"CommandLine":"ls -la"}'),),
            ),
            agy.tool("run_command", '{"CommandLine":"ls -la"}', "README.md\nsrc"),
            agy.reply("It is a small command-line tool."),
        ],
        title="Project Summary",
    )
    agy.index(
        store,
        [
            {
                "conversation_id": A,
                "title": "Generated title",
                "step_count": 4,
                "workspace_uris": agy.workspaces(str(project)),
                "raw_summary": agy.summary_blob(),
            }
        ],
        fts=[(A, "Summarize this project")],
    )


# --- the store ----------------------------------------------------------------------


def test_the_store_is_the_cli_folder_and_shared_state_is_protected(store):
    """Only the four per-conversation folders are deletable; every shared file is protected."""
    adapter = AntigravityCliAdapter()
    assert adapter.store_roots() == [store]
    assert adapter.available()
    assert Capability.MOVE not in adapter.capabilities
    for name in ("history.jsonl", "settings.json", "conversation_summaries.db", "cache/x.json"):
        path = store / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    result = adapter._delete_paths(
        [store / "history.jsonl", store / "settings.json", store / "cache", store / "log"],
        dry_run=False,
    )
    assert result.removed == []
    assert set(result.refused.values()) == {"protected path (refused)"}
    assert (store / "history.jsonl").exists() and (store / "cache" / "x.json").exists()


def test_not_available_without_a_conversations_folder(home):
    """An install that never started a conversation is not offered."""
    assert not AntigravityCliAdapter().available()


def test_file_uris_become_paths():
    """Percent-encoding is decoded; anything but a local file URI is refused."""
    assert uri_to_path("file:///Users/me/My%20Project") == "/Users/me/My Project"
    assert uri_to_path("file:///Users/me/proj/") == "/Users/me/proj"
    assert uri_to_path("file://localhost/x") == "/x"
    assert uri_to_path("file://server/share") is None
    assert uri_to_path("https://example.com/x") is None
    assert uri_to_path(None) is None


# --- discovery ------------------------------------------------------------------------


def test_a_conversation_is_listed_with_its_title_project_and_times(store, project):
    """The annotation's title wins; the project is the first workspace; created comes from the summary."""
    _basic(store, project)
    [session] = AntigravityCliAdapter().discover()
    assert session.session_id == A
    assert session.title == "Project Summary"
    assert session.project_path == str(project)
    assert session.created == datetime.fromtimestamp(agy.T0)
    assert session.modified is not None
    assert session.paths[0] == store / "conversations" / f"{A}.db"
    assert store / "brain" / A in session.paths
    assert not session.is_orphan


def test_title_falls_back_to_the_index_then_the_first_prompt(store, project):
    """No annotation: the generated title; no title either: the first line of the preview."""
    agy.conversation(store, A, [agy.user("hi")])
    agy.conversation(store, B, [agy.user("hello")])
    agy.index(
        store,
        [
            {"conversation_id": A, "title": "From the index"},
            {"conversation_id": B, "preview": "First line\nsecond line"},
        ],
    )
    titles = {s.session_id: s.title for s in AntigravityCliAdapter().discover()}
    assert titles == {A: "From the index", B: "First line"}


def test_an_archived_conversation_says_so(store):
    """``archived: true`` in the annotation is shown in the title."""
    agy.conversation(store, A, [agy.user("hi")])
    (store / "annotations" / f"{A}.pbtxt").write_text(
        'title:"Old"\narchived:true\n', encoding="utf-8"
    )
    agy.index(store, [{"conversation_id": A}])
    [session] = AntigravityCliAdapter().discover()
    assert session.title == "Old (archived)"


def test_a_conversation_the_index_does_not_list_is_read_from_its_database(store, project):
    """No summary row: steps, preview and workspace come from the conversation itself."""
    agy.conversation(
        store, A, [agy.user("Fix the build", workspace=str(project)), agy.reply("Done")]
    )
    [session] = AntigravityCliAdapter().discover()
    assert session.title == "Fix the build"
    assert session.project_path == str(project)


def test_a_missing_workspace_marks_the_conversation_orphaned(store, tmp_path):
    """A project definitely gone is a dead project, as for every harness."""
    gone = tmp_path / "gone"
    agy.conversation(store, A, [agy.user("hi")])
    agy.index(store, [{"conversation_id": A, "workspace_uris": agy.workspaces(str(gone))}])
    [session] = AntigravityCliAdapter().discover()
    assert session.is_orphan


def test_subagents_are_listed_under_the_conversation_that_launched_them(store):
    """A subagent's conversation is part of its parent; listed on its own only if the parent is gone."""
    agy.conversation(store, A, [agy.user("go"), agy.launch("worker", "do it", B)])
    agy.conversation(store, B, [agy.user("do it")])
    agy.index(
        store,
        [
            {"conversation_id": A},
            {"conversation_id": B, "parent_conversation_id": A, "nesting_depth": 1},
            {
                "conversation_id": C,
                "parent_conversation_id": "ffffffff-0000-4000-8000-00000000000f",
            },
        ],
    )
    agy.conversation(store, C, [agy.user("orphaned child")])
    sessions = {s.session_id: s for s in AntigravityCliAdapter().discover()}
    assert set(sessions) == {A, C}
    assert sessions[A].extra["subagents"] == [B]


def test_empty_and_other_apps_conversations_are_not_listed(store):
    """No steps: not listed (offered as an orphan). The IDE's rows: never listed."""
    agy.conversation(store, EMPTY, [])
    agy.index(
        store,
        [
            {"conversation_id": EMPTY, "step_count": 0},
            {"conversation_id": IDE, "app_data_dir": "antigravity"},
        ],
    )
    adapter = AntigravityCliAdapter()
    assert list(adapter.discover()) == []
    assert adapter.store_notes() == [
        (
            "1 conversation(s) the CLI indexes from the Antigravity IDE are not listed: "
            "the IDE stores them encrypted"
        )
    ]


def test_files_that_are_not_conversations_are_ignored(store):
    """Only ``<uuid>.db`` regular files are conversations."""
    (store / "conversations" / "notes.db").write_bytes(b"")
    target = store / "elsewhere.db"
    target.write_bytes(b"")
    (store / "conversations" / f"{A}.db").symlink_to(target)
    assert list(AntigravityCliAdapter().discover()) == []


# --- transcripts ----------------------------------------------------------------------


def test_the_transcript_renders_every_kind_of_step(store, project):
    """Prompts, replies with thinking and tool calls, tool output, system and error messages."""
    agy.conversation(
        store,
        A,
        [
            agy.user("What is this?", workspace=str(project), images=2),
            agy.reply(
                thinking="Let me look.",
                calls=(
                    ("read_file", '{"Path":"/x/README.md"}'),
                    ("run_command", '{"CommandLine":"ls"}'),
                ),
            ),
            agy.tool("read_file", '{"Path":"/x/README.md"}', "# Title"),
            agy.system("All your subagents have been stopped"),
            agy.error("quota exceeded"),
            agy.launch("researcher", "find the docs", B),
            agy.reply("Here is the answer."),
            agy.step(8),  # VIEW_FILE: not rendered, labeled
        ],
    )
    session = next(AntigravityCliAdapter().discover())
    messages = AntigravityCliAdapter().load(session)
    assert [(m.role, m.text) for m in messages] == [
        (Role.USER, "What is this?\n[2 attachment(s)]"),
        (Role.ASSISTANT, ""),
        (Role.TOOL, "# Title"),
        (Role.SYSTEM, "All your subagents have been stopped"),
        (Role.SYSTEM, "Error: quota exceeded"),
        (Role.ASSISTANT, f"find the docs\n→ conversation {B}"),
        (Role.ASSISTANT, "Here is the answer."),
        (Role.OTHER, "[view_file step]"),
    ]
    assert messages[1].thinking == "Let me look."
    assert messages[1].tool_summary == "read_file(/x/README.md); run_command(ls)"
    assert messages[2].tool_summary == "read_file(/x/README.md)"
    assert messages[5].tool_summary == "invoke_subagent(researcher)"
    assert messages[0].timestamp == datetime.fromtimestamp(agy.T0).astimezone()


def test_a_step_that_ended_badly_says_so(store):
    """A canceled or failed step carries a note; an unreadable payload becomes a placeholder."""
    agy.conversation(store, A, [agy.user("go"), agy.reply("partial", status=6), b"\xff\xff\xff"])
    session = next(AntigravityCliAdapter().discover())
    messages = AntigravityCliAdapter().load(session)
    assert messages[1].text == "partial\n[step canceled]"
    assert messages[2].role is Role.OTHER and messages[2].text == "[step 2 could not be read]"


def test_reading_leaves_nothing_behind(store, project):
    """Opening a closed WAL database read-only would create -wal and -shm files; sx's does not."""
    _basic(store, project)
    before = sorted(os.listdir(store / "conversations")) + sorted(os.listdir(store))
    adapter = AntigravityCliAdapter()
    session = next(adapter.discover())
    adapter.load(session)
    adapter.find_orphans()
    adapter.destructive_refusal(session, rewriting=False)
    after = sorted(os.listdir(store / "conversations")) + sorted(os.listdir(store))
    assert after == before


# --- liveness ---------------------------------------------------------------------------


def test_a_held_presence_lock_is_live_and_refuses_the_delete(store, project):
    """While a CLI holds the lock the conversation is live, and nothing of it is deleted."""
    _basic(store, project)
    adapter = AntigravityCliAdapter()
    session = next(adapter.discover())
    with Holder(store / "presence" / f"{A}.lock"):
        assert adapter._presence(A) == locks.HELD
        assert (datetime.now() - adapter.last_activity(session)).total_seconds() < 5
        result = adapter.delete(session)
    assert result.removed == []
    assert "Antigravity has conversation" in next(iter(result.refused.values()))
    assert (store / "conversations" / f"{A}.db").exists()
    assert adapter._presence(A) == locks.FREE


def test_a_free_lock_falls_back_to_when_the_database_was_written(store, project):
    """No holder: the newest write to the database or its log, never the -shm."""
    _basic(store, project)
    db = store / "conversations" / f"{A}.db"
    past = time.time() - 3600
    os.utime(db, (past, past))
    shm = db.with_name(db.name + "-shm")
    shm.write_bytes(b"")
    adapter = AntigravityCliAdapter()
    session = next(adapter.discover())
    assert abs(adapter.last_activity(session).timestamp() - past) < 1


def test_an_unknown_lock_layout_counts_as_open(store, project, monkeypatch):
    """Where sx cannot read a presence lock, every conversation is open and none is deleted."""
    _basic(store, project)
    monkeypatch.setattr(locks, "FLOCK_LAYOUT", None)
    adapter = AntigravityCliAdapter()
    session = next(adapter.discover())
    assert adapter.last_activity(session) is None
    assert "cannot tell" in adapter.destructive_refusal(session, rewriting=False)
    assert adapter.store_warnings()


def test_a_process_with_the_database_open_refuses_the_delete(store, project, monkeypatch):
    """``agy acp``, the desktop app: any Antigravity process with the files open counts."""
    _basic(store, project)
    real = os.path.realpath(store)
    monkeypatch.setattr(
        locks,
        "list_processes",
        lambda **kw: [
            (4242, os.getuid(), "/Applications/Antigravity.app/Contents/MacOS/Antigravity")
        ],
    )
    monkeypatch.setattr(
        locks, "list_open_files", lambda pids, **kw: {4242: [f"{real}/conversations/{A}.db-wal"]}
    )
    adapter = AntigravityCliAdapter()
    session = next(adapter.discover())
    assert "process 4242 has conversation" in adapter.destructive_refusal(session, rewriting=False)


def test_processes_that_cannot_be_listed_refuse_the_delete(store, project, monkeypatch):
    """Not knowing is refusing."""
    _basic(store, project)
    monkeypatch.setattr(locks, "list_processes", lambda **kw: None)
    adapter = AntigravityCliAdapter()
    session = next(adapter.discover())
    assert "cannot list the running processes" in adapter.destructive_refusal(
        session, rewriting=False
    )


def test_only_antigravity_processes_are_looked_at(store, project, monkeypatch):
    """Other processes are not listed with lsof at all."""
    _basic(store, project)
    asked: list = []
    monkeypatch.setattr(
        locks,
        "list_processes",
        lambda **kw: [
            (1, os.getuid(), "/usr/bin/vim notes"),
            (2, os.getuid(), "/opt/homebrew/bin/agy -c"),
            (3, os.getuid() + 1, "antigravity"),
        ],
    )
    monkeypatch.setattr(
        locks, "list_open_files", lambda pids, **kw: asked.append(pids) or {p: [] for p in pids}
    )
    adapter = AntigravityCliAdapter()
    assert adapter.destructive_refusal(next(adapter.discover()), rewriting=False) is None
    assert asked == [[2]]


def test_the_timing_header_agy_adds_to_tool_results_is_left_out(store):
    """``Created At:`` / ``Completed At:`` repeat the message's own time; other text is kept."""
    header = "Created At: 2026-11-26T12:18:37-06:00\nCompleted At: 2026-11-26T12:18:40-06:00\n"
    agy.conversation(
        store,
        A,
        [
            agy.user("go"),
            agy.tool("read_file", "{}", header + "line 1"),
            agy.tool("run_command", "{}", "Created At: whenever\nkept"),
        ],
    )
    messages = AntigravityCliAdapter().load(next(AntigravityCliAdapter().discover()))
    assert [m.text for m in messages[1:]] == ["line 1", "Created At: whenever\nkept"]
