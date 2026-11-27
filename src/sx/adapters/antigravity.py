"""Adapter for the Google Antigravity CLI (``agy``).

The CLI keeps everything under ``~/.gemini/antigravity-cli/`` (no environment
variable moves it). Each conversation is its own SQLite database:

* ``conversations/<id>.db`` (with ``-wal`` / ``-shm`` while open) — a ``steps``
  table with one row per step; ``step_payload`` is a serialized
  ``gemini_coder.Step`` protobuf, read here without a schema
  (:mod:`sx.protowire`);
* ``brain/<id>/`` — the conversation's artifacts: scratch files, uploads, and
  ``.system_generated/`` (step outputs, logs, subagent state, sometimes Git
  worktrees);
* ``annotations/<id>.pbtxt`` — its title (``title:"…"``), archived flag;
* ``presence/<id>.lock`` — an exclusive ``flock`` held while a CLI has the
  conversation open, taken before the first prompt is sent.

Shared stores index the conversations: ``conversation_summaries.db`` (a row per
conversation with its title and ``file://`` workspace URIs, an FTS5 search
index holding the conversation's text, and ``conversation_parents``),
``history.jsonl`` (prompt history, rows tagged with ``conversationId``) and a
few caches. The index also lists conversations the CLI can import from the
Antigravity IDE; those rows (``app_data_dir`` other than ``antigravity-cli``)
belong to the IDE, whose conversations are encrypted, and are never touched.

Field numbers come from the descriptors embedded in ``agy`` 1.5.18
(``third_party/gemini_coder/proto/trajectory.proto`` and
``third_party/jetski/cortex_pb/cortex.proto``) and were checked against real
conversations. A step type ``sx`` does not render is shown as a labeled
placeholder.

Moving a project is not supported: the workspace path is recorded inside the
binary step records as well as in the indexes, and ``sx`` does not rewrite
them. A move reports these conversations as left behind.
"""

from __future__ import annotations

import codecs
import json
import os
import re
import sqlite3
import stat
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .. import locks
from ..fsguard import MISSING, project_state
from ..model import (
    Capability,
    DeleteResult,
    Message,
    Orphan,
    OrphanKind,
    ProjectLeftovers,
    Role,
    Session,
)
from ..protowire import LEN, ProtoError, parse, timestamp
from ..protowire import Message as Proto
from ..util import DROP, dir_size, home, is_under, rewrite_jsonl, write_text_atomic
from .base import HarnessAdapter

#: The value of ``app_data_dir`` on the CLI's own rows in the shared index.
APP_DATA_DIR = "antigravity-cli"

#: Conversation ids are UUIDs; nothing else is ever turned into a path.
_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

#: Top-level entries of the store that hold one item per conversation. Every
#: other entry is shared state or the CLI's own files, and is protected.
_SESSION_DIRS = frozenset({"conversations", "brain", "annotations", "presence"})

#: Shared entries named even before they exist, so the protection does not
#: depend on what a scan happened to find.
_SHARED_ENTRIES = (
    "bin",
    "builtin",
    "cache",
    "cli.log",
    "conversation_summaries.db",
    "conversation_summaries.db-shm",
    "conversation_summaries.db-wal",
    "crashes",
    "daemon",
    "history.jsonl",
    "implicit",
    "installation_id",
    "jetbox_summaries_proto.pb",
    "jetski_state.pbtxt",
    "log",
    "persistent_terminals",
    "plugin_data",
    "scratch",
    "settings.json",
    "sidecar_data",
    "worktrees",
)

#: ``gemini_coder.Step`` fields.
_STEP_TYPE = 1
_STEP_STATUS = 4
_STEP_METADATA = 5
_STEP_USER_INPUT = 19
_STEP_PLANNER_RESPONSE = 20
_STEP_ERROR_MESSAGE = 24
_STEP_RUN_COMMAND = 28
_STEP_CHECKPOINT = 30
_STEP_ERROR = 31
_STEP_NOTIFY_USER = 94
_STEP_EPHEMERAL_MESSAGE = 103
_STEP_SYSTEM_MESSAGE = 114
_STEP_GENERIC = 140
_STEP_INVOKE_SUBAGENT = 143
_STEP_ASK_QUESTION = 154

#: ``CortexStepType`` values ``sx`` renders.
USER_INPUT = 14
PLANNER_RESPONSE = 15
ERROR_MESSAGE = 17
RUN_COMMAND = 21
CHECKPOINT = 23
NOTIFY_USER = 82
EPHEMERAL_MESSAGE = 90
SYSTEM_MESSAGE = 101
INVOKE_SUBAGENT = 127
GENERIC = 132
ASK_QUESTION = 138

#: Every ``CortexStepType`` in agy 1.5.18, for labeling the steps ``sx`` does
#: not render (``CORTEX_STEP_TYPE_`` dropped, lowercased).
STEP_NAMES = {
    1: "dummy", 2: "finish", 3: "plan_input", 4: "mquery", 5: "code_action",
    6: "git_commit", 7: "grep_search", 8: "view_file", 9: "list_directory",
    10: "compile", 13: "view_code_item", 14: "user_input", 15: "planner_response",
    17: "error_message", 21: "run_command", 23: "checkpoint", 24: "propose_code",
    25: "find", 27: "suggested_responses", 28: "command_status", 29: "memory",
    31: "read_url_content", 32: "view_content_chunk", 33: "search_web",
    34: "retrieve_memory", 38: "mcp_tool", 39: "manager_feedback",
    40: "tool_call_proposal", 41: "tool_call_choice", 42: "trajectory_choice",
    45: "clipboard", 47: "view_file_outline", 49: "post_pr_review",
    51: "list_resources", 52: "read_resource", 53: "lint_diff",
    54: "find_all_references", 55: "brain_update", 56: "open_browser_url",
    57: "run_extension_code", 59: "proposal_feedback", 60: "trajectory_search",
    61: "execute_browser_javascript", 62: "list_browser_pages",
    63: "capture_browser_screenshot", 64: "click_browser_pixel", 65: "read_terminal",
    66: "capture_browser_console_logs", 67: "read_browser_page", 68: "browser_get_dom",
    73: "code_search", 74: "browser_input", 75: "browser_move_mouse",
    76: "browser_select_option", 77: "browser_scroll_up", 78: "browser_scroll_down",
    79: "browser_click_element", 80: "browser_press_key", 81: "task_boundary",
    82: "notify_user", 83: "code_acknowledgement", 84: "internal_search",
    85: "browser_subagent", 86: "file_change", 87: "move", 88: "browser_scroll",
    89: "knowledge_generation", 90: "ephemeral_message", 91: "generate_image",
    92: "delete_directory", 93: "compile_applet", 94: "install_applet_dependencies",
    95: "install_applet_package", 96: "browser_resize_window",
    97: "browser_drag_pixel_to_pixel", 98: "conversation_history",
    99: "knowledge_artifacts", 100: "send_command_input", 101: "system_message",
    102: "wait", 103: "agency_tool_call", 104: "cider_agent_dummy", 105: "build_cleaner",
    106: "blaze_build_targets", 107: "blaze_test_targets", 108: "set_up_firebase",
    109: "moma", 110: "restart_dev_server", 111: "deploy_firebase", 112: "shell_exec",
    113: "browser_mouse_wheel", 114: "lint_applet", 116: "ki_insertion",
    117: "retrieve_content", 118: "critique", 119: "findings", 120: "browser_mouse_up",
    121: "browser_mouse_down", 122: "workspace_api", 123: "browser_list_network_requests",
    124: "browser_get_network_request", 125: "browser_refresh_page", 126: "edit_notebook",
    127: "invoke_subagent", 128: "write_blob", 129: "read_notebook",
    130: "propose_ai_comments", 131: "start_code_review", 132: "generic",
    133: "set_up_cloud_sql", 134: "execute_notebook", 135: "cloud_sql_update_schema",
    136: "rpc_action", 137: "cloud_sql_execute_sql", 138: "ask_question",
    139: "directory_rules", 140: "tool_search",
}  # fmt: skip

#: ``CortexStepStatus`` values worth a note under a step.
_STATUS_NOTES = {4: "invalid", 5: "cleared", 6: "canceled", 7: "failed", 12: "interrupted"}

#: The tables and columns of a conversation database that ``sx`` relies on.
_CONVERSATION_SCHEMA = {
    "steps": {"idx", "step_type", "status", "step_payload"},
    "trajectory_meta": {"trajectory_id", "cascade_id"},
}

#: The tables and columns of ``conversation_summaries.db`` a delete edits.
_INDEX_SCHEMA = {
    "conversation_summaries": {"conversation_id", "app_data_dir", "parent_conversation_id"},
    "conversations_fts": {"conversation_id"},
    "conversation_parents": {"conversation_id", "parent_id"},
}

#: The summary columns discovery reads.
_SUMMARY_COLUMNS = (
    "conversation_id",
    "title",
    "preview",
    "step_count",
    "last_modified_time",
    "workspace_uris",
    "parent_conversation_id",
    "nesting_depth",
    "app_data_dir",
    "raw_summary",
)

#: How many conversations one delete may cascade into, subagents included.
_MAX_FAMILY = 1000

#: How long ``ps`` / ``lsof`` may take when checking for open conversations.
_SCAN_TIMEOUT_SECONDS = 5.0

#: Sub-string of a path that only the store's shared worktree folder contains.
_SHARED_WORKTREES = "/antigravity-cli/worktrees/"

#: The lines ``agy`` puts at the top of a tool's result for the model; the
#: message's own timestamp already says when the tool ran.
_TIMING_HEADER = re.compile(
    r"\ACreated At: \d{4}-\d\d-\d\dT[^\n]*\n(?:Completed At: \d{4}-\d\d-\d\dT[^\n]*\n)?"
)

#: Characters Go's ``encoding/json`` escapes by default, as it writes them.
_GO_ESCAPES = {
    "<": "\\u003c",
    ">": "\\u003e",
    "&": "\\u0026",
    "\u2028": "\\u2028",
    "\u2029": "\\u2029",
}


def valid_id(value: object) -> bool:
    """Return True if ``value`` is a conversation id ``sx`` will build a path from."""
    return isinstance(value, str) and bool(_ID_RE.match(value))


def uri_to_path(uri: object) -> str | None:
    """Turn a ``file://`` workspace URI into an absolute path.

    Args:
        uri: A URI such as ``file:///Users/me/My%20Project``.

    Returns:
        The decoded path, or ``None`` for anything that is not a local
        ``file`` URI with an absolute path.
    """
    if not isinstance(uri, str) or not uri.startswith("file:"):
        return None
    parts = urlsplit(uri)
    if parts.netloc not in ("", "localhost"):
        return None
    path = unquote(parts.path)
    if not path.startswith("/"):
        return None
    return path.rstrip("/") or "/"


def _workspaces(raw: object) -> list[str]:
    """Decode a ``workspace_uris`` column (a JSON list of URIs) into paths."""
    if not isinstance(raw, str) or not raw:
        return []
    try:
        uris = json.loads(raw)
    except ValueError:
        return []
    if not isinstance(uris, list):
        return []
    out = []
    for uri in uris:
        path = uri_to_path(uri)
        if path is not None and path not in out:
            out.append(path)
    return out


def _parse_time(value: object) -> float | None:
    """Return a summary timestamp (``2026-11-27 02:10:21.05+00:00``) as epoch seconds."""
    if not isinstance(value, str) or not value or value.startswith("0001-"):
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    try:
        return moment.timestamp()
    except OverflowError, OSError, ValueError:
        return None


def _preview(text: str, limit: int) -> str:
    """Shorten ``text`` to ``limit`` characters, marking the cut."""
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _first_line(text: str, limit: int = 80) -> str:
    """Return the first non-empty line of ``text``, shortened."""
    for line in (text or "").splitlines():
        if line.strip():
            return _preview(line, limit)
    return ""


def _args_preview(raw: str, limit: int = 60) -> str:
    """Summarize a tool call's JSON arguments as their first string value."""
    try:
        args = json.loads(raw) if raw else {}
    except ValueError:
        return _preview(raw, limit)
    if isinstance(args, dict):
        for value in args.values():
            if isinstance(value, str) and value.strip():
                return _preview(value.replace("\n", " "), limit)
        return ""
    return _preview(str(args), limit)


def _error_text(error: Proto | None) -> str:
    """Return the most readable message of a ``CortexErrorDetails``."""
    if error is None:
        return ""
    for number in (1, 2, 9, 3):  # user, short, model, full
        text = error.text(number).strip()
        if text:
            return _first_line(text, 300)
    return ""


def _annotation(path: Path) -> tuple[str, bool]:
    """Read ``title`` and ``archived`` from an ``annotations/<id>.pbtxt`` file.

    The file is protobuf text format, written by Go: a string is quoted with
    backslash escapes, and valid UTF-8 is written as it is.

    Args:
        path: The annotation file.

    Returns:
        ``(title, archived)``; ``("", False)`` when the file is missing or
        unreadable.
    """
    try:
        if path.stat().st_size > 64 * 1024:
            return "", False
        text = path.read_text(encoding="utf-8")
    except OSError, UnicodeError:
        return "", False
    title = ""
    match = re.search(r'^\s*title\s*:\s*"((?:[^"\\\n]|\\.)*)"', text, flags=re.MULTILINE)
    if match:
        try:
            raw = codecs.escape_decode(match.group(1).encode("utf-8"))[0]
            title = raw.decode("utf-8", errors="replace")
        except ValueError, UnicodeError:
            title = match.group(1)
    archived = re.search(r"^\s*archived\s*:\s*true\b", text, flags=re.MULTILINE) is not None
    return title, archived


def _go_json(obj: object, *, newline: bool) -> str:
    """Serialize ``obj`` the way Go's ``json.MarshalIndent(v, "", "  ")`` does."""
    text = json.dumps(obj, indent=2, ensure_ascii=False)
    for char, escape in _GO_ESCAPES.items():
        text = text.replace(char, escape)
    return text + ("\n" if newline else "")


def _encode_field(number: int, payload: bytes) -> bytes:
    """Encode one length-delimited protobuf field."""

    def varint(value: int) -> bytes:
        out = bytearray()
        while True:
            byte = value & 0x7F
            value >>= 7
            if value:
                out.append(byte | 0x80)
            else:
                out.append(byte)
                return bytes(out)

    return varint(number << 3 | LEN) + varint(len(payload)) + payload


@dataclass(slots=True)
class _Summary:
    """One CLI row of ``conversation_summaries``, as discovery uses it.

    Attributes:
        title: The generated title.
        preview: The first user message, shortened by Antigravity.
        step_count: How many steps the conversation holds.
        modified: Last modification, as epoch seconds.
        created: Creation, as epoch seconds, from the raw summary.
        workspaces: Workspace paths, first one first.
        parent: The conversation that launched this one as a subagent, if any.
        nesting: How deep in a subagent tree it sits.
        fork_parent: The conversation this one was forked from, if any.
    """

    title: str = ""
    preview: str = ""
    step_count: int = 0
    modified: float | None = None
    created: float | None = None
    workspaces: list[str] = field(default_factory=list)
    parent: str = ""
    nesting: int = 0
    fork_parent: str = ""


@dataclass(slots=True)
class _Index:
    """What the shared index says about the CLI's conversations.

    Attributes:
        rows: Conversation id -> its CLI summary.
        foreign: Ids the index holds for other Antigravity apps (the IDE).
        readable: False when the index exists but could not be read.
    """

    rows: dict[str, _Summary] = field(default_factory=dict)
    foreign: set[str] = field(default_factory=set)
    readable: bool = True


class AntigravityCliAdapter(HarnessAdapter):
    """Browse, export and delete Antigravity CLI conversations.

    Attributes:
        name: Machine identifier (``"antigravity"``).
        display: Human-readable harness name.
        capabilities: Browse, orphan detection and delete; not move.
    """

    name = "antigravity"
    display = "Antigravity CLI"
    capabilities = Capability.BROWSE | Capability.ORPHANS | Capability.DELETE
    fingerprints_orphans = True

    # --- paths -----------------------------------------------------------

    def _store(self) -> Path:
        """Return ``~/.gemini/antigravity-cli``, resolved at call time."""
        return home() / ".gemini" / APP_DATA_DIR

    def _db(self, cid: str) -> Path:
        """Return a conversation's database (``cid`` must be :func:`valid_id`)."""
        return self._store() / "conversations" / f"{cid}.db"

    def _index_path(self) -> Path:
        """Return ``conversation_summaries.db``."""
        return self._store() / "conversation_summaries.db"

    def _history_path(self) -> Path:
        """Return the CLI's prompt history."""
        return self._store() / "history.jsonl"

    def _presence_path(self, cid: str) -> Path:
        """Return a conversation's presence lock."""
        return self._store() / "presence" / f"{cid}.lock"

    def _conversation_paths(self, cid: str) -> list[Path]:
        """Return every path that belongs to one conversation, database first.

        Args:
            cid: A valid conversation id.

        Returns:
            The database and its WAL files, the artifacts folder, the
            annotation and the presence lock — whichever exist.
        """
        store = self._store()
        db = self._db(cid)
        candidates = [
            db,
            db.with_name(db.name + "-wal"),
            db.with_name(db.name + "-shm"),
            store / "brain" / cid,
            store / "annotations" / f"{cid}.pbtxt",
            self._presence_path(cid),
        ]
        return [path for path in candidates if os.path.lexists(path)]

    def store_roots(self) -> list[Path]:
        """Return the one managed directory, ``~/.gemini/antigravity-cli``."""
        return [self._store()]

    def protected_paths(self) -> list[Path]:
        """Return everything in the store that is not one conversation's.

        Only ``conversations/``, ``brain/``, ``annotations/`` and ``presence/``
        hold per-conversation entries; every other entry — the shared index,
        prompt history, settings, caches, logs, plugins, worktrees — is refused
        outright, whether or not a scan found it. The shared files a delete
        edits are rewritten in place, never removed.

        Returns:
            The protected paths.
        """
        store = self._store()
        named = {store / name for name in _SHARED_ENTRIES}
        try:
            named.update(entry for entry in store.iterdir() if entry.name not in _SESSION_DIRS)
        except OSError:
            pass
        return sorted(named)

    def available(self) -> bool:
        """Return True once the CLI has a conversations folder."""
        return (self._store() / "conversations").is_dir()

    def store_notes(self) -> list[str]:
        """Say how many conversations the index holds for other Antigravity apps."""
        foreign = self._read_index().foreign
        if not foreign:
            return []
        return [
            (
                f"{len(foreign)} conversation(s) the CLI indexes from the Antigravity IDE are "
                "not listed: the IDE stores them encrypted"
            )
        ]

    def store_warnings(self) -> list[str]:
        """Warn where ``sx`` cannot tell whether a conversation is open."""
        if locks.FLOCK_LAYOUT is None:
            return [
                (
                    "sx cannot check Antigravity's presence locks on this platform, so every "
                    "conversation counts as open and none can be deleted"
                )
            ]
        return []

    # --- reading the databases -------------------------------------------

    def _connect(self, db: Path) -> sqlite3.Connection | None:
        """Open a database for reading without leaving anything behind.

        A read-only connection to a WAL database creates ``-wal`` and ``-shm``
        files when they are missing, and cannot remove them again. So a
        database without a write-ahead log in use is opened ``immutable``,
        which creates nothing and takes no locks; one whose log holds data is
        opened ``mode=ro``, which reads the log, through the files that are
        already there.

        Args:
            db: The database file.

        Returns:
            A connection, or ``None`` when it cannot be opened.
        """
        try:
            info = os.lstat(db)
        except OSError:
            return None
        if not stat.S_ISREG(info.st_mode):
            return None
        wal = db.with_name(db.name + "-wal")
        try:
            in_use = os.lstat(wal).st_size > 0
        except OSError:
            in_use = False
        mode = "mode=ro" if in_use else "mode=ro&immutable=1"
        try:
            con = sqlite3.connect(f"{db.as_uri()}?{mode}", uri=True, timeout=2.0)
            con.execute("PRAGMA query_only=1")
            return con
        except sqlite3.Error, OSError, ValueError:
            return None

    def _read_index(self) -> _Index:
        """Read the CLI's rows of ``conversation_summaries``.

        Returns:
            An :class:`_Index`; empty when the index does not exist, and
            marked unreadable when it exists but cannot be read.
        """
        index = _Index()
        path = self._index_path()
        if not os.path.lexists(path):
            return index
        con = self._connect(path)
        if con is None:
            index.readable = False
            return index
        try:
            rows = con.execute(
                f"SELECT {', '.join(_SUMMARY_COLUMNS)} FROM conversation_summaries"
            ).fetchall()
        except sqlite3.Error:
            index.readable = False
            return index
        finally:
            con.close()
        for row in rows:
            data = dict(zip(_SUMMARY_COLUMNS, row, strict=True))
            cid = data["conversation_id"]
            if not valid_id(cid):
                continue
            if data["app_data_dir"] not in (APP_DATA_DIR, ""):
                index.foreign.add(cid)
                continue
            raw = Proto.maybe(data["raw_summary"])
            created = timestamp(raw.child(7)) if raw else None
            index.rows[cid] = _Summary(
                title=data["title"] or "",
                preview=data["preview"] or "",
                step_count=int(data["step_count"] or 0),
                modified=_parse_time(data["last_modified_time"]),
                created=created.timestamp() if created else None,
                workspaces=_workspaces(data["workspace_uris"]),
                parent=data["parent_conversation_id"] or "",
                nesting=int(data["nesting_depth"] or 0),
                fork_parent=raw.text(24) if raw else "",
            )
        index.foreign -= set(index.rows)
        return index

    def _probe(self, cid: str) -> _Summary | None:
        """Read what discovery needs from a conversation the index does not list.

        Args:
            cid: A valid conversation id.

        Returns:
            A summary built from the database itself, or ``None`` if it cannot
            be read.
        """
        con = self._connect(self._db(cid))
        if con is None:
            return None
        try:
            count = con.execute("SELECT COUNT(*) FROM steps").fetchone()[0]
            first = con.execute(
                "SELECT step_payload FROM steps WHERE step_type = ? ORDER BY idx LIMIT 1",
                (USER_INPUT,),
            ).fetchone()
        except sqlite3.Error:
            return None
        finally:
            con.close()
        summary = _Summary(step_count=int(count or 0))
        step = Proto.maybe(first[0]) if first else None
        user = step.child(_STEP_USER_INPUT) if step else None
        if user is not None:
            summary.preview = _user_text(user)
            config = user.child(12)  # user_config (CascadeConfig)
            uris = config.texts(12) if config else []  # dynamic_workspace_uris
            summary.workspaces = [p for p in (uri_to_path(u) for u in uris) if p]
            created = timestamp(step.get(_STEP_METADATA, 1)) if step else None
            summary.created = created.timestamp() if created else None
        return summary

    def _conversation_ids(self) -> list[str]:
        """Return the id of every conversation database in the store."""
        folder = self._store() / "conversations"
        try:
            names = os.listdir(folder)
        except OSError:
            return []
        ids = []
        for name in sorted(names):
            if not name.endswith(".db"):
                continue
            cid = name[: -len(".db")]
            if not valid_id(cid):
                continue
            try:
                if stat.S_ISREG(os.lstat(folder / name).st_mode):
                    ids.append(cid)
            except OSError:
                continue
        return ids

    # --- discovery -------------------------------------------------------

    def discover(self) -> Iterator[Session]:
        """Yield one session per top-level conversation that has steps.

        A subagent's conversation is part of the conversation that launched
        it and is not listed on its own, unless that conversation is gone.
        Conversations without steps — Antigravity's ``/resume`` does not
        list them either — are offered as orphans instead.

        Yields:
            Metadata-only sessions.
        """
        ids = self._conversation_ids()
        if not ids:
            return
        index = self._read_index()
        present = set(ids)
        summaries: dict[str, _Summary] = {}
        for cid in ids:
            summary = index.rows.get(cid) or self._probe(cid)
            if summary is not None:
                summaries[cid] = summary
        children: dict[str, list[str]] = {}
        for cid, summary in summaries.items():
            if summary.parent and summary.parent != cid and summary.parent in present:
                children.setdefault(summary.parent, []).append(cid)
        sessions = []
        for cid in ids:
            summary = summaries.get(cid)
            if summary is None or summary.step_count <= 0:
                continue
            if summary.parent and summary.parent != cid and summary.parent in present:
                continue
            sessions.append(self._session(cid, summary, children.get(cid, [])))
        yield from sessions

    def _session(self, cid: str, summary: _Summary, children: list[str]) -> Session:
        """Build the :class:`Session` for one conversation."""
        store = self._store()
        db = self._db(cid)
        files = [db, db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm")]
        size = 0
        newest = summary.modified or 0.0
        born = summary.created
        for path in files:
            try:
                info = os.lstat(path)
            except OSError:
                continue
            size += info.st_size
            if not path.name.endswith("-shm"):  # readers write the -shm too
                newest = max(newest, info.st_mtime)
            if path == db and born is None:
                born = getattr(info, "st_birthtime", None)
        title, archived = _annotation(store / "annotations" / f"{cid}.pbtxt")
        title = title or summary.title or _first_line(summary.preview) or cid
        if archived:
            title = f"{title} (archived)"
        project = summary.workspaces[0] if summary.workspaces else None
        return Session(
            harness=self.name,
            session_id=cid,
            project_path=project,
            title=_preview(title.replace("\n", " "), 80),
            created=datetime.fromtimestamp(born) if born else None,
            modified=datetime.fromtimestamp(newest) if newest else None,
            message_count=None,
            size_bytes=size,
            paths=self._conversation_paths(cid) or [db],
            is_orphan=bool(project) and project_state(project) == MISSING,
            extra={
                "workspaces": list(summary.workspaces),
                "subagents": sorted(children),
                "fork_parent": summary.fork_parent,
            },
        )

    def sessions_for_project(
        self,
        project: Path,
        sessions: list[Session] | None = None,
    ) -> list[Session]:
        """Return the conversations with any workspace folder in ``project``.

        A conversation can span several folders; it is listed under the
        first, but a move of any of them, or the count of what still uses a
        project before its settings are offered for removal, must see it.

        Args:
            project: The project directory to select for.
            sessions: An already-discovered pool to filter; when omitted the
                adapter discovers its own.

        Returns:
            The matching sessions.
        """
        pool = list(self.discover()) if sessions is None else sessions
        found = []
        for session in pool:
            if session.harness != self.name:
                continue
            folders = session.extra.get("workspaces") or (
                [session.project_path] if session.project_path else []
            )
            if any(is_under(folder, project) for folder in folders):
                found.append(session)
        return found

    # --- loading ---------------------------------------------------------

    def load(self, session: Session) -> list[Message]:
        """Read a conversation's steps and render them as messages.

        Args:
            session: The conversation to load.

        Returns:
            The transcript; empty when the database cannot be read.
        """
        if not valid_id(session.session_id):
            return []
        con = self._connect(self._db(session.session_id))
        if con is None:
            return []
        try:
            rows = con.execute(
                "SELECT idx, step_type, status, step_payload FROM steps ORDER BY idx"
            ).fetchall()
        except sqlite3.Error:
            return []
        finally:
            con.close()
        messages: list[Message] = []
        for idx, column_type, status, payload in rows:
            messages.extend(step_messages(idx, column_type, status, payload))
        return messages

    # --- liveness ----------------------------------------------------------

    def _presence(self, cid: str) -> str:
        """Return the state of a conversation's presence lock (:func:`sx.locks.flock_state`)."""
        return locks.flock_state(self._presence_path(cid), layout=locks.FLOCK_LAYOUT)

    def last_activity(self, session: Session) -> datetime | None:
        """Return when the conversation was last written, or now if a CLI has it open.

        The ``-shm`` file is left out: readers write it too, ``sx`` among them.

        Args:
            session: The conversation.

        Returns:
            Now for a held presence lock; ``None`` when the lock cannot be read
            (treated as active) or nothing of the conversation is left; else the
            newest write to its database or write-ahead log.
        """
        cid = session.session_id
        if not valid_id(cid):
            return None
        state = self._presence(cid)
        if state == locks.HELD:
            return datetime.now()
        if state == locks.UNKNOWN:
            return None
        db = self._db(cid)
        newest = None
        for path in (db, db.with_name(db.name + "-wal")):
            try:
                mtime = os.lstat(path).st_mtime
            except OSError:
                continue
            newest = mtime if newest is None else max(newest, mtime)
        return datetime.fromtimestamp(newest) if newest is not None else None

    def _open_elsewhere(self, ids: list[str]) -> str | None:
        """Return why one of ``ids`` is open in a running Antigravity process, if one is.

        Args:
            ids: Conversation ids.

        Returns:
            A refusal reason, or ``None`` when no process has any of them open
            (:meth:`_open_ids`).
        """
        found, blind = self._open_ids(ids)
        if blind is not None:
            return blind
        for cid in ids:
            if cid in found:
                return f"Antigravity process {found[cid]} has conversation {cid} open (refused)"
        return None

    def _open_ids(self, ids: list[str]) -> tuple[dict[str, int], str | None]:
        """Find which of ``ids`` a running Antigravity process has open, in one scan.

        Antigravity's processes (the CLI, ``agy acp``, the desktop app and the
        remote-control daemon) are found with ``ps`` and their open files
        listed with ``lsof``; a conversation counts as open when any of them
        has its database, write-ahead log, presence lock or artifacts open.

        Args:
            ids: Conversation ids.

        Returns:
            ``(open, blind)``: each open conversation's id -> the process that
            has it open, and a refusal reason when the processes or their open
            files cannot be seen — then any of ``ids`` may be open.
        """
        processes = locks.list_processes(timeout=_SCAN_TIMEOUT_SECONDS)
        if processes is None:
            return {}, (
                "cannot list the running processes to check this conversation is closed (refused)"
            )
        me, uid = os.getpid(), os.getuid()
        pids = [p for p, u, cmd in processes if u == uid and p != me and _runs_antigravity(cmd)]
        if not pids:
            return {}, None
        files = locks.list_open_files(pids, timeout=_SCAN_TIMEOUT_SECONDS)
        if files is None:
            return {}, (
                "cannot list Antigravity's open files to check this conversation is closed (refused)"
            )
        store = os.path.realpath(self._store())
        prefixes = {
            cid: (
                f"{store}/conversations/{cid}.db",
                f"{store}/presence/{cid}.lock",
                f"{store}/brain/{cid}/",
            )
            for cid in ids
        }
        found: dict[str, int] = {}
        for pid in pids:
            names = files.get(pid)
            if names is None:
                if _alive(pid):
                    return found, (
                        f"cannot see what Antigravity process {pid} has open, so cannot tell "
                        "whether it uses this conversation (refused)"
                    )
                continue
            for name in names:
                for cid, wanted in prefixes.items():
                    if cid not in found and name.startswith(wanted):
                        found[cid] = pid
        return found, None

    # --- the checks a delete makes first -----------------------------------

    def destructive_refusal(self, session: Session, *, rewriting: bool) -> str | None:
        """Return why the conversation must not be deleted now, if it must not.

        Asked again at the moment of the delete, for the conversation and every
        subagent conversation it would take with it. Refused: a held or
        unreadable presence lock; a running Antigravity process with any of
        them open; a database that does not have the tables ``sx`` checked
        against or records another conversation's id; a conversation holding
        Git worktrees; and an index ``sx`` would have to edit that has not the
        layout ``sx`` knows.

        Args:
            session: The conversation about to be deleted.
            rewriting: Unused: moving is not supported.

        Returns:
            A refusal reason, or ``None``.
        """
        cid = session.session_id
        if not valid_id(cid):
            return f"not an Antigravity conversation id: {cid!r} (refused)"
        index = self._read_index()
        family = self._family(cid, index)
        return self._family_refusal(family)

    def _family_refusal(self, family: list[str]) -> str | None:
        """Return why a conversation and its subagents must not be deleted, if any."""
        for member in family:
            state = self._presence(member)
            if state == locks.HELD:
                return f"Antigravity has conversation {member} open (refused)"
            if state == locks.UNKNOWN:
                return (
                    f"cannot tell whether Antigravity has conversation {member} open: "
                    "its presence lock cannot be read (refused)"
                )
        reason = self._open_elsewhere(family)
        if reason is not None:
            return reason
        for member in family:
            reason = self._conversation_problem(member) or self._worktree_problem(member)
            if reason is not None:
                return reason
        return self._index_problem()

    def _conversation_problem(self, cid: str) -> str | None:
        """Return why a conversation database is not one ``sx`` may delete, if it is not."""
        db = self._db(cid)
        if not os.path.lexists(db):
            return None
        try:
            if not stat.S_ISREG(os.lstat(db).st_mode):
                return f"{db.name} is not a regular file (refused)"
        except OSError as exc:
            return f"cannot check {db.name}: {exc} (refused)"
        con = self._connect(db)
        if con is None:
            return f"cannot open {db.name} as a database (refused)"
        try:
            problem = _schema_problem(con, _CONVERSATION_SCHEMA)
            if problem is None:
                ids = {row[0] for row in con.execute("SELECT cascade_id FROM trajectory_meta")}
                if ids - {cid, "", None}:
                    problem = "records another conversation's id"
        except sqlite3.Error as exc:
            problem = f"cannot be read ({exc})"
        finally:
            con.close()
        if problem is None:
            return None
        return f"{db.name} {problem}; the format may have changed (refused)"

    def _worktree_problem(self, cid: str) -> str | None:
        """Return why a conversation's Git worktrees keep it from being deleted, if any.

        A subagent can work in a Git worktree of one of your repositories. Its
        folder is in the conversation's artifacts (older CLIs) or in the store's
        shared ``worktrees/`` folder, and the repository keeps a record of it
        that only ``git worktree`` removes, so ``sx`` leaves such conversations
        to Antigravity's own delete.

        Args:
            cid: A valid conversation id.

        Returns:
            A refusal reason, or ``None``.
        """
        own = self._store() / "brain" / cid / ".system_generated" / "worktrees"
        try:
            if own.is_dir():
                with os.scandir(own) as entries:
                    holds = any(True for _ in entries)
            else:
                holds = False
            if holds:
                return (
                    f"conversation {cid} holds Git worktrees; delete it in Antigravity "
                    "(/resume, then F4), which also removes them from their repositories (refused)"
                )
        except OSError as exc:
            return f"cannot check {own}: {exc} (refused)"
        con = self._connect(self._db(cid))
        if con is None:
            return None
        try:
            hit = con.execute(
                "SELECT 1 FROM steps WHERE instr(step_payload, ?) > 0 LIMIT 1",
                (_SHARED_WORKTREES.encode(),),
            ).fetchone()
        except sqlite3.Error:
            hit = None
        finally:
            con.close()
        if hit:
            return (
                f"conversation {cid} used Git worktrees in Antigravity's worktrees folder; "
                "delete it in Antigravity (/resume, then F4), which also removes them (refused)"
            )
        return None

    def _index_problem(self) -> str | None:
        """Return why the shared index cannot be edited, if it cannot."""
        path = self._index_path()
        if not os.path.lexists(path):
            return None
        con = self._connect(path)
        if con is None:
            return "cannot open conversation_summaries.db (refused)"
        try:
            problem = _schema_problem(con, _INDEX_SCHEMA)
        except sqlite3.Error as exc:
            problem = f"cannot be read ({exc})"
        finally:
            con.close()
        if problem is None:
            return None
        return f"conversation_summaries.db {problem}; the format may have changed (refused)"

    def _family(self, cid: str, index: _Index) -> list[str]:
        """Return the subagent conversations ``cid`` launched, deepest first, then ``cid``.

        A subagent is a conversation the shared index lists with ``cid`` as
        its parent, and so on down. The ids in ``cid``'s own launch steps are
        not followed: a fork starts as a copy of the original's steps, launch
        steps included, and the subagents those name belong to the original.
        Forks themselves are separate conversations and never included. Only
        conversations with a database in the store count.

        Args:
            cid: A valid conversation id.
            index: The shared index, as read for this delete.

        Returns:
            Ids in deletion order: every subagent before the conversation that
            launched it.
        """
        kids_of: dict[str, set[str]] = {}
        for child, summary in index.rows.items():
            if summary.parent and summary.parent != child:
                kids_of.setdefault(summary.parent, set()).add(child)
        order: list[str] = []
        seen = {cid}

        def visit(member: str) -> None:
            for kid in sorted(kids_of.get(member, set())):
                if kid in seen or len(seen) >= _MAX_FAMILY:
                    continue
                if not valid_id(kid) or not os.path.lexists(self._db(kid)):
                    continue
                seen.add(kid)
                visit(kid)
            order.append(member)

        visit(cid)
        return order

    def _launched(self, cid: str) -> set[str]:
        """Return the conversation ids ``cid``'s subagent-launch steps recorded."""
        con = self._connect(self._db(cid))
        if con is None:
            return set()
        try:
            rows = con.execute(
                "SELECT step_payload FROM steps WHERE step_type = ?", (INVOKE_SUBAGENT,)
            ).fetchall()
        except sqlite3.Error:
            return set()
        finally:
            con.close()
        found: set[str] = set()
        for (payload,) in rows:
            step = Proto.maybe(payload)
            launch = step.child(_STEP_INVOKE_SUBAGENT) if step else None
            if launch is None:
                continue
            found.update(c for c in launch.texts(5) if valid_id(c))
            for result in launch.children(10):
                found.update(c for c in result.texts(1) if valid_id(c))
        found.discard(cid)
        return found

    def _unclaimed_launches(self, family: list[str], index: _Index) -> list[str]:
        """Return conversations the family launched that the index does not tie to it.

        A launch step names a conversation the index lists with no parent, or
        does not list at all: possibly one of the family's subagents, but not
        one ``sx`` can tell from a copy a fork carries, so it stays, and the
        preview says so. One the index ties to another conversation is that
        conversation's and is not mentioned.

        Args:
            family: The conversations being deleted.
            index: The shared index, as read for this delete.

        Returns:
            The ids, sorted.
        """
        members = set(family)
        found: set[str] = set()
        for member in family:
            for kid in self._launched(member):
                summary = index.rows.get(kid)
                if kid in members or (summary is not None and summary.parent):
                    continue
                if kid in index.foreign or not os.path.lexists(self._db(kid)):
                    continue
                found.add(kid)
        return sorted(found)

    def _forks_of(self, cid: str, index: _Index) -> list[str]:
        """Return the conversations forked from ``cid`` that are still in the store."""
        return sorted(
            other
            for other, summary in index.rows.items()
            if summary.fork_parent == cid and other != cid and os.path.lexists(self._db(other))
        )

    # --- delete ----------------------------------------------------------

    def delete(self, session: Session, *, dry_run: bool = False) -> DeleteResult:
        """Delete a conversation, its subagent conversations, and their index entries.

        What goes is what Antigravity's own delete removes: for each
        conversation, the database and its write-ahead log, the artifacts
        folder, the annotation and the presence lock; then its rows in
        ``conversation_summaries.db`` (summary, search index, parent link), its
        prompt-history rows and its entries in the CLI's caches. Subagent
        conversations go first and the one that launched them last, and the
        delete stops at the first file that will not go, so a conversation is
        never left without the subagents it lists.

        Everything is checked again first (:meth:`destructive_refusal`).
        Conversations forked from this one keep their own files, but lose
        whatever they still used from this one's artifacts folder, as with
        Antigravity's own delete; the preview says so.

        Args:
            session: The conversation to delete.
            dry_run: If True, report what would happen and change nothing.

        Returns:
            A :class:`DeleteResult`; ``note`` records the index edits.
        """
        cid = session.session_id
        target = self._db(cid) if valid_id(cid) else Path(f"<{self.name}:{cid}>")
        if not valid_id(cid):
            return DeleteResult(
                dry_run=dry_run, skipped={target: "not a conversation id (refused)"}
            )
        index = self._read_index()
        if not index.readable:
            return DeleteResult(
                dry_run=dry_run,
                skipped={self._index_path(): "cannot read conversation_summaries.db (refused)"},
            )
        family = self._family(cid, index)
        reason = self._family_refusal(family)
        if reason is not None:
            return DeleteResult(dry_run=dry_run, skipped={target: reason})
        if not os.path.lexists(target):
            return DeleteResult(dry_run=dry_run, skipped={target: "does not exist"})
        result = DeleteResult(dry_run=dry_run)
        gone: list[str] = []

        def keep_from(position: int, paths: list[Path], why: str) -> None:
            """Record that this conversation's other files, and every later one's, stay."""
            rest = list(paths)
            for later in family[position + 1 :]:
                rest.extend(self._conversation_paths(later))
            for path in rest:
                result.skipped.setdefault(path, why)

        for position, member in enumerate(family):
            paths = self._conversation_paths(member)
            first = [p for p in paths if p == self._db(member)]
            # Asked again right before the database goes: a CLI that opened
            # the conversation since the checks above would go on writing to a
            # file that is gone.
            state = locks.FREE if dry_run else self._presence(member)
            if state not in (locks.FREE, locks.ABSENT):
                why = (
                    f"not deleted: Antigravity opened conversation {member} while it was "
                    "being deleted (refused)"
                )
                for path in paths:
                    result.skipped.setdefault(path, why)
                keep_from(position, [], why)
                break
            # The database first, on its own: a conversation whose database
            # will not go keeps everything, and so does every conversation
            # after it — the one that launched it among them.
            part = self._delete_paths(first, dry_run=dry_run)
            _merge(result, part)
            if part.refused:
                why = (
                    f"not deleted: {first[0].name} could not be removed, and a conversation "
                    "goes only after the subagent conversations it launched (refused)"
                )
                keep_from(position, [p for p in paths if p not in first], why)
                break
            # Gone as far as Antigravity is concerned: a file that will not go
            # now is reported as refused, and the index forgets it either way.
            _merge(
                result, self._delete_paths([p for p in paths if p not in first], dry_run=dry_run)
            )
            gone.append(member)
        notes = self._forget(gone, dry_run=dry_run, result=result, index=index)
        forks = self._forks_of(cid, index)
        if forks:
            notes.append(
                f"{len(forks)} conversation(s) forked from this one stay, but lose what they "
                "still used from its artifacts folder"
            )
        unclaimed = self._unclaimed_launches(family, index)
        if unclaimed:
            notes.append(
                f"{len(unclaimed)} conversation(s) it launched stay: Antigravity's index does "
                "not list them as its subagents"
            )
        subagents = [member for member in gone if member != cid]
        if subagents:
            verb = "would delete" if dry_run else "deleted"
            notes.insert(0, f"{verb} {len(subagents)} subagent conversation(s) with it")
        result.note = " · ".join(notes) or None
        return result

    def _forget(
        self,
        ids: list[str],
        *,
        dry_run: bool,
        result: DeleteResult,
        index: _Index | None = None,
    ) -> list[str]:
        """Remove ``ids`` from the shared index, the prompt history and the caches.

        An id whose index row another Antigravity app owns — an IDE
        conversation the CLI imported keeps the IDE's row — keeps every index
        row and cache entry of that app; the CLI's own prompt history still
        forgets it.

        Args:
            ids: Conversations whose files are gone (or would be, in a dry run).
            dry_run: If True, count what would change and write nothing.
            result: Where to record an edit that could not be made.
            index: The shared index as read for this delete; read afresh when
                omitted.

        Returns:
            Human-readable notes on what changed.
        """
        if not ids:
            return []
        if index is None:
            index = self._read_index()
        verb = "would remove" if dry_run else "removed"
        notes = []
        own = [cid for cid in ids if cid not in index.foreign]
        if not index.readable:
            result.skipped[self._index_path()] = "cannot read conversation_summaries.db (refused)"
        elif own:
            rows, error = self._drop_index_rows(own, dry_run=dry_run)
            if error is not None:
                result.skipped[self._index_path()] = error
            elif rows:
                notes.append(f"{verb} {rows} index row(s)")
        wanted = set(ids)
        history = self._history_path()
        if history.is_file():
            count, error = rewrite_jsonl(
                history,
                lambda obj: DROP if obj.get("conversationId") in wanted else None,
                dry_run=dry_run,
            )
            if error is not None:
                result.skipped[history] = error
            elif count:
                notes.append(f"{verb} {count} prompt-history row(s)")
        entries = 0
        for path, edit in self._cache_edits(wanted):
            changed, error = edit(dry_run)
            if error is not None:
                result.skipped[path] = error
            entries += changed
        if entries:
            notes.append(f"{verb} {entries} cache entr{'y' if entries == 1 else 'ies'}")
        return notes

    def _drop_index_rows(self, ids: list[str], *, dry_run: bool) -> tuple[int, str | None]:
        """Remove the CLI's rows for ``ids`` from ``conversation_summaries.db``, in one transaction.

        Rows other Antigravity apps own (``app_data_dir`` set to anything but
        ``antigravity-cli``) are kept, even under the same id.

        Args:
            ids: Conversation ids.
            dry_run: If True, count the rows and change nothing.

        Returns:
            ``(rows, error)``; ``error`` is ``None`` on success.
        """
        path = self._index_path()
        if not os.path.lexists(path):
            return 0, None
        marks = ", ".join("?" for _ in ids)
        statements = (
            "conversation_summaries WHERE conversation_id IN ({}) AND app_data_dir IN (?, '')",
            "conversations_fts WHERE conversation_id IN ({})",
            "conversation_parents WHERE conversation_id IN ({})",
        )
        if dry_run:
            con = self._connect(path)
            if con is None:
                return 0, "cannot open conversation_summaries.db (refused)"
            try:
                total = 0
                for clause in statements:
                    args = [*ids, APP_DATA_DIR] if "app_data_dir" in clause else list(ids)
                    query = "SELECT COUNT(*) FROM " + clause.format(marks)
                    total += con.execute(query, args).fetchone()[0]
                return total, None
            except sqlite3.Error as exc:
                return 0, f"cannot read conversation_summaries.db: {exc} (refused)"
            finally:
                con.close()
        con = None
        try:
            # mode=rw: an index that vanished since it was checked is not re-created.
            con = sqlite3.connect(f"{path.as_uri()}?mode=rw", uri=True, timeout=5.0)
            con.execute("PRAGMA busy_timeout=3000")
            con.execute("BEGIN IMMEDIATE")
            total = 0
            for clause in statements:
                args = [*ids, APP_DATA_DIR] if "app_data_dir" in clause else list(ids)
                cursor = con.execute("DELETE FROM " + clause.format(marks), args)
                total += max(cursor.rowcount, 0)
            con.commit()
            return total, None
        except sqlite3.Error as exc:
            if con is not None:
                try:
                    con.rollback()
                except sqlite3.Error:
                    pass
            return 0, f"index rows not removed: {exc}"
        finally:
            if con is not None:
                con.close()

    def _cache_edits(self, ids: set[str]) -> list[tuple[Path, object]]:
        """Return the cache edits that forget ``ids``, as ``(path, edit)`` pairs.

        Each edit takes ``dry_run`` and returns ``(entries changed, error)``.
        """
        cache = self._store() / "cache"
        metadata = cache / "conversation_metadata.json"
        last = cache / "last_conversations.json"

        def owned(entry: object) -> bool:
            summary = entry.get("summary") if isinstance(entry, dict) else None
            app = summary.get("AppDataDir") if isinstance(summary, dict) else None
            return app in (None, "", APP_DATA_DIR)

        def drop_metadata(data: object) -> int:
            conversations = data.get("conversations") if isinstance(data, dict) else None
            if not isinstance(conversations, dict):
                return 0
            hits = [key for key, entry in conversations.items() if key in ids and owned(entry)]
            for key in hits:
                del conversations[key]
            return len(hits)

        def drop_last(data: object) -> int:
            if not isinstance(data, dict):
                return 0
            hits = [key for key, value in data.items() if value in ids]
            for key in hits:
                del data[key]
            return len(hits)

        return [
            (metadata, lambda dry_run: _edit_json(metadata, drop_metadata, dry_run=dry_run)),
            (last, lambda dry_run: _edit_json(last, drop_last, dry_run=dry_run)),
            (
                self._store() / "jetbox_summaries_proto.pb",
                lambda dry_run: self._edit_jetbox(ids, dry_run=dry_run),
            ),
        ]

    def _edit_jetbox(self, ids: set[str], *, dry_run: bool) -> tuple[int, str | None]:
        """Drop ``ids``' entries from ``jetbox_summaries_proto.pb``.

        The file is a list of ``{1: conversation id, 2: summary}`` entries
        under field 1; the others are written back byte for byte.

        Args:
            ids: Conversation ids.
            dry_run: If True, count the entries and write nothing.

        Returns:
            ``(entries removed, error)``.
        """
        path = self._store() / "jetbox_summaries_proto.pb"
        try:
            before = os.lstat(path)
        except FileNotFoundError, NotADirectoryError:
            return 0, None
        except OSError as exc:
            return 0, f"cannot check: {exc}"
        if not stat.S_ISREG(before.st_mode) or before.st_size > 64 * 1024 * 1024:
            return 0, "not a file sx can edit (refused)"
        try:
            data = path.read_bytes()
            fields = parse(data)
        except (OSError, ProtoError) as exc:
            return 0, f"cannot read it as Antigravity writes it: {exc} (refused)"
        kept, dropped = [], 0
        for number, wire, value in fields:
            if number != 1 or wire != LEN:
                return 0, "holds data sx does not know; the format may have changed (refused)"
            entry = Proto.maybe(value)
            if entry is None:
                return 0, "holds an entry sx cannot read; the format may have changed (refused)"
            if entry.text(1) in ids:
                dropped += 1
                continue
            kept.append(_encode_field(1, value))
        if not dropped or dry_run:
            return dropped, None
        error = _write_bytes_atomic(path, b"".join(kept), expect=before)
        return (0, error) if error else (dropped, None)

    # --- orphans -----------------------------------------------------------

    def find_orphans(self) -> list[Orphan]:
        """Find conversations without steps and files their conversation left behind.

        * A conversation with no steps — created when a CLI started, never
          sent a prompt — which Antigravity's ``/resume`` does not list.
        * An empty database, or one without a conversation's tables, such as
          older CLIs re-created after a delete.
        * An annotation or presence lock whose conversation is gone.

        Nothing a running Antigravity process may be using is offered: a held
        presence lock and any process with the files open both exclude it, and
        everything is checked again when it is deleted.

        Returns:
            The orphans.
        """
        store = self._store()
        orphans: list[Orphan] = []
        ids = self._conversation_ids()
        index = self._read_index()
        candidates: list[tuple[str, OrphanKind, str]] = []
        for cid in ids:
            reason = self._empty_reason(cid, index)
            if reason is not None:
                candidates.append((cid, OrphanKind.EMPTY, reason))
        present = set(ids)
        known = present | set(index.rows) | index.foreign
        for folder, suffix in (("annotations", ".pbtxt"), ("presence", ".lock")):
            if not index.readable:
                break  # cannot confirm their conversations are gone
            try:
                names = os.listdir(store / folder)
            except OSError:
                continue
            for name in sorted(names):
                cid = name[: -len(suffix)] if name.endswith(suffix) else ""
                if not valid_id(cid) or cid in known:
                    continue
                if any(c == cid for c, *_ in candidates):
                    continue
                candidates.append(
                    (cid, OrphanKind.STALE_SESSION, "left behind by a conversation that is gone")
                )
        if not candidates:
            return []
        busy, blind = self._open_ids([cid for cid, *_ in candidates])
        if blind is not None:
            return []  # any of them may be open: none is offered
        candidates = [c for c in candidates if c[0] not in busy]
        for cid, kind, reason in candidates:
            if self._presence(cid) not in (locks.FREE, locks.ABSENT):
                continue
            if kind is OrphanKind.STALE_SESSION:
                paths = [
                    p
                    for p in (store / "annotations" / f"{cid}.pbtxt", self._presence_path(cid))
                    if os.path.lexists(p)
                ]
                if os.path.lexists(store / "brain" / cid):
                    continue  # an artifacts folder may be all that is left of it: kept
            else:
                paths = self._conversation_paths(cid)
            if not paths:
                continue
            orphans.append(
                Orphan(
                    harness=self.name,
                    kind=kind,
                    paths=paths,
                    reason=reason,
                    size_bytes=sum(dir_size(p) for p in paths),
                )
            )
        return orphans

    def _empty_reason(self, cid: str, index: _Index) -> str | None:
        """Return why a conversation database counts as empty, if it does.

        Args:
            cid: A valid conversation id.
            index: The shared index, as read for this scan.

        Returns:
            A reason for an empty or table-less database, or a conversation
            with no steps that no subagent names as its parent; else ``None``.
        """
        db = self._db(cid)
        try:
            size = os.lstat(db).st_size
        except OSError:
            return None
        wal = db.with_name(db.name + "-wal")
        try:
            wal_size = os.lstat(wal).st_size
        except OSError:
            wal_size = 0
        if size == 0 and wal_size == 0:
            return "empty conversation database, holding nothing"
        con = self._connect(db)
        if con is None:
            return None
        try:
            tables = {
                row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if not tables:
                return "conversation database with no tables, holding nothing"
            if "steps" not in tables:
                return None  # not a layout sx knows: left alone
            steps = con.execute("SELECT COUNT(*) FROM steps").fetchone()[0]
        except sqlite3.Error:
            return None
        finally:
            con.close()
        if steps:
            return None
        summary = index.rows.get(cid)
        if summary is not None and summary.step_count:
            return None
        if any(child.parent == cid for child in index.rows.values()):
            return None
        return "conversation with no steps; Antigravity's /resume does not list it"

    def _orphan_revalidation(self, orphan: Orphan) -> dict[Path, str]:
        """Refuse an orphan that a conversation or a running process claims now.

        Args:
            orphan: The orphan about to be deleted.

        Returns:
            Path -> refusal reason, for the paths that must stay.
        """
        cid = _orphan_id(orphan)
        if cid is None:
            return dict.fromkeys(orphan.paths, "not an Antigravity conversation's file (refused)")
        state = self._presence(cid)
        if state not in (locks.FREE, locks.ABSENT):
            why = "Antigravity has this conversation open (refused)"
            if state == locks.UNKNOWN:
                why = "cannot tell whether Antigravity has this conversation open (refused)"
            return dict.fromkeys(orphan.paths, why)
        reason = self._open_elsewhere([cid])
        if reason is not None:
            return dict.fromkeys(orphan.paths, reason)
        if orphan.kind is OrphanKind.EMPTY:
            index = self._read_index()
            if not index.readable:
                return dict.fromkeys(
                    orphan.paths, "cannot read conversation_summaries.db (refused)"
                )
            if self._empty_reason(cid, index) is None:
                return dict.fromkeys(orphan.paths, "the conversation has steps now (refused)")
            reason = self._index_problem()
            if reason is not None:
                return dict.fromkeys(orphan.paths, reason)
        elif os.path.lexists(self._db(cid)):
            return dict.fromkeys(orphan.paths, "its conversation exists again (refused)")
        return {}

    def delete_orphan(self, orphan: Orphan, *, dry_run: bool = False) -> DeleteResult:
        """Delete an orphan; for an empty conversation, its index entries too.

        Args:
            orphan: The orphan to remove, as :meth:`find_orphans` listed it.
            dry_run: If True, report what would be removed without removing it.

        Returns:
            A :class:`DeleteResult`.
        """
        result = super().delete_orphan(orphan, dry_run=dry_run)
        cid = _orphan_id(orphan)
        if orphan.kind is OrphanKind.EMPTY and cid is not None and self._db(cid) in result.removed:
            notes = self._forget([cid], dry_run=dry_run, result=result)
            if notes:
                result.note = " · ".join(notes)
        return result

    # --- project-scoped state ----------------------------------------------

    def _settings_path(self) -> Path:
        """Return the CLI's ``settings.json``."""
        return self._store() / "settings.json"

    def project_leftovers(self, project: str) -> ProjectLeftovers | None:
        """Describe what the CLI keeps for a project beyond its conversations.

        That is the folder's entry in ``trustedWorkspaces``, the "last
        conversation here" pointer ``agy -c`` follows, and prompt-history rows
        recorded in the folder.

        Args:
            project: The project directory.

        Returns:
            A :class:`ProjectLeftovers`, or ``None`` when there is nothing.
        """
        leftovers = ProjectLeftovers(project_path=project)
        settings = _read_json(self._settings_path())
        trusted = settings.get("trustedWorkspaces") if isinstance(settings, dict) else None
        if isinstance(trusted, list) and project in trusted:
            leftovers.config_notes.append("Antigravity CLI trusted-folder entry")
        last = _read_json(self._store() / "cache" / "last_conversations.json")
        if isinstance(last, dict) and project in last:
            leftovers.config_notes.append("Antigravity CLI last-conversation pointer")
        history = self._history_path()
        if history.is_file():
            rows, _ = rewrite_jsonl(
                history, lambda obj: DROP if obj.get("workspace") == project else None, dry_run=True
            )
            if rows:
                leftovers.config_notes.append(f"{rows} Antigravity CLI prompt-history row(s)")
        return leftovers if not leftovers.empty else None

    def delete_project_leftovers(
        self, leftovers: ProjectLeftovers, *, dry_run: bool = False
    ) -> DeleteResult:
        """Remove what :meth:`project_leftovers` described.

        Args:
            leftovers: The plan.
            dry_run: If True, report what would happen and change nothing.

        Returns:
            A :class:`DeleteResult` whose note records the edits.
        """
        project = leftovers.project_path
        result = DeleteResult(dry_run=dry_run)
        verb = "would remove" if dry_run else "removed"
        notes = []

        def drop_trust(data: object) -> int:
            trusted = data.get("trustedWorkspaces") if isinstance(data, dict) else None
            if not isinstance(trusted, list) or project not in trusted:
                return 0
            data["trustedWorkspaces"] = [entry for entry in trusted if entry != project]
            return len(trusted) - len(data["trustedWorkspaces"])

        def drop_last(data: object) -> int:
            if not isinstance(data, dict) or project not in data:
                return 0
            del data[project]
            return 1

        settings = self._settings_path()
        changed, error = _edit_json(settings, drop_trust, dry_run=dry_run)
        if error is not None:
            result.skipped[settings] = error
        elif changed:
            notes.append(f"{verb} trusted-folder entry")
        last = self._store() / "cache" / "last_conversations.json"
        changed, error = _edit_json(last, drop_last, dry_run=dry_run)
        if error is not None:
            result.skipped[last] = error
        elif changed:
            notes.append(f"{verb} last-conversation pointer")
        history = self._history_path()
        if history.is_file():
            rows, error = rewrite_jsonl(
                history,
                lambda obj: DROP if obj.get("workspace") == project else None,
                dry_run=dry_run,
            )
            if error is not None:
                result.skipped[history] = error
            elif rows:
                notes.append(f"{verb} {rows} prompt-history row(s)")
        result.note = " · ".join(notes) or None
        return result


# --- rendering ---------------------------------------------------------------


def _user_text(user: Proto) -> str:
    """Return what the user typed, from a ``CortexStepUserInput``."""
    text = user.text(2) or user.text(1)  # user_response, query
    if not text:
        text = "\n".join(t for item in user.children(3) if (t := item.text(1)))
    attachments = len(user.raw(5)) + len(user.raw(9))  # images, media
    if attachments:
        text = f"{text}\n[{attachments} attachment(s)]".strip()
    return text.strip()


def step_messages(idx: int, column_type: object, status: object, payload: object) -> list[Message]:
    """Render one step as zero or more messages.

    Args:
        idx: The step's index, for a placeholder's label.
        column_type: The ``step_type`` column, used when the payload has none.
        status: The ``status`` column.
        payload: The ``step_payload`` blob.

    Returns:
        The messages; a labeled placeholder for a step ``sx`` cannot read.
    """
    step = Proto.maybe(payload)
    if step is None:
        return [Message(Role.OTHER, text=f"[step {idx} could not be read]")]
    kind = step.varint(_STEP_TYPE) or (column_type if isinstance(column_type, int) else 0)
    meta = step.child(_STEP_METADATA)
    ts = timestamp(meta.child(1)) if meta else None
    message = _render(kind, step, meta, ts)
    if message is None:
        return []
    notes = []
    state = step.varint(_STEP_STATUS) or (status if isinstance(status, int) else 0)
    if state in _STATUS_NOTES:
        notes.append(f"[step {_STATUS_NOTES[state]}]")
    error = _error_text(step.child(_STEP_ERROR))
    if error and kind != ERROR_MESSAGE:
        notes.append(f"Error: {error}")
    if notes:
        message.text = "\n".join(part for part in (message.text, *notes) if part)
    return [message]


def _render(kind: int, step: Proto, meta: Proto | None, ts: datetime | None) -> Message | None:
    """Map one step to a message, or ``None`` for a step that shows nothing."""
    if kind == USER_INPUT:
        user = step.child(_STEP_USER_INPUT)
        text = _user_text(user) if user else ""
        return Message(Role.USER, text=text, timestamp=ts) if text else None
    if kind == PLANNER_RESPONSE:
        reply = step.child(_STEP_PLANNER_RESPONSE)
        if reply is None:
            return None
        text = (reply.text(1) or reply.text(8)).strip()
        thinking = reply.text(3).strip() or ("[thinking redacted]" if reply.varint(5) else "")
        calls = [
            f"{call.text(2) or 'tool'}({_args_preview(call.text(3))})" for call in reply.children(7)
        ]
        if not (text or thinking or calls):
            return None
        return Message(
            Role.ASSISTANT,
            text=text,
            thinking=thinking or None,
            tool_summary="; ".join(calls) or None,
            timestamp=ts,
        )
    if kind == GENERIC:
        generic = step.child(_STEP_GENERIC)
        result = generic.child(2) if generic else None
        output = _TIMING_HEADER.sub("", result.text(1), count=1) if result else ""
        return Message(
            Role.TOOL, text=_preview(output, 400), tool_summary=_tool_label(meta), timestamp=ts
        )
    if kind == RUN_COMMAND:
        command = step.child(_STEP_RUN_COMMAND)
        if command is None:
            return None
        line = command.text(23) or command.text(1)
        output = "\n".join(p for p in (command.text(4), command.text(5)) if p)
        code = command.varint(6)
        if code:
            output = f"{output}\n[exit code {code}]".strip()
        return Message(
            Role.TOOL,
            text=_preview(output, 400),
            tool_summary=f"run_command({_preview(line, 60)})",
            timestamp=ts,
        )
    if kind in (SYSTEM_MESSAGE, EPHEMERAL_MESSAGE):
        number = _STEP_SYSTEM_MESSAGE if kind == SYSTEM_MESSAGE else _STEP_EPHEMERAL_MESSAGE
        body = step.child(number)
        text = body.text(1).strip() if body else ""
        return Message(Role.SYSTEM, text=text, timestamp=ts) if text else None
    if kind == ERROR_MESSAGE:
        body = step.child(_STEP_ERROR_MESSAGE)
        text = _error_text(body.child(3) if body else None) or "an error occurred"
        return Message(Role.SYSTEM, text=f"Error: {text}", timestamp=ts)
    if kind == NOTIFY_USER:
        body = step.child(_STEP_NOTIFY_USER)
        text = body.text(2).strip() if body else ""
        return Message(Role.ASSISTANT, text=text, timestamp=ts) if text else None
    if kind == INVOKE_SUBAGENT:
        body = step.child(_STEP_INVOKE_SUBAGENT)
        if body is None:
            return None
        name = body.text(1) or "subagent"
        text = _preview(body.text(2), 400)
        child = next((c for c in body.texts(5) if valid_id(c)), "")
        if child:
            text = f"{text}\n→ conversation {child}".strip()
        return Message(
            Role.ASSISTANT, text=text, tool_summary=f"invoke_subagent({name})", timestamp=ts
        )
    if kind == ASK_QUESTION:
        body = step.child(_STEP_ASK_QUESTION)
        questions = [q.text(1) for q in body.children(1)] if body else []
        text = "\n".join(q for q in questions if q)
        return Message(Role.ASSISTANT, text=text or "[question]", timestamp=ts)
    if kind == CHECKPOINT:
        body = step.child(_STEP_CHECKPOINT)
        summary = _preview(body.text(5), 400) if body else ""
        text = "Conversation compacted" + (f": {summary}" if summary else "")
        return Message(Role.SYSTEM, text=text, timestamp=ts)
    label = STEP_NAMES.get(kind, f"type {kind}")
    if meta is not None and meta.child(4) is not None:
        return Message(
            Role.TOOL, text=f"[{label} step]", tool_summary=_tool_label(meta), timestamp=ts
        )
    return Message(Role.OTHER, text=f"[{label} step]", timestamp=ts)


def _tool_label(meta: Proto | None) -> str:
    """Return ``name(first argument)`` for a step's tool call, from its metadata."""
    call = meta.child(4) if meta else None  # tool_call (ChatToolCall)
    if call is not None and call.text(2):
        return f"{call.text(2)}({_args_preview(call.text(3))})"
    summary = meta.text(30) if meta else ""  # tool_summary
    return _preview(summary, 60) or "tool"


# --- helpers -------------------------------------------------------------------


def _runs_antigravity(command: str) -> bool:
    """Return True if a command line may be one of Antigravity's processes.

    The CLI runs as ``…/antigravity`` (Homebrew's ``agy`` wrapper execs it),
    the desktop app and its helpers from ``Antigravity.app``; a false match
    only costs a look at that process's open files.

    Args:
        command: The command line, as ``ps`` shows it.

    Returns:
        Whether to look at the process's open files.
    """
    lowered = command.lower()
    if "antigravity" in lowered:
        return True
    first = command.split(None, 1)[0] if command.strip() else ""
    return os.path.basename(first) == "agy"


def _alive(pid: int) -> bool:
    """Return True if a process may still exist (signal 0 sends nothing)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _schema_problem(con: sqlite3.Connection, expected: dict[str, set[str]]) -> str | None:
    """Return what is missing from a database's tables, or ``None``."""
    for table, columns in expected.items():
        found = {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
        if not found:
            return f"has no {table} table"
        missing = columns - found
        if missing:
            return f"has no {', '.join(sorted(missing))} column in {table}"
    return None


def _orphan_id(orphan: Orphan) -> str | None:
    """Return the conversation id every path of an orphan belongs to, if they agree."""
    found = set()
    for path in orphan.paths:
        name = path.name
        for suffix in (".db-wal", ".db-shm", ".db", ".pbtxt", ".lock"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        found.add(name)
    if len(found) != 1:
        return None
    cid = found.pop()
    return cid if valid_id(cid) else None


def _merge(result: DeleteResult, part: DeleteResult) -> None:
    """Add what ``part`` removed and skipped to ``result``."""
    result.removed.extend(part.removed)
    result.freed_bytes += part.freed_bytes
    result.skipped.update(part.skipped)


def _read_json(path: Path) -> object:
    """Read a small JSON file, or return ``None``."""
    try:
        if path.stat().st_size > 64 * 1024 * 1024:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError, UnicodeError:
        return None


def _edit_json(path: Path, edit, *, dry_run: bool) -> tuple[int, str | None]:
    """Apply ``edit`` to a JSON file the CLI writes, and write it back as the CLI does.

    The file is rewritten atomically, and abandoned if it changed while it was
    being edited — the CLI may be running.

    Args:
        path: The file.
        edit: Called with the parsed data; changes it in place and returns how
            many entries it changed.
        dry_run: If True, count and write nothing.

    Returns:
        ``(entries changed, error)``.
    """
    try:
        before = os.lstat(path)
    except FileNotFoundError, NotADirectoryError:
        return 0, None
    except OSError as exc:
        return 0, f"cannot check: {exc}"
    if not stat.S_ISREG(before.st_mode):
        return 0, "not a regular file (refused)"
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, ValueError, UnicodeError) as exc:
        return 0, f"cannot read: {exc} (refused)"
    changed = edit(data)
    if not changed or dry_run:
        return changed, None
    error = write_text_atomic(path, _go_json(data, newline=raw.endswith("\n")), expect=before)
    return (0, error) if error else (changed, None)


def _write_bytes_atomic(path: Path, data: bytes, *, expect: os.stat_result) -> str | None:
    """Replace a binary file atomically, abandoning the write if it changed meanwhile."""
    tmp = path.with_name(f".sx-{path.name}.{os.getpid()}.{time.monotonic_ns()}")
    try:
        mode = stat.S_IMODE(expect.st_mode)
        # Owner-only until it takes the original's mode: the copy holds the same data.
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
        with os.fdopen(fd, "wb") as out:
            out.write(data)
        os.chmod(tmp, mode)
        after = os.lstat(path)
        if after.st_size != expect.st_size or after.st_mtime_ns != expect.st_mtime_ns:
            tmp.unlink(missing_ok=True)
            return "changed while being rewritten (refused)"
        os.replace(tmp, path)
        return None
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return f"write failed: {exc}"
