"""Builders for synthetic Antigravity CLI stores.

Every database is created with the exact table definitions ``agy`` 1.5.18
uses (copied from a real store), and every step payload is encoded here, by a
tiny protobuf writer, with the field numbers ``agy``'s embedded descriptors
give. Nothing in this module reads a real store.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Self

CONVERSATION_SCHEMA = """
CREATE TABLE `trajectory_meta` (`trajectory_id` text,`cascade_id` text,`trajectory_type` integer,`source` integer,PRIMARY KEY (`trajectory_id`));
CREATE TABLE `steps` (`idx` integer,`step_type` integer NOT NULL DEFAULT 0,`status` integer NOT NULL DEFAULT 0,`has_subtrajectory` numeric NOT NULL DEFAULT false,`metadata` blob,`error_details` blob,`permissions` blob,`task_details` blob,`render_info` blob,`step_payload` blob,`step_format` integer NOT NULL DEFAULT 0,PRIMARY KEY (`idx`));
CREATE INDEX `idx_steps_status` ON `steps`(`status`);
CREATE INDEX `idx_steps_step_type` ON `steps`(`step_type`);
CREATE TABLE `gen_metadata` (`idx` integer,`data` blob,`size` integer NOT NULL DEFAULT 0,PRIMARY KEY (`idx`));
CREATE TABLE `executor_metadata` (`idx` integer,`data` blob,PRIMARY KEY (`idx`));
CREATE TABLE `parent_references` (`idx` integer,`data` blob,PRIMARY KEY (`idx`));
CREATE TABLE `trajectory_metadata_blob` (`id` text DEFAULT "main",`data` blob,PRIMARY KEY (`id`));
CREATE TABLE `battle_mode_infos` (`idx` integer,`data` blob,PRIMARY KEY (`idx`));
"""

INDEX_SCHEMA = """
CREATE TABLE `conversation_summaries` (`conversation_id` text,`title` text NOT NULL DEFAULT "",`preview` text NOT NULL DEFAULT "",`step_count` integer NOT NULL DEFAULT 0,`last_modified_time` datetime NOT NULL,`workspace_uris` text NOT NULL,`status` text NOT NULL DEFAULT "",`source` text NOT NULL DEFAULT "",`project_id` text NOT NULL DEFAULT "",`agent_name` text NOT NULL DEFAULT "",`parent_conversation_id` text NOT NULL DEFAULT "",`nesting_depth` integer NOT NULL DEFAULT 0,`battle_id` text NOT NULL DEFAULT "",`winning_conversation_id` text NOT NULL DEFAULT "",`not_fully_idle` numeric NOT NULL DEFAULT false,`killed` numeric NOT NULL DEFAULT false,`last_user_input_time` datetime NOT NULL,`last_user_input_step_index` integer NOT NULL DEFAULT -1,`app_data_dir` text NOT NULL DEFAULT "",`raw_summary` blob,`group_id` text NOT NULL DEFAULT "",PRIMARY KEY (`conversation_id`));
CREATE INDEX `idx_conversation_summaries_last_user_input_time` ON `conversation_summaries`(`last_user_input_time`);
CREATE INDEX `idx_conversation_summaries_last_modified_time` ON `conversation_summaries`(`last_modified_time`);
CREATE TABLE search_schema_version (version INTEGER PRIMARY KEY);
CREATE VIRTUAL TABLE conversations_fts USING fts5(conversation_id UNINDEXED, chunk_index UNINDEXED, chunk_first_step UNINDEXED, chunk_last_step UNINDEXED, last_modified_time UNINDEXED, step_count UNINDEXED, last_step_created_at UNINDEXED, archived UNINDEXED, user_step_byte_offsets UNINDEXED, agent_step_byte_offsets UNINDEXED, title, workspace, project, user_body, agent_body, tokenize = 'unicode61 remove_diacritics 2', prefix = '1 2 3');
CREATE TABLE conversation_parents (conversation_id TEXT PRIMARY KEY, parent_id TEXT NOT NULL) WITHOUT ROWID;
"""

#: A fixed moment for step timestamps (2026-01-15 12:00:00 UTC).
T0 = 1768478400

A = "a1a1a1a1-0000-4000-8000-000000000001"
B = "b1b1b1b1-0000-4000-8000-000000000002"
C = "c1c1c1c1-0000-4000-8000-000000000003"
EMPTY = "e0e0e0e0-0000-4000-8000-0000000000e0"
IDE = "1de1de1d-0000-4000-8000-0000000001de"


# --- a protobuf writer ---------------------------------------------------------------


def varint(value: int) -> bytes:
    """Encode a non-negative integer as a base-128 varint."""
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def f(number: int, value: int | bool | str | bytes) -> bytes:
    """Encode one field: an integer as a varint, text or bytes length-delimited."""
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, int):
        return varint(number << 3) + varint(value)
    data = value.encode("utf-8") if isinstance(value, str) else value
    return varint(number << 3 | 2) + varint(len(data)) + data


def ts(seconds: int, nanos: int = 0) -> bytes:
    """Encode a ``google.protobuf.Timestamp``."""
    return f(1, seconds) + (f(2, nanos) if nanos else b"")


def step(
    kind: int, body: bytes = b"", *, when: int = T0, status: int = 3, meta: bytes = b""
) -> bytes:
    """Encode a ``gemini_coder.Step`` of type ``kind`` holding ``body``."""
    return f(1, kind) + f(4, status) + f(5, f(1, ts(when)) + meta) + body


def user(text: str, *, workspace: str | None = None, when: int = T0, images: int = 0) -> bytes:
    """A ``USER_INPUT`` step, with the workspace in its config as ``agy`` records it."""
    payload = f(2, text) + f(3, f(1, text))
    payload += b"".join(f(5, f(1, b"png")) for _ in range(images))
    if workspace is not None:
        payload += f(12, f(12, Path(workspace).as_uri()))
    return step(14, f(19, payload), when=when)


def reply(
    text: str = "",
    *,
    thinking: str = "",
    calls: tuple[tuple[str, str], ...] = (),
    when: int = T0,
    status: int = 3,
) -> bytes:
    """A ``PLANNER_RESPONSE`` step: text, thinking and tool calls ``(name, json args)``."""
    payload = (f(1, text) if text else b"") + (f(3, thinking) if thinking else b"")
    for number, (name, args) in enumerate(calls):
        payload += f(7, f(1, f"call_{number}") + f(2, name) + f(3, args))
    return step(15, f(20, payload), when=when, status=status)


def tool(name: str, args: str, output: str, *, when: int = T0) -> bytes:
    """A ``GENERIC`` tool step: the call in its metadata, the output in its result."""
    call = f(1, "call_0") + f(2, name) + f(3, args)
    return step(132, f(140, f(2, f(1, output))), when=when, meta=f(4, call))


def system(text: str, *, when: int = T0) -> bytes:
    """A ``SYSTEM_MESSAGE`` step."""
    return step(101, f(114, f(1, text)), when=when)


def error(text: str, *, when: int = T0) -> bytes:
    """An ``ERROR_MESSAGE`` step."""
    return step(17, f(24, f(3, f(2, text))), when=when)


def launch(name: str, prompt: str, child: str, *, when: int = T0) -> bytes:
    """An ``INVOKE_SUBAGENT`` step naming the child conversation."""
    return step(127, f(143, f(1, name) + f(2, prompt) + f(5, child)), when=when)


def summary_blob(*, created: int = T0, fork_parent: str = "") -> bytes:
    """A ``CascadeTrajectorySummary`` (``raw_summary``): created time and fork parent."""
    return f(7, ts(created)) + (f(24, fork_parent) if fork_parent else b"")


# --- the store -----------------------------------------------------------------------


def store_dir(home: Path) -> Path:
    """Return (and create) ``~/.gemini/antigravity-cli`` with its folders."""
    store = home / ".gemini" / "antigravity-cli"
    for name in ("conversations", "brain", "annotations", "presence", "cache", "log"):
        (store / name).mkdir(parents=True, exist_ok=True)
    return store


def conversation(
    store: Path,
    cid: str,
    steps: list[bytes],
    *,
    cascade_id: str | None = None,
    schema: str = CONVERSATION_SCHEMA,
    brain: bool = True,
    title: str | None = None,
    presence: bool = True,
) -> Path:
    """Write one conversation database (WAL mode, closed cleanly) and its files."""
    db = store / "conversations" / f"{cid}.db"
    con = sqlite3.connect(db)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(schema)
    if "`cascade_id`" in schema:
        con.execute(
            "INSERT INTO trajectory_meta VALUES (?, ?, 4, 17)",
            ("7a7a7a7a-0000-4000-8000-00000000007a", cascade_id or cid),
        )
    for idx, payload in enumerate(steps):
        kind = _type_of(payload) if payload[:1] == b"\x08" else 0
        con.execute(
            "INSERT INTO steps (idx, step_type, status, step_payload) VALUES (?, ?, 3, ?)",
            (idx, kind, payload),
        )
    con.commit()
    con.close()
    if brain:
        (store / "brain" / cid / "scratch").mkdir(parents=True, exist_ok=True)
        (store / "brain" / cid / "scratch" / "notes.md").write_text("notes", encoding="utf-8")
    if title is not None:
        (store / "annotations" / f"{cid}.pbtxt").write_text(f'title:"{title}"\n', encoding="utf-8")
    if presence:
        (store / "presence" / f"{cid}.lock").write_bytes(b"")
    return db


def _type_of(payload: bytes) -> int:
    """Return field 1 of an encoded step (its type), however many bytes it takes."""
    value, shift, pos = 0, 0, 1
    while True:
        byte = payload[pos]
        value |= (byte & 0x7F) << shift
        pos += 1
        if not byte & 0x80:
            return value
        shift += 7


def index(store: Path, rows: list[dict], *, fts: list[tuple[str, str]] = (), parents=()) -> Path:
    """Write ``conversation_summaries.db`` holding ``rows`` (dicts of column values)."""
    path = store / "conversation_summaries.db"
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(INDEX_SCHEMA)
    con.execute("INSERT INTO search_schema_version VALUES (1)")
    for row in rows:
        values = {
            "title": "",
            "preview": "",
            "step_count": 1,
            "last_modified_time": "2026-11-27 02:10:21.052879+00:00",
            "workspace_uris": "[]",
            "last_user_input_time": "2026-11-27 02:10:21.052879+00:00",
            "app_data_dir": "antigravity-cli",
            **row,
        }
        columns = ", ".join(values)
        marks = ", ".join("?" for _ in values)
        con.execute(
            f"INSERT INTO conversation_summaries ({columns}) VALUES ({marks})",
            list(values.values()),
        )
    for cid, body in fts:
        con.execute(
            "INSERT INTO conversations_fts (conversation_id, chunk_index, title, user_body) "
            "VALUES (?, 0, '', ?)",
            (cid, body),
        )
    for child, parent in parents:
        con.execute("INSERT INTO conversation_parents VALUES (?, ?)", (child, parent))
    con.commit()
    con.close()
    return path


def workspaces(*paths: str) -> str:
    """Encode a ``workspace_uris`` column."""
    return json.dumps([Path(p).as_uri() for p in paths])


def go_json(obj: object, *, newline: bool) -> str:
    """Write JSON the way ``agy`` (Go's ``MarshalIndent``) does."""
    return json.dumps(obj, indent=2, ensure_ascii=False) + ("\n" if newline else "")


def rows_of(path: Path, query: str, args=()) -> list[tuple]:
    """Run a read-only query on a database the test wrote."""
    con = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        return con.execute(query, args).fetchall()
    finally:
        con.close()


# --- a CLI with a conversation open ------------------------------------------------------

_HOLDER = r"""
import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX)
print("ready", flush=True)
sys.stdin.read()
"""


class Holder:
    """A child process holding a presence lock, as ``agy`` does while a conversation is open."""

    def __init__(self, lock: Path) -> None:
        """Take ``lock`` in a child process and wait until it is held."""
        self._child = subprocess.Popen(
            [sys.executable, "-I", "-c", _HOLDER, str(lock)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        assert self._child.stdout is not None
        assert self._child.stdout.readline().strip() == "ready"

    def close(self) -> None:
        """Release the lock and wait for the child to exit."""
        assert self._child.stdin is not None and self._child.stdout is not None
        self._child.stdin.close()
        self._child.wait(timeout=10)
        self._child.stdout.close()

    def __enter__(self) -> Self:
        """Hold the lock for a ``with`` block."""
        return self

    def __exit__(self, *exc) -> None:
        """Release it."""
        self.close()
