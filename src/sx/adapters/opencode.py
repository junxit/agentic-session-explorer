"""Adapter for opencode chat sessions (SQLite-backed).

Unlike the JSONL harnesses (one file per session), opencode stores **every**
session as rows in one shared SQLite database at
``~/.local/share/opencode/opencode.db`` (WAL journal mode). The transcript lives
across three tables:

* ``session`` — one row per conversation (``directory`` = project cwd, ``title``,
  millisecond ``time_created``/``time_updated``; ``parent_id`` set for internal
  sub-agent sessions);
* ``message`` — one row per turn, carrying the ``role`` in its JSON ``data``;
* ``part`` — the actual content units (``text``, ``reasoning``, …) in JSON ``data``.

A tiny per-session sidecar ``storage/session_diff/<id>.json`` accompanies each
session on disk.

Because sessions are rows, **deletion must remove rows, never the ``.db`` file**
(that would destroy every session). This adapter therefore overrides
:meth:`delete` to run a scoped, transactional cascade
(``DELETE … WHERE session_id = ?``) plus unlink the sidecar, and never touches
``opencode.db`` itself, the shared ``log/`` files, or any other session.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from ..model import (
    Capability,
    DeleteResult,
    Message,
    Orphan,
    OrphanKind,
    Role,
    Session,
)
from ..util import home, is_within
from .base import HarnessAdapter

# Tables (besides ``session`` itself) that carry a ``session_id`` column and must
# be cleared when a session is deleted. Children first; ``session`` is removed
# last and separately (it is keyed by ``id``, not ``session_id``). This is a
# fixed constant — never user input — so interpolating it into SQL is safe.
_CASCADE_TABLES = ("part", "message", "todo", "session_share", "session_message")


def _truncate(text: str, limit: int) -> str:
    """Shorten ``text`` to ``limit`` characters, appending an ellipsis."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _ms_to_dt(value: object) -> datetime | None:
    """Convert a millisecond epoch timestamp to a local datetime, tolerantly.

    opencode stores timestamps as integer milliseconds since the epoch (e.g.
    ``1780273855373``), so the ISO-only :func:`sx.util.parse_ts` does not apply.

    Args:
        value: A millisecond epoch value (int or float).

    Returns:
        A ``datetime``, or ``None`` if the value is missing or unparseable.
    """
    if not isinstance(value, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(value / 1000.0)
    except (OverflowError, OSError, ValueError):
        return None


#: opencode session ids look like ``ses_17f68c472ffeYwIg8S4I4Nz4Z8``. Anything
#: outside this alphabet is rejected before it reaches a filesystem path.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def _valid_session_id(session_id: object) -> bool:
    """Return True if ``session_id`` is safe to interpolate into a path.

    The id originates in the database, which any process with filesystem access
    could write to, so it is treated as untrusted input rather than assumed
    well-formed.
    """
    return isinstance(session_id, str) and bool(_SESSION_ID_RE.match(session_id))


def _loads(data: object) -> dict:
    """Parse a JSON string into a dict, tolerantly returning ``{}`` on failure."""
    if isinstance(data, str) and data:
        try:
            obj = json.loads(data)
        except (json.JSONDecodeError, ValueError):
            return {}
        if isinstance(obj, dict):
            return obj
    elif isinstance(data, dict):
        return data
    return {}


class OpencodeAdapter(HarnessAdapter):
    """Adapter for opencode sessions stored in a shared SQLite database.

    Attributes:
        name: Machine identifier (``"opencode"``).
        display: Human-readable harness name.
        capabilities: Browse, orphan detection, and delete.
    """

    name = "opencode"
    display = "opencode"
    capabilities = Capability.BROWSE | Capability.ORPHANS | Capability.DELETE

    # --- paths -----------------------------------------------------------

    def _store(self) -> Path:
        """Return the single store root (``~/.local/share/opencode``).

        Resolved from :func:`sx.util.home` at call time so tests can redirect it.
        """
        return home() / ".local" / "share" / "opencode"

    def _db_path(self) -> Path:
        """Return the path to ``opencode.db``."""
        return self._store() / "opencode.db"

    def _sidecar_path(self, session_id: str) -> Path:
        """Return the per-session sidecar ``storage/session_diff/<id>.json``.

        The id comes from a database column, so it is interpolated into a path
        only after :func:`_valid_session_id` has vetted it; otherwise a crafted
        id such as ``../../x`` would escape the sidecar directory.
        """
        return self._store() / "storage" / "session_diff" / f"{session_id}.json"

    def store_roots(self) -> list[Path]:
        """Return the one managed directory.

        Note that ``opencode.db`` lives *inside* this root, so the delete guard
        (:func:`sx.util.is_within`) does **not** by itself protect the database
        file. The database is protected because :meth:`delete` never passes its
        path to :meth:`_delete_paths` — only the sidecar file is ever unlinked.

        Returns:
            ``[~/.local/share/opencode]``.
        """
        return [self._store()]

    def available(self) -> bool:
        """Return True only if the opencode database file exists.

        Stricter than the default (any store root exists): an empty data dir with
        no database is not a usable opencode install.
        """
        return self._db_path().exists()

    # --- connection ------------------------------------------------------

    def _connect_ro(self) -> sqlite3.Connection | None:
        """Open a read-only connection to the database, or ``None`` on failure.

        Read-only ``mode=ro`` URI access works even while opencode holds the WAL
        open. Any error (missing, locked, corrupt) yields ``None`` so the harness
        degrades gracefully rather than crashing the whole tool.
        """
        db = self._db_path()
        if not db.exists():
            return None
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2.0)
            con.row_factory = sqlite3.Row
            return con
        except (sqlite3.Error, OSError):
            return None

    # --- discovery -------------------------------------------------------

    def discover(self) -> Iterator[Session]:
        """Yield one :class:`Session` per top-level conversation.

        Internal sub-agent sessions (those with a non-null ``parent_id``) are
        skipped. The full transcript is left for :meth:`load`.

        Yields:
            A :class:`Session` for each top-level opencode session.
        """
        con = self._connect_ro()
        if con is None:
            return
        try:
            rows = con.execute(
                "SELECT id, parent_id, directory, title, time_created, time_updated "
                "FROM session ORDER BY time_updated DESC"
            ).fetchall()
        except sqlite3.Error:
            con.close()
            return

        for row in rows:
            try:
                if row["parent_id"] is not None:
                    continue  # skip sub-agent sessions
                if not _valid_session_id(row["id"]):
                    continue  # malformed id — never build a path from it
                session = self._session_from_row(row)
            except (sqlite3.Error, KeyError, IndexError):
                continue
            yield session
        con.close()

    def _session_from_row(self, row: sqlite3.Row) -> Session:
        """Build a metadata-only :class:`Session` from a ``session`` row."""
        sid = row["id"]
        directory = row["directory"]
        sidecar = self._sidecar_path(sid)

        created = _ms_to_dt(row["time_created"])
        modified = _ms_to_dt(row["time_updated"])
        # Prefer the later of the DB's time_updated and the sidecar mtime so the
        # listed recency is accurate even though the live-guard keys off the
        # sidecar's mtime specifically.
        try:
            if sidecar.exists():
                sc_mtime = datetime.fromtimestamp(sidecar.stat().st_mtime)
                if modified is None or sc_mtime > modified:
                    modified = sc_mtime
        except OSError:
            pass

        size = 0
        try:
            if sidecar.exists():
                size = sidecar.stat().st_size
        except OSError:
            pass

        return Session(
            harness=self.name,
            session_id=sid,
            project_path=directory or None,
            title=_truncate(row["title"] or sid, 80),
            created=created,
            modified=modified,
            message_count=None,
            size_bytes=size,
            paths=[sidecar],
            is_orphan=bool(directory) and not Path(directory).exists(),
        )

    # --- loading ---------------------------------------------------------

    def load(self, session: Session) -> list[Message]:
        """Reconstruct a session's transcript from the ``message``/``part`` join.

        Parts are ordered by creation time (with the part id as a stable
        tiebreaker). The role of each turn comes from the part's owning message.

        Args:
            session: The session to load.

        Returns:
            The normalized transcript; empty if the database is unreadable.
        """
        con = self._connect_ro()
        if con is None:
            return []
        try:
            rows = con.execute(
                "SELECT p.data AS part_data, m.data AS msg_data, "
                "p.time_created AS t "
                "FROM part p JOIN message m ON m.id = p.message_id "
                "WHERE p.session_id = ? "
                "ORDER BY p.time_created ASC, p.id ASC",
                (session.session_id,),
            ).fetchall()
        except sqlite3.Error:
            con.close()
            return []
        con.close()

        messages: list[Message] = []
        for row in rows:
            msg = self._part_to_message(row)
            if msg is not None:
                messages.append(msg)
        return messages

    def _part_to_message(self, row: sqlite3.Row) -> Message | None:
        """Map one joined ``part``+``message`` row to a :class:`Message`.

        Args:
            row: A row with ``part_data``, ``msg_data``, and ``t`` columns.

        Returns:
            A :class:`Message`, or ``None`` for parts that carry no transcript
            content (step markers, snapshots, file refs, unknown types).
        """
        part = _loads(row["part_data"])
        msg = _loads(row["msg_data"])
        ptype = part.get("type")
        role_raw = msg.get("role")
        ts = _ms_to_dt(part.get("time")) or _ms_to_dt(row["t"])

        if ptype == "text":
            text = (part.get("text") or "").strip()
            if not text:
                return None
            role = Role.USER if role_raw == "user" else Role.ASSISTANT
            return Message(role, text=text, timestamp=ts)

        if ptype == "reasoning":
            thinking = (part.get("text") or "").strip()
            if not thinking:
                return None
            return Message(Role.ASSISTANT, thinking=thinking, timestamp=ts)

        if ptype == "tool":
            # Defensive: this shape was not present in the sampled DB, so read it
            # cautiously. Surface the tool name and a short preview of its output.
            name = part.get("tool") or part.get("name") or "tool"
            state = part.get("state") if isinstance(part.get("state"), dict) else {}
            preview = state.get("output") or state.get("title") or part.get("text") or ""
            return Message(
                Role.TOOL,
                text=_truncate(str(preview), 200),
                tool_summary=str(name),
                timestamp=ts,
            )

        # step-start, step-finish, snapshot, file, and any unknown type → skip.
        return None

    # --- orphans ---------------------------------------------------------

    def find_orphans(self) -> list[Orphan]:
        """Find sidecar files with no matching session row.

        Dead-project sessions (whose ``directory`` no longer exists) are NOT
        returned here; they are flagged via :attr:`Session.is_orphan` during
        discovery so their cleanup routes through the row-cascading :meth:`delete`
        rather than the file-only :meth:`delete_orphan`.

        Returns:
            ``STRAY_TEMP`` orphans for stale ``session_diff`` sidecars.
        """
        diff_dir = self._store() / "storage" / "session_diff"
        if not diff_dir.is_dir():
            return []

        live_ids: set[str] = set()
        con = self._connect_ro()
        if con is not None:
            try:
                live_ids = {r["id"] for r in con.execute("SELECT id FROM session")}
            except sqlite3.Error:
                live_ids = set()
            finally:
                con.close()
        else:
            # Cannot confirm which sidecars are stale without the DB; be safe.
            return []

        orphans: list[Orphan] = []
        for json_file in sorted(diff_dir.glob("*.json")):
            if json_file.stem in live_ids:
                continue
            size = 0
            try:
                size = json_file.stat().st_size
            except OSError:
                pass
            orphans.append(
                Orphan(
                    harness=self.name,
                    kind=OrphanKind.STRAY_TEMP,
                    paths=[json_file],
                    reason="session_diff sidecar with no matching session",
                    size_bytes=size,
                )
            )
        return orphans

    # --- delete (overridden: rows, not the db file) ----------------------

    def delete(self, session: Session, *, dry_run: bool = False) -> DeleteResult:
        """Permanently delete a session: its DB rows plus the sidecar file.

        The deletion is scoped strictly to this session id, so no other session
        is affected. The ``opencode.db`` file itself is never removed, never
        VACUUMed, and the shared ``log/`` files are left untouched.

        Args:
            session: The session to delete.
            dry_run: If True, report what would be removed without changing anything.

        Returns:
            A :class:`DeleteResult`. ``removed`` holds the unlinked sidecar (if
            any); ``note`` records the database row count.
        """
        sid = session.session_id
        if not _valid_session_id(sid):
            return DeleteResult(
                dry_run=dry_run,
                note=f"refused: malformed session id {sid!r}",
            )
        sidecar = self._sidecar_path(sid)

        if dry_run:
            n = self._count_rows(sid)
            result = DeleteResult(dry_run=True)
            if sidecar.exists():
                result.removed.append(sidecar)
                try:
                    result.freed_bytes += sidecar.stat().st_size
                except OSError:
                    pass
            result.note = f"would delete {n} db row(s)"
            return result

        # The database transaction runs FIRST and the sidecar is unlinked only
        # after it commits. The reverse order left three failure paths (missing
        # db, guard refusal, sqlite error) in which the sidecar was already gone
        # while every row survived — an inconsistent state nothing could detect,
        # since orphan scanning only looks for sidecars without sessions.
        result = DeleteResult()

        db = self._db_path()
        if not db.exists():
            result.note = "database not found; nothing deleted"
            return result
        if not is_within(db, self.store_roots()):
            result.skipped[db] = "database outside store root (refused)"
            return result

        total = 0
        con: sqlite3.Connection | None = None
        try:
            con = sqlite3.connect(str(db), timeout=5.0)
            con.execute("PRAGMA busy_timeout=2000")
            con.execute("BEGIN")
            for table in _CASCADE_TABLES:
                cur = con.execute(
                    f"DELETE FROM {table} WHERE session_id = ?", (sid,)
                )
                total += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
            cur = con.execute("DELETE FROM session WHERE id = ?", (sid,))
            total += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
            con.commit()
        except sqlite3.Error as exc:
            if con is not None:
                try:
                    con.rollback()
                except sqlite3.Error:
                    pass
            result.skipped[db] = f"db delete failed: {exc}"
            return result
        finally:
            if con is not None:
                con.close()

        # Rows are gone; now remove the sidecar through the guarded remover
        # (it lives inside the store root, so the allowlist permits it).
        sidecar_result = self._delete_paths([sidecar], dry_run=False)
        result.removed.extend(sidecar_result.removed)
        result.freed_bytes += sidecar_result.freed_bytes
        for path, reason in sidecar_result.refused.items():
            result.skipped[path] = reason

        result.note = f"deleted {total} db row(s) across {len(_CASCADE_TABLES) + 1} tables"
        return result

    def last_activity(self, session: Session) -> datetime | None:
        """Return the session's last-write time from the database.

        Overrides the file-mtime default: opencode's per-session sidecar is
        optional and lags the conversation, so using it as the liveness signal
        meant the live-delete guard almost never engaged. ``session.time_updated``
        is what actually tracks activity.

        Args:
            session: The session to probe.

        Returns:
            The last-update time, or ``None`` if it cannot be read.
        """
        con = self._connect_ro()
        if con is None:
            return None
        try:
            row = con.execute(
                "SELECT time_updated FROM session WHERE id = ?", (session.session_id,)
            ).fetchone()
        except sqlite3.Error:
            return None
        finally:
            con.close()
        return _ms_to_dt(row["time_updated"]) if row else None

    def _count_rows(self, session_id: str) -> int:
        """Count the rows a delete would remove for ``session_id`` (best effort)."""
        con = self._connect_ro()
        if con is None:
            return 0
        total = 0
        try:
            for table in _CASCADE_TABLES:
                row = con.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                total += row[0] if row else 0
            row = con.execute(
                "SELECT COUNT(*) FROM session WHERE id = ?", (session_id,)
            ).fetchone()
            total += row[0] if row else 0
        except sqlite3.Error:
            pass
        finally:
            con.close()
        return total
