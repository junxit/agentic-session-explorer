"""Deletion orchestration: previews, the active-session check, and the op-log.

The hard safety guard (refusing to touch anything outside a harness's store
roots) lives in :meth:`sx.adapters.base.HarnessAdapter.delete`. This service
layers the cross-cutting concerns on top:

* a **dry-run preview** so the UI can show exactly what will be removed;
* **active-session detection** — a session whose file changed within the last
  :data:`ACTIVE_WINDOW_SECONDS` is flagged as live, so the UI can warn and
  require an extra confirmation before deleting it; and
* an **append-only op-log** recording every deletion for accountability
  (deletion is permanent — there is no undo).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

from sx.model import DeleteResult, Orphan, Session

#: A session modified within this many seconds is considered "live".
ACTIVE_WINDOW_SECONDS = 90


def default_log_path() -> Path:
    """Return the op-log path: ``./sx-deletions.log``, or ``SX_LOG_FILE``.

    The log is written beside the working directory so it stays visible next to
    the work it describes. Note that it is therefore per-directory: running
    ``sx`` from several places produces several logs. Set ``SX_LOG_FILE`` to
    collect every deletion in one file instead.

    Returns:
        The op-log path.
    """
    override = os.environ.get("SX_LOG_FILE")
    if override:
        return Path(override).expanduser()
    return Path.cwd() / "sx-deletions.log"


def _no_adapter(harness: str, *, dry_run: bool) -> DeleteResult:
    """Return a refusing result for a harness with no registered adapter.

    Returned instead of raising ``KeyError``: these calls run inside Textual
    workers, where an unhandled exception tears down the whole app — possibly
    part-way through a batch of deletions.
    """
    return DeleteResult(
        dry_run=dry_run,
        skipped={Path(f"<{harness}>"): "no adapter registered (refused)"},
    )


class DeleteService:
    """Coordinates previews, deletions, the active check, and op-logging.

    Args:
        adapters_by_name: Mapping of harness name to adapter instance.
        log_path: Where to append the deletion op-log; defaults to
            :func:`default_log_path`.
    """

    def __init__(self, adapters_by_name: dict, log_path: Path | None = None) -> None:
        """Store adapters and resolve the op-log location."""
        self._adapters = adapters_by_name
        self._log_path = log_path or default_log_path()
        #: Set when the last op-log write failed, so the UI can surface it.
        self.last_log_error: str | None = None

    # --- active-session detection ---------------------------------------

    def is_active(self, session: Session, *, window: int = ACTIVE_WINDOW_SECONDS) -> bool:
        """Return True if the session was written within ``window`` seconds.

        Liveness is asked of the owning adapter
        (:meth:`~sx.adapters.base.HarnessAdapter.last_activity`) and recomputed at
        call time, so the answer reflects the moment of deletion. File-backed
        harnesses re-``stat`` their transcript; database-backed ones consult the
        row that actually tracks conversation activity.

        When liveness cannot be determined the session is treated as **active**.
        That is the fail-safe direction: the cost of a wrong "live" is one extra
        typed confirmation, while a wrong "not live" removes the only guard
        standing between a keystroke and a conversation being written right now.

        Args:
            session: The session to test.
            window: Recency window in seconds.

        Returns:
            True if the session appears to be in active use.
        """
        adapter = self._adapters.get(session.harness)
        if adapter is None:
            return True
        try:
            last = adapter.last_activity(session)
        except Exception:  # noqa: BLE001 - a broken probe must not disable the guard
            return True
        if last is None:
            return True
        return (time.time() - last.timestamp()) <= window

    # --- sessions --------------------------------------------------------

    def preview(self, session: Session) -> DeleteResult:
        """Return a dry-run :class:`DeleteResult` for a session.

        Args:
            session: The session to preview deleting.

        Returns:
            A dry-run result listing every path that would be removed.
        """
        adapter = self._adapters.get(session.harness)
        if adapter is None:
            return _no_adapter(session.harness, dry_run=True)
        return adapter.delete(session, dry_run=True)

    def delete(self, session: Session) -> DeleteResult:
        """Permanently delete a session and record it in the op-log.

        Args:
            session: The session to delete.

        Returns:
            The :class:`DeleteResult` describing what was removed.
        """
        adapter = self._adapters.get(session.harness)
        if adapter is None:
            return _no_adapter(session.harness, dry_run=False)
        result = adapter.delete(session, dry_run=False)
        self._log(
            action="delete_session",
            harness=session.harness,
            identifier=session.session_id,
            title=session.title,
            result=result,
        )
        return result

    # --- orphans ---------------------------------------------------------

    def preview_orphan(self, orphan: Orphan) -> DeleteResult:
        """Return a dry-run :class:`DeleteResult` for an orphan."""
        adapter = self._adapters.get(orphan.harness)
        if adapter is None:
            return _no_adapter(orphan.harness, dry_run=True)
        return adapter.delete_orphan(orphan, dry_run=True)

    def delete_orphan(self, orphan: Orphan) -> DeleteResult:
        """Permanently delete an orphan and record it in the op-log."""
        adapter = self._adapters.get(orphan.harness)
        if adapter is None:
            return _no_adapter(orphan.harness, dry_run=False)
        result = adapter.delete_orphan(orphan, dry_run=False)
        self._log(
            action="delete_orphan",
            harness=orphan.harness,
            identifier=orphan.kind.value,
            title=orphan.reason,
            result=result,
        )
        return result

    # --- op-log ----------------------------------------------------------

    def _log(
        self,
        *,
        action: str,
        harness: str,
        identifier: str,
        title: str,
        result: DeleteResult,
    ) -> None:
        """Append one JSON record describing a deletion to the op-log.

        Logging failures are swallowed: an audit-trail problem must never abort
        or mask the deletion result the caller is acting on.

        Args:
            action: ``"delete_session"`` or ``"delete_orphan"``.
            harness: Harness name.
            identifier: Session id or orphan kind.
            title: Human-readable label (title or reason).
            result: The deletion result to record.
        """
        if result.dry_run:
            return
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "action": action,
            "harness": harness,
            "id": identifier,
            "title": title,
            "removed": [str(p) for p in result.removed],
            "freed_bytes": result.freed_bytes,
            "skipped": {str(k): v for k, v in result.skipped.items()},
        }
        if result.note:
            entry["note"] = result.note
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            existed = self._log_path.exists()
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
            if not existed:
                # The log records chat-derived titles and absolute project
                # paths; keep it owner-only rather than world-readable.
                try:
                    self._log_path.chmod(0o600)
                except OSError:
                    pass
            self.last_log_error = None
        except OSError as exc:
            # Never abort the caller over an audit-trail problem, but do not
            # hide it either: a silent failure means deletions happen with no
            # record at all.
            self.last_log_error = f"could not write {self._log_path}: {exc}"
