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
import time
from datetime import datetime
from pathlib import Path

from sx.model import DeleteResult, Orphan, Session

#: A session modified within this many seconds is considered "live".
ACTIVE_WINDOW_SECONDS = 90


def default_log_path() -> Path:
    """Return the default op-log path (``./sx-deletions.log``)."""
    return Path.cwd() / "sx-deletions.log"


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

    # --- active-session detection ---------------------------------------

    def is_active(self, session: Session, *, window: int = ACTIVE_WINDOW_SECONDS) -> bool:
        """Return True if the session's file changed within ``window`` seconds.

        The primary file is re-``stat``-ed at call time (not the cached mtime
        from discovery) so the check reflects the moment of deletion.

        Args:
            session: The session to test.
            window: Recency window in seconds.

        Returns:
            True if the session appears to be in active use.
        """
        path = session.primary_path
        if path is None:
            return False
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return False
        return (time.time() - mtime) <= window

    # --- sessions --------------------------------------------------------

    def preview(self, session: Session) -> DeleteResult:
        """Return a dry-run :class:`DeleteResult` for a session.

        Args:
            session: The session to preview deleting.

        Returns:
            A dry-run result listing every path that would be removed.
        """
        adapter = self._adapters[session.harness]
        return adapter.delete(session, dry_run=True)

    def delete(self, session: Session) -> DeleteResult:
        """Permanently delete a session and record it in the op-log.

        Args:
            session: The session to delete.

        Returns:
            The :class:`DeleteResult` describing what was removed.
        """
        adapter = self._adapters[session.harness]
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
        adapter = self._adapters[orphan.harness]
        return adapter.delete_orphan(orphan, dry_run=True)

    def delete_orphan(self, orphan: Orphan) -> DeleteResult:
        """Permanently delete an orphan and record it in the op-log."""
        adapter = self._adapters[orphan.harness]
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
        try:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError:
            pass
