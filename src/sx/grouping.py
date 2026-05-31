"""Grouping and filtering of sessions for the TUI tree.

Three grouping modes are offered, cycled with a key in the UI:

* ``PROJECT`` — by originating project path (the default);
* ``DATE`` — by a coarse recency bucket (Today, Yesterday, …);
* ``RECENCY`` — a single flat group, newest first.

These are pure functions over :class:`~sx.model.Session` so they can be unit
tested without a running app.
"""

from __future__ import annotations

import enum
from datetime import date, datetime

from sx.model import Session


class GroupMode(enum.Enum):
    """How the session tree is grouped.

    Attributes:
        PROJECT: Group by project path.
        DATE: Group by a coarse recency bucket.
        RECENCY: One flat group, newest first.
    """

    PROJECT = "project"
    DATE = "date"
    RECENCY = "recency"

    def label(self) -> str:
        """Return a short human label for this mode."""
        return {"project": "Project", "date": "Date", "recency": "Recency"}[self.value]

    def next(self) -> GroupMode:
        """Return the next mode in the cycle (PROJECT → DATE → RECENCY → …)."""
        order = [GroupMode.PROJECT, GroupMode.DATE, GroupMode.RECENCY]
        return order[(order.index(self) + 1) % len(order)]


def _recency_key(session: Session) -> float:
    """Sort key: most-recently-modified first (missing dates sort last)."""
    return session.modified.timestamp() if session.modified else 0.0


def _date_bucket(when: datetime | None, *, today: date) -> str:
    """Return a coarse recency bucket label for a timestamp.

    Args:
        when: The session timestamp (modified time).
        today: The reference "today" date (injected for testability).

    Returns:
        One of: ``Today``, ``Yesterday``, ``Earlier this week``,
        ``This month``, an ``YYYY-MM`` month label, or ``Unknown``.
    """
    if when is None:
        return "Unknown"
    d = when.date()
    delta = (today - d).days
    if delta <= 0:
        return "Today"
    if delta == 1:
        return "Yesterday"
    if delta < 7:
        return "Earlier this week"
    if d.year == today.year and d.month == today.month:
        return "This month"
    return f"{d.year:04d}-{d.month:02d}"


#: Sort priority for the named date buckets (lower sorts first). Month labels
#: (``YYYY-MM``) fall back to reverse-chronological after these.
_DATE_ORDER = {
    "Today": 0,
    "Yesterday": 1,
    "Earlier this week": 2,
    "This month": 3,
}


def filter_sessions(sessions: list[Session], query: str) -> list[Session]:
    """Filter sessions by a case-insensitive substring of title or project.

    Args:
        sessions: Sessions to filter.
        query: Search text; empty/whitespace returns all sessions.

    Returns:
        The matching sessions (input order preserved).
    """
    q = query.strip().lower()
    if not q:
        return list(sessions)
    out = []
    for s in sessions:
        haystack = f"{s.title}\n{s.project_path or ''}\n{s.session_id}".lower()
        if q in haystack:
            out.append(s)
    return out


def group_sessions(
    sessions: list[Session],
    mode: GroupMode,
    *,
    today: date | None = None,
) -> list[tuple[str, list[Session]]]:
    """Group and order sessions for display.

    Args:
        sessions: Sessions to group (already filtered to one harness).
        mode: The grouping mode.
        today: Reference date for ``DATE`` bucketing (defaults to today).

    Returns:
        An ordered list of ``(group_label, sessions)`` pairs. Within each group
        sessions are newest-first.
    """
    if mode is GroupMode.RECENCY:
        ordered = sorted(sessions, key=_recency_key, reverse=True)
        return [("All sessions, newest first", ordered)] if ordered else []

    groups: dict[str, list[Session]] = {}
    if mode is GroupMode.PROJECT:
        for s in sessions:
            groups.setdefault(s.project_path or "(unknown project)", []).append(s)
        ordered_keys = sorted(groups)
    else:  # DATE
        ref = today or datetime.now().date()
        for s in sessions:
            groups.setdefault(_date_bucket(s.modified, today=ref), []).append(s)
        ordered_keys = sorted(
            groups,
            key=lambda k: (_DATE_ORDER.get(k, 4), _month_sort(k)),
        )

    result = []
    for key in ordered_keys:
        result.append((key, sorted(groups[key], key=_recency_key, reverse=True)))
    return result


def _month_sort(label: str) -> str:
    """Secondary sort for date groups so ``YYYY-MM`` months sort newest-first.

    Named buckets share the same primary order and return ``""`` here; month
    labels are inverted by digit complement so a later month sorts earlier.

    Args:
        label: A group label.

    Returns:
        A comparable string.
    """
    if label and label[0].isdigit():
        # Invert each digit so larger (more recent) months sort first.
        return "".join(str(9 - int(c)) if c.isdigit() else c for c in label)
    return ""
