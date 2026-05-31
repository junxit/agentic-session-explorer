"""Tests for sx.grouping: GroupMode cycle, grouping modes, filtering.

Ported from /tmp/verify_grouping.py. Session objects are built directly; no
files are needed.
"""

from __future__ import annotations

from datetime import date, datetime

from sx.grouping import GroupMode, filter_sessions, group_sessions
from sx.model import Session

TODAY = date(2026, 5, 30)


def _session(sid: str, proj: str | None, mod: datetime | None, title: str = "t") -> Session:
    """Build a metadata-only Session for grouping tests."""
    return Session(
        harness="h",
        session_id=sid,
        project_path=proj,
        title=title,
        modified=mod,
    )


def _sample_sessions() -> list[Session]:
    """A spread of sessions covering every date bucket and an unknown one."""
    return [
        _session("a", "/proj/x", datetime(2026, 5, 30, 9)),  # today
        _session("b", "/proj/x", datetime(2026, 5, 29, 9)),  # yesterday
        _session("c", "/proj/y", datetime(2026, 5, 26, 9)),  # earlier this week
        _session("d", "/proj/y", datetime(2026, 5, 5, 9)),   # this month
        _session("e", "/proj/z", datetime(2026, 1, 15, 9)),  # 2026-01
        _session("f", "/proj/z", None),                       # unknown
    ]


def test_group_mode_next_cycle():
    """next() cycles PROJECT -> DATE -> RECENCY -> PROJECT."""
    assert GroupMode.PROJECT.next() is GroupMode.DATE
    assert GroupMode.DATE.next() is GroupMode.RECENCY
    assert GroupMode.RECENCY.next() is GroupMode.PROJECT


def test_group_project_mode():
    """PROJECT groups by project_path, alpha-sorted, newest-first within group."""
    g = group_sessions(_sample_sessions(), GroupMode.PROJECT, today=TODAY)
    keys = [k for k, _ in g]
    assert len(g) == 3
    assert keys == ["/proj/x", "/proj/y", "/proj/z"]
    # First group (/proj/x) has a, b newest-first.
    assert [s.session_id for s in g[0][1]] == ["a", "b"]


def test_group_date_mode_bucket_order():
    """DATE buckets appear in priority order with Unknown sorting last."""
    g = group_sessions(_sample_sessions(), GroupMode.DATE, today=TODAY)
    keys = [k for k, _ in g]
    assert keys[0] == "Today"
    assert keys[1] == "Yesterday"
    assert keys[2] == "Earlier this week"
    assert keys[3] == "This month"
    assert "2026-01" in keys
    assert "Unknown" in keys
    assert keys.index("Unknown") > keys.index("This month")


def test_group_recency_mode():
    """RECENCY is a single flat group, newest-first, None-modified last."""
    g = group_sessions(_sample_sessions(), GroupMode.RECENCY, today=TODAY)
    assert len(g) == 1
    ids = [s.session_id for s in g[0][1]]
    assert ids == ["a", "b", "c", "d", "e", "f"]


def test_filter_empty_returns_all():
    """An empty query returns every session."""
    sessions = _sample_sessions()
    assert len(filter_sessions(sessions, "")) == len(sessions)
    assert len(filter_sessions(sessions, "   ")) == len(sessions)


def test_filter_by_project_substring():
    """A project substring narrows to matching sessions."""
    assert len(filter_sessions(_sample_sessions(), "proj/x")) == 2


def test_filter_by_title_and_case_insensitive():
    """Title substrings match, case-insensitively."""
    sessions = [
        _session("k", "/p", datetime(2026, 5, 1), title="Fix the bug"),
        _session("m", "/p", datetime(2026, 5, 1), title="Add feature"),
    ]
    assert [s.session_id for s in filter_sessions(sessions, "bug")] == ["k"]
    assert len(filter_sessions(sessions, "FIX")) == 1
