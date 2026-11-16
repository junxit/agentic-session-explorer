"""Times read from inside session files display in local time, like file times.

Harnesses record their timestamps in UTC (``…Z``), while ``sx`` reads file
modification times in local time. Until 0.8.1 the two were shown side by side
on different clocks: in UTC-6, an export of a session started at 08:12 said
"Created 14:12" and "Modified 08:18". Every test here pins the process's time
zone, so it means the same on a machine, or a CI runner, set to UTC.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from sx.adapters import pi as pi_module
from sx.adapters.pi import PiAdapter, encode_session_folder
from sx.export import messages_to_markdown
from sx.model import BranchMode
from sx.util import parse_ts
from tests.conftest import write_compact_jsonl

#: A zone that is never UTC: UTC-6 in November, as Central Standard Time.
ZONE = "America/Chicago"


@pytest.fixture
def central_time():
    """Run the test with the process's local time zone set to :data:`ZONE`."""
    saved = os.environ.get("TZ")
    os.environ["TZ"] = ZONE
    time.tzset()
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = saved
        time.tzset()


def test_a_utc_timestamp_reads_as_local_time(central_time):
    """``…Z`` becomes the local wall time of the same instant, and stays aware."""
    parsed = parse_ts("2026-11-16T14:12:45.683Z")

    assert parsed == datetime(2026, 11, 16, 14, 12, 45, 683000, tzinfo=UTC)
    assert parsed.utcoffset() == timedelta(hours=-6)
    assert parsed.strftime("%Y-%m-%d %H:%M") == "2026-11-16 08:12"


def test_any_recorded_offset_reads_as_local_time(central_time):
    """An explicit offset is converted too, not only ``Z``."""
    parsed = parse_ts("2026-11-16T15:12:45+01:00")

    assert parsed == datetime(2026, 11, 16, 14, 12, 45, tzinfo=UTC)
    assert parsed.strftime("%Y-%m-%d %H:%M") == "2026-11-16 08:12"


def test_a_timestamp_without_an_offset_is_kept_as_written(central_time):
    """With no offset there is no instant to convert, so nothing moves."""
    parsed = parse_ts("2026-11-16T14:12:45")

    assert parsed == datetime(2026, 11, 16, 14, 12, 45)
    assert parsed.tzinfo is None


def test_an_instant_local_time_cannot_hold_is_kept_as_recorded(central_time):
    """A timestamp at the edge of the calendar never breaks parsing.

    In UTC-6 the first instant of year 1 falls in year 0, which ``datetime``
    cannot hold, so it is kept on its recorded clock instead.
    """
    parsed = parse_ts("0001-01-01T00:00:00Z")

    assert parsed == datetime(1, 1, 1, tzinfo=UTC)
    assert parsed.utcoffset() == timedelta(0)


def test_an_export_shows_every_time_on_the_local_clock(central_time, tmp_path: Path, monkeypatch):
    """Created and message times in an export are on the same clock as Modified.

    The session starts at 03:30 UTC, which in UTC-6 is the evening before, so
    the date moves as well as the hour.
    """
    monkeypatch.setattr(pi_module, "home", lambda: tmp_path)
    monkeypatch.setitem(sys.modules, "sx.adapters.omp", None)
    stamp = "2026-11-16T03:30:00.000Z"
    folder = tmp_path / ".pi" / "agent" / "sessions" / encode_session_folder("/old/proj")
    message = {"role": "user", "content": "hello", "timestamp": 1}
    write_compact_jsonl(
        folder / "2026-11-16T03-30-00-000Z_S1.jsonl",
        [
            {"type": "session", "version": 3, "id": "S1", "timestamp": stamp, "cwd": "/old/proj"},
            {
                "type": "message",
                "id": "u1",
                "parentId": None,
                "timestamp": stamp,
                "message": message,
            },
        ],
    )
    adapter = PiAdapter()
    (session,) = adapter.discover()

    markdown = messages_to_markdown(
        session, adapter.load_transcript(session).messages(BranchMode.CURRENT)
    )

    assert "- **Created:** 2026-11-15 21:30\n" in markdown
    assert " · 2026-11-15 21:30\n" in markdown
    assert "03:30" not in markdown
