"""Tests for sx.export: Markdown rendering, filename, and file writing.

Ported from the export portion of /tmp/verify_m4_core.py. A tiny fake adapter
provides messages; nothing touches a real harness store.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sx.export import export_filename, export_session, messages_to_markdown
from sx.model import Capability, Message, Role, Session
from sx.adapters.base import HarnessAdapter


class _FakeAdapter(HarnessAdapter):
    """Minimal adapter whose load() returns two fixed messages."""

    name = "tmp"
    display = "Tmp"
    capabilities = Capability.BROWSE | Capability.DELETE

    def store_roots(self) -> list[Path]:
        """No real roots needed for export tests."""
        return []

    def discover(self):
        """Discovery is unused here."""
        return iter(())

    def load(self, session: Session) -> list[Message]:
        """Return a fixed two-message transcript."""
        return [
            Message(Role.USER, text="hello world", timestamp=datetime(2026, 5, 30, 12, 0)),
            Message(
                Role.ASSISTANT,
                text="hi",
                thinking="ponder",
                tool_summary="Bash",
                timestamp=datetime(2026, 5, 30, 12, 1),
            ),
        ]


def _session() -> Session:
    """A session with a deterministic id/title/modified for export tests."""
    return Session(
        harness="tmp",
        session_id="abcd1234",
        project_path="/some/proj",
        title="My Test Session!",
        modified=datetime(2026, 5, 30, 12, 0),
    )


def test_messages_to_markdown_structure():
    """The Markdown has the title heading, harness meta, and all sections."""
    session = _session()
    md = messages_to_markdown(
        session, _FakeAdapter().load(session), now=datetime(2026, 5, 30, 13, 0)
    )
    assert md.startswith("# My Test Session!")
    assert "**Harness:** tmp" in md
    assert "user" in md  # user section heading
    assert "assistant" in md  # assistant section heading
    assert "> 💭" in md  # thinking blockquote
    assert "Bash" in md  # tool summary


def test_messages_to_markdown_empty():
    """An empty transcript renders the no-messages placeholder."""
    md = messages_to_markdown(_session(), [], now=datetime(2026, 5, 30, 13, 0))
    assert "_(no messages)_" in md


def test_export_filename_slug_format():
    """The filename follows <harness>-<YYYYMMDD>-<8id>-<slug>.md."""
    assert export_filename(_session()) == "tmp-20260530-abcd1234-my-test-session.md"


def test_export_session_writes_file(tmp_path: Path):
    """export_session writes a non-empty file into dest_dir and returns it."""
    session = _session()
    out = export_session(
        _FakeAdapter(), session, dest_dir=tmp_path / "exports", now=datetime(2026, 5, 30, 13, 0)
    )
    assert out.exists()
    assert out.parent == tmp_path / "exports"
    assert out.name == "tmp-20260530-abcd1234-my-test-session.md"
    assert out.read_text(encoding="utf-8").strip() != ""
