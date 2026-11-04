"""Tests for sx.export: Markdown rendering, filename, and file writing.

Ported from the export portion of /tmp/verify_m4_core.py. A tiny fake adapter
provides messages; nothing touches a real harness store. A session whose file
was deleted is refused rather than archived as an empty document.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from sx.adapters.base import HarnessAdapter
from sx.export import (
    SessionGoneError,
    export_filename,
    export_session,
    messages_to_markdown,
    session_gone,
)
from sx.model import Capability, Message, Role, Session


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


@pytest.mark.parametrize(
    ("session_id", "short"),
    [
        ("be0c3352-1f2e-4d5c-8b9a-0123456789ab", "be0c3352"),  # a UUID, as before
        ("ses_3b2f9a1c7d", "ses_3b2f"),  # OpenCode keeps its underscore
        ("ab/../cd-x", "ab-cd-x"),  # a path separator in a Pi header id
        ("[red]zzzzz", "red-zzzz"),
        ("\x1b]0;x\x07", "0-x"),
        ("../..", "session"),
        ("", "session"),
    ],
)
def test_export_filename_slugifies_the_session_id(session_id, short):
    """The id is read from the session file, so it is made safe for a filename (T8)."""
    session = _session()
    session.session_id = session_id
    assert export_filename(session) == f"tmp-20260530-{short}-my-test-session.md"


def test_an_id_with_a_slash_still_exports_into_dest_dir(tmp_path: Path):
    """A ``/`` in the id used to send the file into a directory that does not exist (T8)."""
    session = _session()
    session.session_id = "ab/../cd"
    out = export_session(_FakeAdapter(), session, dest_dir=tmp_path / "exports")
    assert out.parent == tmp_path / "exports"
    assert out.name == "tmp-20260530-ab-cd-my-test-session.md"


# --- a session whose file is gone -----------------------------------------------


class _ReadsTheFile(_FakeAdapter):
    """Reads nothing for a session whose file is gone, as file-backed adapters do."""

    def load(self, session: Session) -> list[Message]:
        """Return the messages while the session's file exists, else none."""
        if session.primary_path is None or not session.primary_path.exists():
            return []
        return super().load(session)


class _Database(_FakeAdapter):
    """Keeps sessions in a database; the file a session lists is an optional sidecar."""

    def last_activity(self, session: Session) -> datetime | None:
        """The session's row is there, whatever became of its sidecar."""
        return datetime(2026, 5, 30, 12, 0)


def _with_file(tmp_path: Path, *, exists: bool) -> Session:
    """Return the export session, listing a file under ``tmp_path`` that may exist."""
    session = _session()
    session.paths = [tmp_path / "s.jsonl"]
    if exists:
        session.paths[0].write_text("{}\n", encoding="utf-8")
    return session


def test_a_session_whose_file_is_gone_is_not_exported(tmp_path: Path):
    """No empty archive is written for a session deleted elsewhere, nor its directory."""
    session = _with_file(tmp_path, exists=False)
    with pytest.raises(SessionGoneError, match=r"the session's file no longer exists: .*s\.jsonl"):
        export_session(_ReadsTheFile(), session, dest_dir=tmp_path / "exports")
    assert not (tmp_path / "exports").exists()


def test_a_transcript_read_before_the_file_went_is_still_archived(tmp_path: Path):
    """What was read is a real copy of the session, so it is written."""
    session = _with_file(tmp_path, exists=False)
    transcript = _FakeAdapter().load_transcript(session)
    out = export_session(_ReadsTheFile(), session, tmp_path / "exports", transcript=transcript)
    assert "hello world" in out.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("adapter", "exists", "gone"),
    [
        (_ReadsTheFile(), True, False),
        (_ReadsTheFile(), False, True),  # no file, and no activity either
        (_Database(), False, False),  # the database still has the session
        (object(), False, True),  # an adapter that cannot be asked: the file decides
    ],
    ids=["file-there", "file-gone", "database", "no-probe"],
)
def test_session_gone_needs_the_file_gone_and_no_activity(tmp_path: Path, adapter, exists, gone):
    """The file alone does not settle it: a database harness's sidecar is optional."""
    assert session_gone(adapter, _with_file(tmp_path, exists=exists)) is gone


def test_a_session_without_files_or_a_working_probe_is_not_gone(tmp_path: Path):
    """Nothing to check, or a probe that fails, proves nothing: the session counts as there."""

    class _Broken(_FakeAdapter):
        """An adapter whose liveness probe raises."""

        def last_activity(self, session: Session) -> datetime | None:
            """Fail the way a bug in an adapter would."""
            raise RuntimeError("boom")

    assert session_gone(_ReadsTheFile(), _session()) is False
    assert session_gone(_Broken(), _with_file(tmp_path, exists=False)) is False
