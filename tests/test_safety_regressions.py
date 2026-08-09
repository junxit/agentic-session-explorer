"""Regression tests for the v0.4.0 safety pass.

Each test here pins a defect that shipped in an earlier version. The unifying
theme is that `sx` must never claim to have done something it did not do, and
must never destroy more than its confirmation disclosed.

Everything runs against synthetic data under ``tmp_path``; the user's real
harness stores are never read or written.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from sx.adapters.base import HarnessAdapter
from sx.model import Capability, DeleteResult, Message, Orphan, OrphanKind, Role, Session
from sx.service import DeleteService
from sx.util import is_within, mount_unavailable, sanitize_text


# --- helpers ---------------------------------------------------------------

class _RefusingAdapter(HarnessAdapter):
    """An adapter that reports orphans its own store roots do not permit.

    This is exactly the shape of the shipped Gemini bug: ``find_orphans`` emitted
    paths one level above the declared roots, so the delete guard refused all of
    them while the UI reported success.
    """

    name = "refusing"
    display = "Refusing"
    capabilities = Capability.BROWSE | Capability.ORPHANS | Capability.DELETE

    def __init__(self, root: Path, stray: Path) -> None:
        """Store the (narrow) root and the out-of-root stray file."""
        self._root = root
        self._stray = stray

    def store_roots(self) -> list[Path]:
        """Return a root that does NOT contain the reported orphan."""
        return [self._root]

    def discover(self):
        """No sessions are needed for these tests."""
        return iter(())

    def load(self, session: Session) -> list[Message]:
        """No transcript is needed for these tests."""
        return []

    def find_orphans(self) -> list[Orphan]:
        """Report the out-of-root stray file as an orphan."""
        return [
            Orphan(
                harness=self.name,
                kind=OrphanKind.STRAY_TEMP,
                paths=[self._stray],
                reason="stray file outside the declared root",
                size_bytes=1,
            )
        ]


# --- the guard must actually be reachable ----------------------------------

def test_orphan_paths_must_be_inside_declared_store_roots():
    """Every real adapter's orphans must be deletable by its own guard.

    The shipped Gemini adapter violated this: 24 orphans, 24 refusals. This
    invariant makes that class of bug impossible to reintroduce silently.
    """
    from sx.registry import build_registry

    adapters, _ = build_registry()
    offenders: list[str] = []
    for adapter in adapters:
        if Capability.ORPHANS not in adapter.capabilities or not adapter.available():
            continue
        roots = adapter.store_roots()
        for orphan in adapter.find_orphans():
            for path in orphan.paths:
                if not is_within(path, roots):
                    offenders.append(f"{adapter.name}: {path}")
    assert offenders == [], f"orphans outside their own store roots: {offenders}"


def test_delete_refuses_a_store_root_itself(tmp_path: Path):
    """A store root is 'within itself' — deleting it must still be refused."""
    root = tmp_path / "store"
    root.mkdir()
    (root / "keep.txt").write_text("data", encoding="utf-8")
    adapter = _RefusingAdapter(root, root)

    result = adapter._delete_paths([root], dry_run=False)
    assert result.removed == []
    assert "store root" in next(iter(result.skipped.values()))
    assert root.exists()


# --- failure must not be reported as success -------------------------------

def test_refused_delete_is_marked_failed_not_silent(tmp_path: Path):
    """A fully-refused delete reports failure rather than an empty success."""
    root = tmp_path / "store"
    root.mkdir()
    stray = tmp_path / "outside.tmp"  # deliberately NOT under root
    stray.write_text("x", encoding="utf-8")
    adapter = _RefusingAdapter(root, stray)

    orphan = adapter.find_orphans()[0]
    result = adapter.delete_orphan(orphan, dry_run=False)

    assert result.removed == []
    assert result.refused, "the refusal must be visible to callers"
    assert result.failed is True
    assert stray.exists(), "the file was refused, so it must still be on disk"


def test_note_counts_as_success_even_with_no_paths():
    """A database-backed delete reports success via ``note``, not ``removed``."""
    result = DeleteResult(
        removed=[],
        skipped={Path("/tmp/gone.json"): "does not exist"},
        note="deleted 8 db row(s)",
    )
    assert result.failed is False
    assert "8 db row(s)" in result.summary()
    assert result.refused == {}, "'does not exist' is not a refusal"


def test_summary_reports_partial_outcomes():
    """A mixed result names both what went and what was refused."""
    result = DeleteResult(
        removed=[Path("/a")],
        freed_bytes=2048,
        skipped={Path("/b"): "outside store root (refused)"},
    )
    assert result.failed is False
    summary = result.summary()
    assert "1 path(s)" in summary
    assert "1 refused" in summary


# --- export must never overwrite its own archive ---------------------------

def test_export_never_overwrites_an_existing_archive(tmp_path: Path):
    """Colliding export names get a suffix instead of destroying the first file.

    Real duplicate sessions produce identical names (same harness, date,
    truncated id and title slug). Since export doubles as the pre-delete safety
    net, an overwrite would destroy the archive it was asked to make.
    """
    from sx.export import export_session

    class _A:
        def load(self, session):
            return [Message(Role.USER, text="hello")]

    session = Session(
        harness="claude",
        session_id="abcd1234-0000-0000-0000-000000000000",
        title="Same Title",
        modified=datetime(2026, 5, 10, 12, 0),
    )

    first = export_session(_A(), session, tmp_path)
    second = export_session(_A(), session, tmp_path)
    third = export_session(_A(), session, tmp_path)

    assert first != second != third
    assert {p.name for p in tmp_path.iterdir()} == {first.name, second.name, third.name}
    assert len(list(tmp_path.iterdir())) == 3, "no archive may be overwritten"


# --- cascade deletion must be exact ----------------------------------------

def test_cascade_ignores_non_uuid_session_ids(tmp_path: Path, monkeypatch):
    """A short/malformed session id must never sweep unrelated state.

    With substring matching, a session file named ``a.jsonl`` yielded the id
    ``a`` and matched every cascade entry containing the letter "a".
    """
    import sx.adapters.claude as claude_mod

    fake_home = tmp_path
    monkeypatch.setattr(claude_mod, "home", lambda: fake_home)
    hist = fake_home / ".claude" / "file-history"
    hist.mkdir(parents=True)
    (hist / "an-unrelated-backup").mkdir()
    (hist / "another-one").mkdir()

    adapter = claude_mod.ClaudeAdapter()
    assert adapter.correlated_paths(Session(harness="claude", session_id="a")) == []
    assert adapter.correlated_paths(Session(harness="claude", session_id="")) == []


def test_cascade_matches_a_real_uuid(tmp_path: Path, monkeypatch):
    """A proper UUID still matches its own cascade entries."""
    import sx.adapters.claude as claude_mod

    sid = "11111111-2222-3333-4444-555555555555"
    monkeypatch.setattr(claude_mod, "home", lambda: tmp_path)
    hist = tmp_path / ".claude" / "file-history"
    hist.mkdir(parents=True)
    (hist / sid).mkdir()
    (hist / "unrelated").mkdir()
    todos = tmp_path / ".claude" / "todos"
    todos.mkdir(parents=True)
    (todos / f"{sid}-agent-x.json").write_text("{}", encoding="utf-8")

    matches = claude_mod.ClaudeAdapter().correlated_paths(
        Session(harness="claude", session_id=sid)
    )
    names = {p.name for p in matches}
    assert sid in names
    assert f"{sid}-agent-x.json" in names
    assert "unrelated" not in names


# --- unmounted volumes are not dead projects -------------------------------

def test_unmounted_volume_is_not_treated_as_deleted():
    """A project on an absent volume is unavailable, not an orphan."""
    assert mount_unavailable("/Volumes/DefinitelyNotMounted42/proj") is True
    assert mount_unavailable("/Users/somebody/gone") is False


def test_session_on_missing_volume_is_not_flagged_orphan(tmp_path: Path):
    """`is_orphan` must stay False when the whole volume is simply absent."""

    class _Adapter(HarnessAdapter):
        name = "vol"
        display = "Vol"
        capabilities = Capability.BROWSE

        def store_roots(self):
            return [tmp_path]

        def discover(self):
            return iter(())

        def load(self, session):
            return []

    from sx.adapters.base import JsonlFolderAdapter

    class _Folder(JsonlFolderAdapter):
        name = "vol"
        display = "Vol"

        def store_roots(self):
            return [tmp_path]

        def group_key(self, path, first):
            return "/Volumes/DefinitelyNotMounted42/proj"

    f = tmp_path / "s.jsonl"
    f.write_text('{"x":1}\n', encoding="utf-8")
    session = _Folder()._session_from_file(f)
    assert session.is_orphan is False


# --- liveness guard --------------------------------------------------------

def test_unknown_liveness_is_treated_as_live(tmp_path: Path):
    """When liveness can't be determined the session counts as active.

    opencode sessions have an optional sidecar; when it was missing the guard
    returned "not live", so deleting a conversation being written needed no
    typed confirmation.
    """

    class _NoSignal(HarnessAdapter):
        name = "quiet"
        display = "Quiet"
        capabilities = Capability.BROWSE | Capability.DELETE

        def store_roots(self):
            return [tmp_path]

        def discover(self):
            return iter(())

        def load(self, session):
            return []

        def last_activity(self, session):
            return None

    service = DeleteService({"quiet": _NoSignal()}, log_path=tmp_path / "ops.log")
    assert service.is_active(Session(harness="quiet", session_id="x")) is True


def test_adapter_supplied_liveness_is_used(tmp_path: Path):
    """An adapter's own activity signal decides liveness, not a file mtime."""

    class _Old(HarnessAdapter):
        name = "old"
        display = "Old"
        capabilities = Capability.BROWSE | Capability.DELETE

        def store_roots(self):
            return [tmp_path]

        def discover(self):
            return iter(())

        def load(self, session):
            return []

        def last_activity(self, session):
            return datetime.now() - timedelta(hours=5)

    service = DeleteService({"old": _Old()}, log_path=tmp_path / "ops.log")
    assert service.is_active(Session(harness="old", session_id="x")) is False


# --- terminal escapes ------------------------------------------------------

def test_control_characters_are_neutralized_but_visible():
    """Escape sequences become inert glyphs; tabs and newlines survive."""
    evil = "hi\x1b]0;pwned\x07\x1b[2J"
    out = sanitize_text(evil)
    assert "\x1b" not in out
    assert "\x07" not in out
    assert "␛" in out and "␇" in out
    assert sanitize_text("a\tb\nc") == "a\tb\nc"


def test_rendered_transcript_contains_no_escapes():
    """A hostile transcript cannot drive the terminal when viewed."""
    from sx.render import messages_to_text

    text = messages_to_text(
        [Message(Role.USER, text="x\x1b]52;c;cHduZWQ=\x07", thinking="t\x1b[31m")]
    )
    assert "\x1b" not in text.plain


def test_exported_markdown_contains_no_escapes(tmp_path: Path):
    """Escapes must not survive into the exported archive either."""
    from sx.export import messages_to_markdown

    md = messages_to_markdown(
        Session(harness="h", session_id="i", title="t\x1b[31m"),
        [Message(Role.USER, text="body\x1b]0;x\x07")],
    )
    assert "\x1b" not in md


# --- op-log location -------------------------------------------------------

def test_op_log_defaults_to_the_working_directory(tmp_path: Path, monkeypatch):
    """The op-log sits beside the work it describes, never under ~/.local."""
    from sx.service import default_log_path

    monkeypatch.delenv("SX_LOG_FILE", raising=False)
    monkeypatch.chdir(tmp_path)
    path = default_log_path()
    assert path == tmp_path / "sx-deletions.log"
    assert ".local" not in str(path)


def test_op_log_honors_an_explicit_override(tmp_path: Path, monkeypatch):
    """``SX_LOG_FILE`` collects deletions from every directory into one file."""
    from sx.service import default_log_path

    target = tmp_path / "central" / "all-deletions.log"
    monkeypatch.setenv("SX_LOG_FILE", str(target))
    assert default_log_path() == target


def test_op_log_is_owner_only_and_records_the_deletion(tmp_path: Path):
    """The log is written 0600 — it holds chat-derived titles and paths."""
    import stat

    root = tmp_path / "store"
    root.mkdir()
    victim = root / "gone.jsonl"
    victim.write_text("{}", encoding="utf-8")
    log = tmp_path / "sx-deletions.log"

    class _A(HarnessAdapter):
        name = "t"
        display = "T"
        capabilities = Capability.BROWSE | Capability.DELETE

        def store_roots(self):
            return [root]

        def discover(self):
            return iter(())

        def load(self, session):
            return []

    service = DeleteService({"t": _A()}, log_path=log)
    service.delete(Session(harness="t", session_id="s1", title="x", paths=[victim]))

    assert log.exists()
    assert stat.S_IMODE(log.stat().st_mode) == 0o600
    assert service.last_log_error is None
    assert json.loads(log.read_text(encoding="utf-8").splitlines()[-1])["id"] == "s1"


# --- update cache ----------------------------------------------------------

@pytest.mark.parametrize("payload", ["[]", "null", '"a string"', "42"])
def test_non_dict_update_cache_does_not_crash(tmp_path: Path, monkeypatch, payload):
    """A corrupt cache must not crash `sx update` with an AttributeError."""
    import sx.update as update_mod

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("SX_NO_UPDATE_CHECK", raising=False)
    cache = tmp_path / "sx" / "update-check.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(payload, encoding="utf-8")

    assert update_mod._read_cache() is None
    monkeypatch.setattr(update_mod, "fetch_latest_version", lambda *a, **k: None)
    assert update_mod.check_for_update(force=True) is None


def test_failed_fetch_is_cached_to_throttle_retries(tmp_path: Path, monkeypatch):
    """A `None` result is still recorded, so the daily throttle engages."""
    import sx.update as update_mod

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("SX_NO_UPDATE_CHECK", raising=False)
    monkeypatch.setattr(update_mod, "fetch_latest_version", lambda *a, **k: None)

    update_mod.check_for_update(force=True)
    cached = json.loads((tmp_path / "sx" / "update-check.json").read_text())
    assert "checked_at" in cached
    assert cached["latest"] is None
