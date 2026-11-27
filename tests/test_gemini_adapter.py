"""Tests for sx.adapters.gemini mutation-log replay on SYNTHETIC data.

A synthetic chat file is built under tmp_path mimicking the real grammar:
a header line, a ``$set.messages`` bulk replace, flat append records, and a
trailing ``$set`` that is not a message bulk. ``load`` is called with an
explicit Session whose paths point at the synthetic file, so the real
``~/.gemini`` store is never read.
"""

from __future__ import annotations

import json
from pathlib import Path

from sx.adapters.gemini import GeminiAdapter
from sx.model import Role, Session


def _build_chat(tmp_path: Path) -> tuple[Path, str]:
    """Create a synthetic gemini chat file + .project_root marker under tmp.

    Layout mirrors production: ``<root>/<hash>/chats/session-*.jsonl`` with the
    marker at ``<root>/<hash>/.project_root``.

    Returns:
        A tuple of (chat file path, project_root contents).
    """
    project_root = "/real/project/dir"
    root = tmp_path / "root"
    hash_dir = root / "h"
    chats = hash_dir / "chats"
    chats.mkdir(parents=True)
    (hash_dir / ".project_root").write_text(project_root, encoding="utf-8")

    lines = [
        {
            "sessionId": "s1",
            "projectHash": "h",
            "startTime": "2026-05-30T13:27:04Z",
            "kind": "chat",
        },
        {
            "$set": {
                "messages": [
                    {
                        "id": "m0",
                        "timestamp": "2026-05-30T13:27:05Z",
                        "type": "user",
                        "content": [{"text": "<session_context>noise"}],
                    }
                ]
            }
        },
        {
            "id": "m1",
            "timestamp": "2026-05-30T13:27:06Z",
            "type": "user",
            "content": [{"text": "Write reverse.py"}],
        },
        {
            "id": "m2",
            "timestamp": "2026-05-30T13:27:07Z",
            "type": "gemini",
            "content": [{"text": "Here you go"}],
        },
        {"$set": {"lastUpdated": "2026-05-30T13:27:08Z"}},
    ]
    chat = chats / "session-1.jsonl"
    chat.write_text("".join(json.dumps(l) + "\n" for l in lines), encoding="utf-8")
    return chat, project_root


def test_gemini_replay_skips_session_context_and_orders(tmp_path: Path):
    """Replay drops the <session_context> bulk record and keeps real turns."""
    chat, _ = _build_chat(tmp_path)
    session = Session(harness="gemini", session_id="s1", paths=[chat])
    messages = GeminiAdapter().load(session)

    texts = [m.text for m in messages]
    roles = [m.role for m in messages]
    assert texts == ["Write reverse.py", "Here you go"]
    assert roles == [Role.USER, Role.ASSISTANT]


def test_gemini_group_key_resolves_marker(tmp_path: Path):
    """group_key reads the sibling .project_root marker contents."""
    chat, project_root = _build_chat(tmp_path)
    assert GeminiAdapter().group_key(chat, None) == project_root


def test_gemini_group_key_missing_marker(tmp_path: Path):
    """A chat file with no marker yields None for group_key."""
    chats = tmp_path / "root" / "nomarker" / "chats"
    chats.mkdir(parents=True)
    chat = chats / "session-2.jsonl"
    chat.write_text(
        json.dumps({"sessionId": "s2", "startTime": "2026-05-30T00:00:00Z"}) + "\n",
        encoding="utf-8",
    )
    assert GeminiAdapter().group_key(chat, None) is None


def test_antigravitys_folders_in_dot_gemini_are_protected(tmp_path, monkeypatch):
    """Gemini's store root is all of ~/.gemini; Antigravity's entries in it are refused outright."""
    from sx.adapters import gemini as gemini_module
    from sx.adapters.gemini import GeminiAdapter

    monkeypatch.setattr(gemini_module, "home", lambda: tmp_path)
    base = tmp_path / ".gemini"
    for name in ("antigravity-cli", "antigravity-nightly", "config", "tmp"):
        (base / name).mkdir(parents=True)
        (base / name / "f").write_text("x", encoding="utf-8")
    adapter = GeminiAdapter()
    protected = adapter.protected_paths()
    assert base / "antigravity-cli" in protected and base / "antigravity-nightly" in protected
    assert base / "antigravity-ide" in protected  # named even before it exists
    result = adapter.delete_paths(
        [base / "antigravity-cli" / "f", base / "config", base / "tmp" / "f"]
    )
    assert set(result.refused) == {base / "antigravity-cli" / "f", base / "config"}
    assert result.removed == [base / "tmp" / "f"]
