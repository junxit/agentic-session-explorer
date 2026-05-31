"""Tests for sx.adapters.codex on SYNTHETIC data.

parse_line and the meta-reading hooks are exercised directly; the real
``~/.codex`` store is never read.
"""

from __future__ import annotations

import json
from pathlib import Path

from sx.adapters.codex import CodexAdapter
from sx.model import Role


def _ri(payload: dict, ts: str = "2026-05-30T12:00:00Z") -> dict:
    """Wrap a payload as a Codex ``response_item`` top-level record."""
    return {"type": "response_item", "payload": payload, "timestamp": ts}


def _meta(cwd: str = "/home/proj", sid: str = "sess-123") -> dict:
    """Build a Codex ``session_meta`` first record."""
    return {
        "type": "session_meta",
        "payload": {
            "id": sid,
            "cwd": cwd,
            "timestamp": "2026-05-30T12:00:00Z",
        },
    }


def test_message_user_role():
    """A response_item message with role=user yields a USER message."""
    m = CodexAdapter().parse_line(
        _ri({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]})
    )
    assert m is not None
    assert m.role is Role.USER
    assert m.text == "hello"


def test_message_assistant_role():
    """A response_item message with role=assistant yields an ASSISTANT message."""
    m = CodexAdapter().parse_line(
        _ri(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi back"}],
            }
        )
    )
    assert m is not None
    assert m.role is Role.ASSISTANT
    assert m.text == "hi back"


def test_message_environment_context_skipped():
    """A message whose text starts with <environment_context> is skipped."""
    m = CodexAdapter().parse_line(
        _ri(
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "<environment_context>noise here"}],
            }
        )
    )
    assert m is None


def test_reasoning_sets_thinking():
    """A reasoning record with a summary yields ASSISTANT with thinking set."""
    m = CodexAdapter().parse_line(
        _ri({"type": "reasoning", "summary": [{"type": "summary_text", "text": "think hard"}]})
    )
    assert m is not None
    assert m.role is Role.ASSISTANT
    assert m.thinking == "think hard"


def test_function_call_tool_summary_contains_name():
    """A function_call yields ASSISTANT whose tool_summary contains the name."""
    m = CodexAdapter().parse_line(
        _ri({"type": "function_call", "name": "shell", "arguments": '{"cmd": "ls"}'})
    )
    assert m is not None
    assert m.role is Role.ASSISTANT
    assert "shell" in m.tool_summary


def test_function_call_output_is_tool_role():
    """A function_call_output yields a TOOL message."""
    m = CodexAdapter().parse_line(_ri({"type": "function_call_output", "output": "done"}))
    assert m is not None
    assert m.role is Role.TOOL


def test_group_key_and_session_id_from_meta(tmp_path: Path):
    """group_key and session_id_for read cwd/id from the session_meta record."""
    path = tmp_path / "rollout-2026-05-30-abc.jsonl"
    path.write_text(json.dumps(_meta()) + "\n", encoding="utf-8")
    first = _meta()
    adapter = CodexAdapter()
    assert adapter.group_key(path, first) == "/home/proj"
    assert adapter.session_id_for(path, first) == "sess-123"
