"""Tests for sx.adapters.claude on SYNTHETIC files in tmp_path.

Methods are called directly (title_for, parse_line, _clean_title_text,
_decode_folder) so the real ``~/.claude`` store is never read.
"""

from __future__ import annotations

import json
from pathlib import Path

from sx.adapters.claude import ClaudeAdapter, _clean_title_text, _decode_folder
from sx.model import Role


def _user(content) -> dict:
    """Build a Claude ``user`` record with the given message content."""
    return {"type": "user", "message": {"role": "user", "content": content}}


def _assistant(content) -> dict:
    """Build a Claude ``assistant`` record with the given message content."""
    return {"type": "assistant", "message": {"role": "assistant", "content": content}}


def _write(path: Path, records: list[dict]) -> Path:
    """Write Claude JSONL records to ``path`` and return it."""
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    return path


# --- _clean_title_text ---------------------------------------------------


def test_clean_title_strips_command_block():
    """A pure command-name/message block reduces to an empty string."""
    text = (
        "<command-name>/model</command-name>\n"
        "<command-message>model</command-message>\n"
        "<command-args></command-args>"
    )
    assert _clean_title_text(text) == ""


def test_clean_title_strips_inline_command_and_collapses_ws():
    """An inline command tag is removed and whitespace collapsed."""
    assert (
        _clean_title_text("Please   <command-name>/foo</command-name>  fix the bug")
        == "Please fix the bug"
    )


def test_clean_title_local_command_block():
    """A local-command block is also stripped to empty."""
    text = "<local-command-stdout>blah</local-command-stdout>"
    assert _clean_title_text(text) == ""


# --- _decode_folder ------------------------------------------------------


def test_decode_folder():
    """Encoded project folder decodes to an absolute path."""
    assert _decode_folder("-Users-alice-x") == "/Users/alice/x"


# --- title_for -----------------------------------------------------------


def test_title_skips_pure_command_first_message(tmp_path: Path):
    """A pure-command first user msg is skipped; the next real msg is the title."""
    f = _write(
        tmp_path / "c1.jsonl",
        [
            _user(
                "<command-name>/model</command-name>\n"
                "<command-message>model</command-message>\n"
                "<command-args></command-args>"
            ),
            _user("Help me understand the repo structure"),
        ],
    )
    assert ClaudeAdapter().title_for(f, None) == "Help me understand the repo structure"


def test_title_inline_command_noise_stripped(tmp_path: Path):
    """Inline command noise in the first message is cleaned for the title."""
    f = _write(
        tmp_path / "c2.jsonl",
        [_user("Please <command-name>/foo</command-name> fix the bug")],
    )
    assert ClaudeAdapter().title_for(f, None) == "Please fix the bug"


def test_title_custom_preferred_over_ai_over_user(tmp_path: Path):
    """customTitle wins over aiTitle, which wins over the first user message."""
    f = _write(
        tmp_path / "c3.jsonl",
        [
            _user("raw prompt"),
            {"type": "ai-title", "aiTitle": "AI Generated"},
            {"type": "custom-title", "customTitle": "My Custom"},
        ],
    )
    assert ClaudeAdapter().title_for(f, None) == "My Custom"


def test_title_ai_preferred_over_user(tmp_path: Path):
    """aiTitle wins over the first user message when no customTitle exists."""
    f = _write(
        tmp_path / "c4.jsonl",
        [
            _user("raw prompt"),
            {"type": "ai-title", "aiTitle": "AI Generated"},
        ],
    )
    assert ClaudeAdapter().title_for(f, None) == "AI Generated"


# --- parse_line ----------------------------------------------------------


def test_parse_user_string_content():
    """A user record whose content is a string yields a USER message."""
    m = ClaudeAdapter().parse_line(_user("hello"))
    assert m is not None
    assert m.role is Role.USER
    assert m.text == "hello"


def test_parse_user_list_text_blocks():
    """A user record whose content is a list of text blocks yields USER text."""
    m = ClaudeAdapter().parse_line(_user([{"type": "text", "text": "hi there"}]))
    assert m is not None
    assert m.role is Role.USER
    assert m.text == "hi there"


def test_parse_assistant_thinking_text_tool():
    """Assistant list with thinking+text+tool_use sets all three fields."""
    m = ClaudeAdapter().parse_line(
        _assistant(
            [
                {"type": "thinking", "thinking": "pondering"},
                {"type": "text", "text": "the answer"},
                {"type": "tool_use", "name": "Bash"},
            ]
        )
    )
    assert m is not None
    assert m.role is Role.ASSISTANT
    assert m.text == "the answer"
    assert m.thinking == "pondering"
    assert m.tool_summary == "Bash"


def test_parse_meta_record_skipped():
    """A record with isMeta=True is skipped (returns None)."""
    assert ClaudeAdapter().parse_line({"isMeta": True, **_user("x")}) is None


def test_parse_tool_result_only_user_is_tool_role():
    """A user record carrying only tool_result content yields a TOOL message."""
    m = ClaudeAdapter().parse_line(
        _user([{"type": "tool_result", "content": "result text here"}])
    )
    assert m is not None
    assert m.role is Role.TOOL
    assert "result text here" in m.text
    assert m.tool_summary == "tool result"
