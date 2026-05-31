"""Tests for sx.render.messages_to_text: empty case and markup safety."""

from __future__ import annotations

from sx.model import Message, Role
from sx.render import messages_to_text


def test_render_empty_messages():
    """An empty transcript renders the no-messages placeholder."""
    assert "(no messages)" in messages_to_text([]).plain


def test_render_is_markup_injection_safe():
    """Bracketed text in content appears literally (not parsed as markup)."""
    text = "[not markup]"
    result = messages_to_text([Message(Role.USER, text=text)])
    assert text in result.plain
