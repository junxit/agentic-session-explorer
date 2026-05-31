"""Tests for sx.registry: the registry assembles without reading real data."""

from __future__ import annotations

from sx.registry import build_registry


def test_build_registry_returns_lists_and_real_adapters():
    """build_registry returns (list, list) including the three real adapters.

    This only asserts the registry assembles; it never enumerates discovered
    sessions, so no real ``~/.claude``/``~/.codex``/``~/.gemini`` data is read.
    """
    adapters, errors = build_registry()
    assert isinstance(adapters, list)
    assert isinstance(errors, list)

    names = {a.name for a in adapters}
    assert {"claude", "codex", "gemini"} <= names
