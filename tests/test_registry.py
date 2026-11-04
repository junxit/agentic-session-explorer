"""Tests for sx.registry: the registry assembles without reading real data."""

from __future__ import annotations

from sx.registry import build_registry


def test_build_registry_returns_lists_and_real_adapters():
    """build_registry returns (list, list) including the real adapters.

    This only asserts the registry assembles; it never enumerates discovered
    sessions, so no real ``~/.claude``/``~/.codex``/``~/.gemini``/``~/.pi``/
    ``~/.omp`` data is read.
    """
    adapters, errors = build_registry()
    assert isinstance(adapters, list)
    assert isinstance(errors, list)

    names = {a.name for a in adapters}
    assert {"claude", "codex", "gemini", "pi", "omp"} <= names
    assert not {name for name, _ in errors} & {"PiAdapter", "OmpAdapter"}


def test_an_adapter_hands_its_store_to_no_other_harness_by_default():
    """``store_owner`` is ``None`` unless an adapter has stepped aside for another harness."""
    from sx.adapters.base import HarnessAdapter
    from sx.adapters.claude import ClaudeAdapter
    from sx.adapters.codex import CodexAdapter
    from sx.adapters.gemini import GeminiAdapter
    from sx.adapters.opencode import OpencodeAdapter

    class Minimal(HarnessAdapter):
        """The least an adapter implements."""

        def store_roots(self):
            """Own nothing."""
            return []

        def discover(self):
            """Find nothing."""
            return iter(())

        def load(self, session):
            """Load nothing."""
            return []

    for adapter in (Minimal(), ClaudeAdapter(), CodexAdapter(), GeminiAdapter(), OpencodeAdapter()):
        assert adapter.store_owner() is None
