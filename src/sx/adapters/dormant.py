"""Dormant adapters: harnesses sx recognizes but does not yet parse.

Each adapter here names a real AI coding harness and the location it stores
sessions, but advertises :attr:`~sx.model.Capability.NONE`. That makes the
harness appear in the registry and UI — greyed out — without ever browsing or
deleting anything, which keeps the tool honest until a full adapter is written.

``store_roots`` returns each harness's *canonical* locations (unfiltered by
existence) so the value is stable: it doubles as the "where sx would look"
display and as the delete-guard allowlist. Installed-vs-not is decided
separately by :meth:`~sx.adapters.base.HarnessAdapter.available`, which checks
whether any of those roots actually exists.
"""

from __future__ import annotations

from pathlib import Path

from ..model import Capability
from ..util import home
from .base import HarnessAdapter


class DormantAdapter(HarnessAdapter):
    """Base for harnesses that are recognized but not yet parse-supported.

    Subclasses set :attr:`name`, :attr:`display`, and :meth:`store_roots`.
    Capabilities stay :attr:`~sx.model.Capability.NONE`, so discovery yields
    nothing, loading returns nothing, and deletion is never offered.
    """

    capabilities = Capability.NONE

    def discover(self):
        """Yield nothing; dormant harnesses expose no sessions.

        Returns:
            An empty iterator.
        """
        return iter(())

    def load(self, session):
        """Return no messages; dormant harnesses cannot be browsed.

        Args:
            session: Unused; present to satisfy the adapter contract.

        Returns:
            An empty list.
        """
        return []


class QwenAdapter(DormantAdapter):
    """Qwen Code, a Gemini CLI fork. JSONL chats under ``~/.qwen``; pending."""

    name = "qwen"
    display = "Qwen Code"

    def store_roots(self) -> list[Path]:
        """Return Qwen's canonical store locations."""
        return [home() / ".qwen" / "tmp", home() / ".qwen" / "history"]


class ContinueAdapter(DormantAdapter):
    """Continue (IDE extension). JSON session files; parse support pending."""

    name = "continue"
    display = "Continue"

    def store_roots(self) -> list[Path]:
        """Return Continue's canonical store location."""
        return [home() / ".continue" / "sessions"]


class GooseAdapter(DormantAdapter):
    """Goose (Block's CLI agent). Session files; parse support pending."""

    name = "goose"
    display = "Goose"

    def store_roots(self) -> list[Path]:
        """Return Goose's canonical store location."""
        return [home() / ".local" / "share" / "goose" / "sessions"]


class OpencodeAdapter(DormantAdapter):
    """opencode (SST). Stores session data on disk; parse support pending."""

    name = "opencode"
    display = "opencode"

    def store_roots(self) -> list[Path]:
        """Return opencode's canonical store locations."""
        return [home() / ".local" / "share" / "opencode", home() / ".opencode"]


class ClineAdapter(DormantAdapter):
    """Cline (VS Code extension). Per-workspace state dirs; parse pending."""

    name = "cline"
    display = "Cline"

    def store_roots(self) -> list[Path]:
        """Return Cline's canonical store location."""
        return [home() / ".cline" / "data" / "workspaces"]


class CursorAdapter(DormantAdapter):
    """Cursor (AI IDE). SQLite-backed chat history; parse support pending."""

    name = "cursor"
    display = "Cursor"

    def store_roots(self) -> list[Path]:
        """Return Cursor's canonical configuration directory."""
        return [home() / ".cursor"]


class CrushAdapter(DormantAdapter):
    """Crush (Charm CLI). SQLite-backed sessions; parse support pending."""

    name = "crush"
    display = "Crush"

    def store_roots(self) -> list[Path]:
        """Return Crush's canonical store locations."""
        return [home() / ".local" / "share" / "crush", home() / ".crush"]


DORMANT_ADAPTERS = [
    QwenAdapter,
    ContinueAdapter,
    GooseAdapter,
    OpencodeAdapter,
    ClineAdapter,
    CursorAdapter,
    CrushAdapter,
]
