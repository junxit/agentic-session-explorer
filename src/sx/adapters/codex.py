from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from ..model import Message, Role, Session
from ..util import home, parse_ts, read_first_line
from .base import JsonlFolderAdapter

# Matches the trailing UUID in a ``rollout-<ts>-<uuid>.jsonl`` filename.
_UUID_RE = re.compile(
    r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)


def _preview(value, limit: int) -> str:
    """Render an arbitrary value to a short single-line preview string.

    Args:
        value: The value to summarise (string, mapping, or other JSON value).
        limit: Maximum number of characters to keep.

    Returns:
        A truncated preview, or an empty string for ``None``.
    """
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


class CodexAdapter(JsonlFolderAdapter):
    """Adapter for Codex chat sessions.

    Sessions are stored as date-partitioned rollout files at
    ``~/.codex/sessions/YYYY/MM/DD/rollout-<ts>-<uuid>.jsonl``. The first record
    is a ``session_meta`` envelope carrying the session id, working directory
    and creation timestamp.

    Attributes:
        name: Machine identifier (``"codex"``).
        display: Human-readable harness name.
    """

    name = "codex"
    display = "Codex"

    def store_roots(self) -> list[Path]:
        """Return the managed directories for Codex.

        Returns:
            A single-element list containing ``~/.codex/sessions``.
        """
        return [home() / ".codex" / "sessions"]

    def session_files(self) -> Iterator[Path]:
        """Yield Codex rollout files.

        The date-partitioned glob is used first; if it yields nothing (an
        unexpected layout) a recursive scan is used as a fallback.

        Returns:
            An iterator over rollout ``.jsonl`` files.
        """
        root = home() / ".codex" / "sessions"
        if not root.exists():
            return
        files = sorted(root.glob("*/*/*/*.jsonl"))
        if files:
            yield from files
        else:
            yield from sorted(root.rglob("*.jsonl"))

    @staticmethod
    def _meta_payload(first: dict | None) -> dict | None:
        """Return the ``session_meta`` payload from a file's first record.

        Args:
            first: The first parsed record of a session file.

        Returns:
            The payload mapping, or ``None`` if the record is not session meta.
        """
        if isinstance(first, dict) and first.get("type") == "session_meta":
            payload = first.get("payload")
            if isinstance(payload, dict):
                return payload
        return None

    def group_key(self, path: Path, first: dict | None) -> str | None:
        """Resolve the project path for a session file.

        Args:
            path: The session file being inspected.
            first: The first parsed record.

        Returns:
            The working directory from the session meta, or ``None``.
        """
        payload = self._meta_payload(first)
        if payload:
            cwd = payload.get("cwd")
            if isinstance(cwd, str) and cwd:
                return cwd
        return None

    def session_id_for(self, path: Path, first: dict | None) -> str:
        """Return the session id for a file.

        The ``session_meta`` id is preferred; otherwise the UUID embedded in the
        filename is used, falling back to the filename stem.

        Args:
            path: The session file being inspected.
            first: The first parsed record.

        Returns:
            The session identifier.
        """
        payload = self._meta_payload(first)
        if payload:
            sid = payload.get("id")
            if isinstance(sid, str) and sid:
                return sid
        m = _UUID_RE.search(path.name)
        if m:
            return m.group(1)
        return path.stem

    def title_for(self, path: Path, first: dict | None) -> str:
        """Derive a cheap title from the working directory and date.

        Args:
            path: The session file being inspected.
            first: The first parsed record.

        Returns:
            A title such as ``"project · 2025-09-04"``, or the filename stem.
        """
        payload = self._meta_payload(first)
        if payload:
            cwd = payload.get("cwd")
            base = Path(cwd).name if isinstance(cwd, str) and cwd else ""
            ts = payload.get("timestamp")
            date = ""
            if isinstance(ts, str) and ts:
                date = ts[:10]
            if base and date:
                return f"{base} · {date}"
            if base:
                return base
        return path.stem

    def _session_from_file(self, path: Path) -> Session:
        """Build a :class:`Session` and enrich it with the creation time.

        Args:
            path: The session file to describe.

        Returns:
            A :class:`Session` with ``created`` populated from session meta.
        """
        session = super()._session_from_file(path)
        first = read_first_line(path)
        payload = self._meta_payload(first)
        if payload:
            session.created = parse_ts(payload.get("timestamp"))
        return session

    def parse_line(self, obj: dict) -> Message | None:
        """Convert a Codex top-level record into a :class:`Message`.

        Only ``response_item`` records carry transcript content. Environment
        context messages and empty reasoning blocks are skipped.

        Args:
            obj: A parsed top-level JSON record.

        Returns:
            A :class:`Message`, or ``None`` if the record should be skipped.
        """
        if obj.get("type") != "response_item":
            return None
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            return None
        pt = payload.get("type")
        ts = parse_ts(obj.get("timestamp"))

        if pt == "message":
            content = payload.get("content")
            parts: list[str] = []
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        t = item.get("text")
                        if isinstance(t, str):
                            parts.append(t)
                    elif isinstance(item, str):
                        parts.append(item)
            elif isinstance(content, str):
                parts.append(content)
            text = "\n".join(p for p in parts if p).strip()
            if text.startswith("<environment_context>"):
                return None
            role_raw = payload.get("role")
            if role_raw == "user":
                role = Role.USER
            elif role_raw == "assistant":
                role = Role.ASSISTANT
            else:
                role = Role.OTHER
            if not text:
                return None
            return Message(role, text=text, timestamp=ts)

        if pt == "reasoning":
            summary = payload.get("summary")
            parts: list[str] = []
            if isinstance(summary, list):
                for item in summary:
                    if isinstance(item, dict):
                        t = item.get("text")
                        if isinstance(t, str):
                            parts.append(t)
                    elif isinstance(item, str):
                        parts.append(item)
            thinking = "\n".join(p for p in parts if p).strip()
            if not thinking:
                return None
            return Message(Role.ASSISTANT, thinking=thinking, timestamp=ts)

        if pt == "function_call":
            name = payload.get("name") or "function"
            return Message(
                Role.ASSISTANT,
                text=_preview(payload.get("arguments"), 120),
                tool_summary=f"{name}(...)",
                timestamp=ts,
            )

        if pt == "function_call_output":
            return Message(
                Role.TOOL,
                text=_preview(self._output_text(payload.get("output")), 200),
                tool_summary="output",
                timestamp=ts,
            )

        if pt == "custom_tool_call":
            name = payload.get("name") or "tool"
            return Message(
                Role.ASSISTANT,
                text=_preview(payload.get("input"), 120),
                tool_summary=f"{name}(...)",
                timestamp=ts,
            )

        if pt == "custom_tool_call_output":
            return Message(
                Role.TOOL,
                text=_preview(self._output_text(payload.get("output")), 200),
                tool_summary="output",
                timestamp=ts,
            )

        return None

    @staticmethod
    def _output_text(output) -> str:
        """Normalise a tool-output value to text.

        Codex tool output may be a plain string or a mapping with an ``output``
        key (and optional ``metadata``).

        Args:
            output: The raw output value.

        Returns:
            A text rendering of the output.
        """
        if isinstance(output, dict):
            inner = output.get("output")
            if isinstance(inner, str):
                return inner
            return str(output)
        if isinstance(output, str):
            return output
        if output is None:
            return ""
        return str(output)
