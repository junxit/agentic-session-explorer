from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..model import Message, Orphan, OrphanKind, Role, Session
from ..util import (
    dir_size,
    home,
    iter_jsonl,
    parse_ts,
    read_first_line,
)
from .base import JsonlFolderAdapter

# Roles that denote an assistant turn in either Gemini record shape.
_ASSISTANT_TYPES = ("gemini", "model", "assistant")


class GeminiAdapter(JsonlFolderAdapter):
    """Adapter for Gemini CLI chat sessions.

    Chat logs live under ``~/.gemini/tmp/<hash>/chats/session-*.jsonl`` (and the
    newer ``~/.gemini/history/<hash>/chats/...`` location). Each file is a
    *mutation log*: an opening header followed by ``$set`` operations, where a
    ``$set`` carrying a ``messages`` list replaces the entire conversation. The
    project working directory for a hash is recorded in a sibling
    ``.project_root`` marker file.

    Each message record uses a flat shape: ``{"id", "timestamp", "type",
    "content"}`` for user turns, plus optional ``thoughts`` (reasoning) and
    ``toolCalls`` (a list of ``{"name", "args", ...}``) on assistant turns.

    Attributes:
        name: Machine identifier (``"gemini"``).
        display: Human-readable harness name.
    """

    name = "gemini"
    display = "Gemini CLI"

    def store_roots(self) -> list[Path]:
        """Return the managed directory for Gemini.

        This is ``~/.gemini`` itself rather than only ``tmp``/``history``,
        because stray ``projects.json.*.tmp`` files reported by
        :meth:`find_orphans` live directly in that directory. When the roots were
        narrower, every one of those orphans was silently refused by the delete
        guard while the UI reported success.

        Returned unfiltered by existence so the value is stable and doubles as
        the delete-guard allowlist.

        Returns:
            ``[~/.gemini]``.
        """
        return [home() / ".gemini"]

    def session_files(self) -> Iterator[Path]:
        """Yield Gemini chat session files.

        Only the ``chats`` subdirectories are scanned; ``~/.gemini/tmp`` holds
        many unrelated files that must not be treated as sessions.

        Returns:
            An iterator over ``session-*.jsonl`` chat files.
        """
        base = home() / ".gemini"
        files: list[Path] = []
        for sub in ("tmp", "history"):
            root = base / sub
            if root.exists():
                files.extend(root.glob("*/chats/*.jsonl"))
        yield from sorted(files)

    @staticmethod
    def _project_root_marker(path: Path) -> Path:
        """Return the ``.project_root`` marker path for a chat file.

        The chat file lives at ``<hash>/chats/session-*.jsonl`` and the marker
        is ``<hash>/.project_root``.

        Args:
            path: A chat session file.

        Returns:
            The expected marker path.
        """
        return path.parent.parent / ".project_root"

    def group_key(self, path: Path, first: dict | None) -> str | None:
        """Resolve the project path for a chat file via its marker.

        Args:
            path: The chat session file being inspected.
            first: The first parsed record (unused).

        Returns:
            The project working directory, or ``None`` if the marker is missing
            or empty.
        """
        marker = self._project_root_marker(path)
        try:
            text = marker.read_text(encoding="utf-8").strip()
        except (OSError, FileNotFoundError):
            return None
        return text or None

    @staticmethod
    def _is_header(obj: dict) -> bool:
        """Return whether a record is the opening session header.

        Args:
            obj: A parsed record.

        Returns:
            ``True`` if the record carries a ``sessionId``.
        """
        return "sessionId" in obj

    def session_id_for(self, path: Path, first: dict | None) -> str:
        """Return the session id for a file.

        Args:
            path: The chat session file being inspected.
            first: The first parsed record (the header, if present).

        Returns:
            The header ``sessionId`` when available, else the filename stem.
        """
        if isinstance(first, dict) and self._is_header(first):
            sid = first.get("sessionId")
            if isinstance(sid, str) and sid:
                return sid
        return path.stem

    def title_for(self, path: Path, first: dict | None) -> str:
        """Derive a title from the project directory and start date.

        Args:
            path: The chat session file being inspected.
            first: The first parsed record (the header, if present).

        Returns:
            A title string, falling back to the filename stem.
        """
        project = self.group_key(path, first)
        base = Path(project).name if project else ""
        date = ""
        if isinstance(first, dict):
            start = first.get("startTime")
            if isinstance(start, str) and start:
                date = start[:10]
        if base and date:
            return f"{base} · {date}"
        if base:
            return base
        return path.stem

    def _replay(self, path: Path) -> list[dict]:
        """Replay the mutation log into a final list of message records.

        Args:
            path: The chat session file to replay.

        Returns:
            The reconstructed list of raw message records.
        """
        records: list[dict] = []
        for obj in iter_jsonl(path):
            if not isinstance(obj, dict):
                continue
            if self._is_header(obj):
                msgs = obj.get("messages")
                if isinstance(msgs, list) and msgs:
                    records = list(msgs)
                continue
            if "$set" in obj:
                changes = obj.get("$set")
                if isinstance(changes, dict) and isinstance(
                    changes.get("messages"), list
                ):
                    records = list(changes["messages"])
                continue
            if obj.get("type") in ("user",) + _ASSISTANT_TYPES:
                records.append(obj)
        return records

    @staticmethod
    def _content_text(content) -> str:
        """Flatten a Gemini ``content`` value into plain text.

        ``content`` is normally a list of ``{"text": ...}`` parts, but a plain
        string and bare string items are tolerated.

        Args:
            content: The record's ``content`` value.

        Returns:
            The joined text, possibly empty.
        """
        parts: list[str] = []
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
        return "\n".join(t for t in parts if t).strip()

    @classmethod
    def _convert(cls, record: dict) -> Message | None:
        """Convert a single Gemini message record into a :class:`Message`.

        Records use a flat ``type``/``content`` shape. Assistant turns may also
        carry a ``thoughts`` field (reasoning) and a ``toolCalls`` list (tool
        invocations, each ``{"name", "args", "result"/"resultDisplay", ...}``).
        System-context bootstrap messages are skipped.

        Args:
            record: A raw message record.

        Returns:
            A :class:`Message`, or ``None`` if it should be skipped.
        """
        ts = parse_ts(record.get("timestamp"))
        speaker = record.get("type") or record.get("role")
        if speaker == "user":
            role = Role.USER
        elif speaker in _ASSISTANT_TYPES:
            role = Role.ASSISTANT
        else:
            role = Role.OTHER

        text = cls._content_text(record.get("content"))
        if text.startswith("<session_context>"):
            return None

        thinking = ""
        thoughts = record.get("thoughts")
        if isinstance(thoughts, str):
            thinking = thoughts.strip()
        elif isinstance(thoughts, list):
            bits: list[str] = []
            for t in thoughts:
                if isinstance(t, str):
                    bits.append(t)
                elif isinstance(t, dict):
                    bits.append(
                        t.get("description") or t.get("subject") or t.get("text") or ""
                    )
            thinking = "\n".join(b for b in bits if b).strip()

        tool_names: list[str] = []
        tool_calls = record.get("toolCalls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if isinstance(call, dict):
                    name = call.get("name") or call.get("displayName")
                    if name:
                        tool_names.append(name)
        tool_summary = "; ".join(tool_names) if tool_names else None

        if not text and not thinking and not tool_summary:
            return None
        return Message(
            role,
            text=text,
            thinking=thinking or None,
            tool_summary=tool_summary,
            timestamp=ts,
        )

    def _messages(self, path: Path) -> list[Message]:
        """Replay a chat file and convert its records into messages.

        Args:
            path: The chat session file to read.

        Returns:
            The converted list of messages in chronological order.
        """
        messages: list[Message] = []
        for record in self._replay(path):
            if not isinstance(record, dict):
                continue
            msg = self._convert(record)
            if msg is not None:
                messages.append(msg)
        return messages

    def load(self, session: Session) -> list[Message]:
        """Load the full transcript by replaying each chat file.

        Args:
            session: The session whose files should be read.

        Returns:
            A list of :class:`Message` objects in chronological order.
        """
        messages: list[Message] = []
        for path in session.paths:
            messages.extend(self._messages(path))
        return messages

    def _session_from_file(self, path: Path) -> Session:
        """Build a :class:`Session`, enriching creation time from the header.

        Only the first line is read. An earlier version replayed the whole
        mutation log here to populate ``message_count``, which made discovery
        O(file size x number of ``$set`` operations) for *every* session and
        violated the "cheap metadata only" contract of
        :meth:`~sx.adapters.base.JsonlFolderAdapter.discover`. The count is
        available once a session is opened.

        Args:
            path: The chat session file to describe.

        Returns:
            A populated :class:`Session`.
        """
        session = super()._session_from_file(path)
        first = read_first_line(path)
        if isinstance(first, dict) and self._is_header(first):
            session.created = parse_ts(first.get("startTime"))
        return session

    def find_orphans(self) -> list[Orphan]:
        """Find orphaned Gemini artifacts.

        A hash directory with a ``chats`` subdirectory but a missing or dangling
        ``.project_root`` marker is reported as :attr:`OrphanKind.DEAD_PROJECT`.
        Leftover ``projects.json.*.tmp`` files are reported as
        :attr:`OrphanKind.STRAY_TEMP`.

        Returns:
            A list of discovered orphans.
        """
        base = home() / ".gemini"
        orphans: list[Orphan] = []
        for sub in ("tmp", "history"):
            root = base / sub
            if not root.is_dir():
                continue
            for hash_dir in sorted(root.iterdir()):
                if not hash_dir.is_dir():
                    continue
                if not (hash_dir / "chats").is_dir():
                    continue
                marker = hash_dir / ".project_root"
                target = None
                if marker.is_file():
                    try:
                        target = marker.read_text(encoding="utf-8").strip()
                    except (OSError, FileNotFoundError):
                        target = None
                if not target:
                    reason = "missing or empty .project_root marker"
                elif not Path(target).exists():
                    reason = f"project directory no longer exists: {target}"
                else:
                    continue
                orphans.append(
                    Orphan(
                        harness=self.name,
                        kind=OrphanKind.DEAD_PROJECT,
                        paths=[hash_dir],
                        reason=reason,
                        size_bytes=dir_size(hash_dir),
                    )
                )
        for tmp in sorted(base.glob("projects.json.*.tmp")):
            orphans.append(
                Orphan(
                    harness=self.name,
                    kind=OrphanKind.STRAY_TEMP,
                    paths=[tmp],
                    reason="stray projects.json temp file",
                    size_bytes=dir_size(tmp),
                )
            )
        return orphans
