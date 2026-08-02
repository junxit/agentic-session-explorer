from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from ..model import Message, Orphan, OrphanKind, Role, Session
from ..util import dir_size, home, human_size, iter_jsonl, mount_unavailable, parse_ts
from .base import JsonlFolderAdapter

# Slash-command / local-command wrappers that can lead a Claude user message
# (e.g. ``<command-name>/model</command-name>`` or a ``<local-command-caveat>``
# block). They are stripped from title fallbacks so a session's title is the
# real first prompt rather than command plumbing.
_COMMAND_NOISE_RE = re.compile(
    r"<(command-[a-z]+|local-command-[a-z]+)>.*?</\1>", re.DOTALL
)
_STRAY_TAG_RE = re.compile(r"</?[a-z][a-z0-9-]*>")

# Session ids are UUIDs. Cascade deletion keys off this shape so a malformed or
# hand-placed session file can never sweep unrelated ~/.claude state.
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

# Auxiliary ("cascade") directories under ``~/.claude`` that may hold artifacts
# correlated with a session. They are included in :meth:`store_roots` so the
# delete guard permits cascade deletion, but on this platform none of them name
# their entries by session id, so :meth:`correlated_paths` stays conservative.
_CASCADE_DIRS = ("todos", "file-history", "shell-snapshots", "sessions")


def _truncate(text: str, limit: int) -> str:
    """Shorten ``text`` to ``limit`` characters, appending an ellipsis.

    Args:
        text: The string to shorten.
        limit: Maximum length of the returned string (excluding the ellipsis).

    Returns:
        The original string when short enough, otherwise a truncated copy
        ending in a single-character ellipsis.
    """
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _clean_title_text(text: str) -> str:
    """Strip slash-command / local-command plumbing from candidate title text.

    Removes ``<command-*>…</command-*>`` and ``<local-command-*>…</…>`` blocks
    and any stray tags, then collapses whitespace. Returns an empty string when
    nothing human-readable remains (so the caller can fall through to the next
    title candidate).

    Args:
        text: Raw user-message text.

    Returns:
        Cleaned title text, possibly empty.
    """
    cleaned = _COMMAND_NOISE_RE.sub(" ", text)
    cleaned = _STRAY_TAG_RE.sub(" ", cleaned)
    cleaned = " ".join(cleaned.split())
    return cleaned


def _decode_folder(name: str) -> str:
    """Decode an encoded Claude project folder name into a filesystem path.

    Claude encodes the project working directory by replacing every ``/`` with
    ``-`` and prefixing the result with a leading ``-``. The transformation is
    lossy because genuine dashes in a path are indistinguishable from path
    separators, so the decoded value is only an approximation.

    Args:
        name: The encoded folder name (e.g. ``-Users-alice-project``).

    Returns:
        An approximate absolute path (e.g. ``/Users/alice/project``).
    """
    stripped = name[1:] if name.startswith("-") else name
    return "/" + stripped.replace("-", "/")


class ClaudeAdapter(JsonlFolderAdapter):
    """Adapter for Claude Code chat sessions.

    Sessions live at ``~/.claude/projects/<encoded-cwd>/<sessionId>.jsonl``
    where the folder name encodes the project working directory. Several sibling
    directories under ``~/.claude`` hold correlated artifacts; the primary store
    root is always ``~/.claude/projects``.

    Attributes:
        name: Machine identifier (``"claude"``).
        display: Human-readable harness name.
    """

    name = "claude"
    display = "Claude Code"

    def store_roots(self) -> list[Path]:
        """Return the managed directories for Claude.

        The primary session store (``~/.claude/projects``) is always first.
        Existing cascade directories are appended so the deletion guard permits
        removing correlated artifacts.

        Returns:
            A list of directories, primary store first.
        """
        base = home() / ".claude"
        roots = [base / "projects"]
        for sub in _CASCADE_DIRS:
            d = base / sub
            if d.exists():
                roots.append(d)
        return roots

    def session_files(self) -> Iterator[Path]:
        """Yield only the per-session JSONL files under the projects store.

        The cascade directories are intentionally excluded so they are not
        mistaken for sessions.

        Returns:
            An iterator over ``~/.claude/projects/*/*.jsonl`` files.
        """
        projects = home() / ".claude" / "projects"
        if not projects.exists():
            return
        yield from sorted(projects.glob("*/*.jsonl"))

    def _cwd_from_file(self, path: Path) -> str | None:
        """Read the real working directory recorded inside a session file.

        Claude annotates user/assistant/system records with a ``cwd`` field.
        Early lines (mode, permission-mode, file-history-snapshot) may omit it,
        so a bounded number of lines are scanned for the first non-empty value.

        Args:
            path: The session file to inspect.

        Returns:
            The recorded working directory, or ``None`` if none was found.
        """
        for i, obj in enumerate(iter_jsonl(path)):
            if i > 50:
                break
            cwd = obj.get("cwd")
            if isinstance(cwd, str) and cwd.strip():
                return cwd
        return None

    def group_key(self, path: Path, first: dict | None) -> str | None:
        """Resolve the project path for a session file.

        The recorded ``cwd`` inside the file is preferred because it is exact;
        the lossy folder-name decoding is used only as a fallback.

        Args:
            path: The session file being inspected.
            first: The first parsed record (unused; ``cwd`` may appear later).

        Returns:
            The project working directory, or ``None`` if it cannot be derived.
        """
        cwd = self._cwd_from_file(path)
        if cwd:
            return cwd
        # The folder-name decode is lossy — a genuine hyphen and a path
        # separator are indistinguishable, and spaces encode as hyphens too, so
        # it is wrong for most real project paths. Use it only when it happens to
        # name a directory that exists; otherwise report "unknown" rather than a
        # fabricated path that would make the session look like an orphan.
        folder = path.parent.name
        if folder:
            decoded = _decode_folder(folder)
            if Path(decoded).exists():
                return decoded
        return None

    def session_id_for(self, path: Path, first: dict | None) -> str:
        """Return the session id for a file.

        The filename stem is the session UUID.

        Args:
            path: The session file being inspected.
            first: The first parsed record (unused).

        Returns:
            The session identifier.
        """
        return path.stem

    def title_for(self, path: Path, first: dict | None) -> str:
        """Derive a human-readable title for a session.

        A user-set ``custom-title`` (field ``customTitle``) wins; otherwise the
        generated ``ai-title`` (field ``aiTitle``) is used. The first real user
        message is the next fallback, and the filename stem is the last resort.
        Since the title line can be rewritten, the scan keeps the latest value.

        Args:
            path: The session file being inspected.
            first: The first parsed record (unused; lines are scanned).

        Returns:
            A title string.
        """
        custom_title: str | None = None
        ai_title: str | None = None
        first_user: str | None = None
        for i, obj in enumerate(iter_jsonl(path)):
            if i > 200:
                break
            rtype = obj.get("type")
            if rtype == "custom-title":
                value = obj.get("customTitle")
                if isinstance(value, str) and value.strip():
                    custom_title = value.strip()
            elif rtype == "ai-title":
                value = obj.get("aiTitle")
                if isinstance(value, str) and value.strip():
                    ai_title = value.strip()
            elif (
                first_user is None
                and rtype == "user"
                and obj.get("isMeta") is not True
            ):
                text = _clean_title_text(self._user_text(obj))
                if text:
                    first_user = text
        if custom_title:
            return _truncate(custom_title, 80)
        if ai_title:
            return _truncate(ai_title, 80)
        if first_user:
            return _truncate(first_user, 60)
        return path.stem

    @staticmethod
    def _user_text(obj: dict) -> str:
        """Extract plain user text from a Claude ``user`` record.

        Args:
            obj: A parsed ``user`` record.

        Returns:
            The joined user text, or an empty string if there is none.
        """
        content = (obj.get("message") or {}).get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            return "\n".join(p for p in parts if p).strip()
        return ""

    @staticmethod
    def _tool_result_text(content) -> str:
        """Render the body of a ``tool_result`` block to plain text.

        The block ``content`` may be a string or a list whose items are either
        strings or ``{"type": "text", "text": ...}`` dictionaries.

        Args:
            content: The ``tool_result`` content value.

        Returns:
            A plain-text rendering, possibly empty.
        """
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "\n".join(p for p in parts if p).strip()
        return ""

    def parse_line(self, obj: dict) -> Message | None:
        """Convert a Claude JSONL record into a :class:`Message`.

        Meta records are skipped. ``user`` records yield a user message, or a
        tool-result summary when they carry only tool output. ``assistant``
        records yield an assistant message with optional thinking text and a
        summary of any tool calls.

        Args:
            obj: A parsed JSON record.

        Returns:
            A :class:`Message`, or ``None`` if the record carries no content.
        """
        if obj.get("isMeta") is True:
            return None
        ts = parse_ts(obj.get("timestamp"))
        rtype = obj.get("type")

        if rtype == "user":
            content = (obj.get("message") or {}).get("content")
            if isinstance(content, str):
                text = content.strip()
                return Message(Role.USER, text=text, timestamp=ts) if text else None
            if isinstance(content, list):
                text_parts: list[str] = []
                tool_results: list[str] = []
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "text":
                        text_parts.append(b.get("text", ""))
                    elif b.get("type") == "tool_result":
                        tool_results.append(self._tool_result_text(b.get("content")))
                text = "\n".join(p for p in text_parts if p).strip()
                if text:
                    return Message(Role.USER, text=text, timestamp=ts)
                joined_results = "\n".join(r for r in tool_results if r).strip()
                if joined_results:
                    return Message(
                        Role.TOOL,
                        text=_truncate(joined_results, 200),
                        tool_summary="tool result",
                        timestamp=ts,
                    )
            return None

        if rtype == "assistant":
            content = (obj.get("message") or {}).get("content")
            if not isinstance(content, list):
                return None
            text_parts: list[str] = []
            thinking_parts: list[str] = []
            tools: list[str] = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "text":
                    text_parts.append(b.get("text", ""))
                elif bt == "thinking":
                    thinking_parts.append(b.get("thinking", ""))
                elif bt == "tool_use":
                    name = b.get("name")
                    if name:
                        tools.append(name)
            text = "\n".join(p for p in text_parts if p).strip()
            thinking = "\n".join(p for p in thinking_parts if p).strip()
            tool_summary = "; ".join(tools) if tools else None
            if not text and not thinking and not tool_summary:
                return None
            return Message(
                Role.ASSISTANT,
                text=text,
                thinking=thinking or None,
                tool_summary=tool_summary,
                timestamp=ts,
            )

        return None

    def correlated_paths(self, session: Session) -> list[Path]:
        """Return cascade artifacts keyed by this session's id.

        These feed permanent deletion, so matching is exact rather than a
        substring test: the id must be a UUID and an entry must either *be* that
        id or begin with it followed by a separator (``<id>.json``,
        ``<id>-agent-<x>.json``). A plain substring match would be catastrophic —
        a session file named ``a.jsonl`` yields the id ``a``, which appears in
        nearly every filename in these directories.

        Requiring UUID shape also means a hand-placed or malformed session file
        can never trigger a cascade at all.

        Args:
            session: The session to inspect.

        Returns:
            A list of matching auxiliary paths (commonly non-empty: on the
            sampled machine ``file-history/`` keys entries by session id).
        """
        sid = session.session_id
        if not sid or not _UUID_RE.match(sid):
            return []
        base = home() / ".claude"
        matches: list[Path] = []
        for sub in _CASCADE_DIRS:
            directory = base / sub
            if not directory.is_dir():
                continue
            for entry in sorted(directory.iterdir()):
                name = entry.name
                if name == sid or name.startswith(f"{sid}.") or name.startswith(f"{sid}-"):
                    matches.append(entry)
        return matches

    def _folder_cwd(self, session_files: list[Path]) -> str | None:
        """Return the project directory recorded inside a folder's sessions.

        Every session in the folder is consulted, not just the alphabetically
        first one: a single unreadable or truncated file must not decide the fate
        of the whole folder.

        Args:
            session_files: The folder's top-level session files.

        Returns:
            The recorded working directory, or ``None`` if none could be read.
        """
        for path in session_files:
            cwd = self._cwd_from_file(path)
            if cwd:
                return cwd
        return None

    def _leftover_reason(self, folder: Path) -> tuple[str, int]:
        """Describe what a folder without top-level sessions actually contains.

        Deletion is recursive, so "contains no session files" is a dangerously
        incomplete description of a folder that still holds nested transcripts,
        memory files, or tool output.

        Args:
            folder: The project folder to describe.

        Returns:
            A ``(reason, size_bytes)`` pair.
        """
        try:
            files = [p for p in folder.rglob("*") if p.is_file()]
        except OSError:
            files = []
        size = dir_size(folder)
        if not files:
            return "empty project folder", size
        nested = [p for p in files if p.suffix == ".jsonl"]
        memory = [p for p in files if p.suffix == ".md" and "memory" in p.parts]
        extras = [f"{len(files)} file(s), {human_size(size)}"]
        if nested:
            extras.insert(0, f"{len(nested)} nested transcript(s)")
        if memory:
            extras.insert(0, f"{len(memory)} memory file(s)")
        return "no top-level sessions; contains " + ", ".join(extras), size

    def find_orphans(self) -> list[Orphan]:
        """Find orphaned Claude project folders.

        A folder without top-level session files is reported as
        :attr:`OrphanKind.EMPTY`, with a reason stating exactly what it still
        contains. A folder whose recorded working directory no longer exists is
        reported as :attr:`OrphanKind.DEAD_PROJECT`.

        Two deliberate abstentions keep real data out of the deletion list:
        a folder whose working directory cannot be read is never called dead
        (the lossy folder-name decode is not trusted for this decision), and a
        directory on an unmounted volume is treated as unavailable rather than
        gone.

        Returns:
            A list of discovered orphans.
        """
        projects = home() / ".claude" / "projects"
        orphans: list[Orphan] = []
        if not projects.is_dir():
            return orphans
        for folder in sorted(projects.iterdir()):
            if not folder.is_dir():
                continue
            session_files = sorted(folder.glob("*.jsonl"))
            if not session_files:
                reason, size = self._leftover_reason(folder)
                orphans.append(
                    Orphan(
                        harness=self.name,
                        kind=OrphanKind.EMPTY,
                        paths=[folder],
                        reason=reason,
                        size_bytes=size,
                    )
                )
                continue
            cwd = self._folder_cwd(session_files)
            if cwd is None:
                continue  # unknown project directory — never assume it is dead
            if mount_unavailable(cwd):
                continue  # volume offline, not deleted
            if not Path(cwd).exists():
                orphans.append(
                    Orphan(
                        harness=self.name,
                        kind=OrphanKind.DEAD_PROJECT,
                        paths=[folder],
                        reason=f"project directory no longer exists: {cwd}",
                        size_bytes=dir_size(folder),
                    )
                )
        return orphans
