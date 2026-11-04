"""Pi coding agent sessions, and the format base Oh My Pi builds on.

Pi (CLI ``pi``, npm ``@earendil-works/pi-coding-agent``) appends one compact
JSON line per entry to ``<agent dir>/sessions/--<encoded cwd>--/<ts>_<id>.jsonl``.
The first line is a ``session`` header holding the session id, the working
directory and, for a fork, the transcript it was forked from. Every later line
is a tree entry with an ``id`` and a ``parentId``: retrying or editing an earlier
prompt starts a new branch in the same file, and the session resumes from the
entry written last.

:class:`PiFormatAdapter` implements everything that follows from that format —
discovery, the transcript tree, moving, deleting and orphan detection — behind a
small set of hooks, so Oh My Pi (a Pi fork that kept the format but moved its
store) overrides only what differs. :class:`PiAdapter` adds Pi's own store
location and folder encoding.

Every ``.jsonl`` file in a project folder is one of three things (see
:class:`TranscriptKind`): a session; a *fragment* — an empty file, or one whose
first line is a tree entry, which is what Pi's append leaves when the session
file was deleted or moved under it — offered for cleanup; or a file ``sx`` does
not recognize, which is listed as ``(unrecognized format) <name>`` and can be
browsed, but is never deleted, moved or offered for cleanup.
"""

from __future__ import annotations

import enum
import json
import os
import re
import stat
import threading
import time
from abc import abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote_to_bytes

from ..fsguard import MISSING, project_state
from ..model import (
    Branch,
    Capability,
    DeleteResult,
    Message,
    MovePlan,
    MoveResult,
    MoveUnit,
    Orphan,
    OrphanKind,
    ProjectLeftovers,
    Role,
    Session,
    Transcript,
)
from ..util import (
    dir_size,
    file_mtime,
    home,
    human_size,
    is_under,
    is_within,
    iter_jsonl,
    parse_ts,
    repoint,
    rewrite_jsonl,
    short_path,
)
from .base import EncodedFolderAdapter
from .relocation import Namespace, is_transcript, same_entry

#: Pi's agent-directory override. Pi reads it through ``normalizePath``: no
#: trimming, ``~`` and ``~/…`` expanded, a ``file://`` URL turned into its path.
AGENT_DIR_ENV = "PI_CODING_AGENT_DIR"

#: Pi's session-directory override. Sessions written there sit flat in one
#: directory rather than in per-project folders, and ``sx`` does not read them.
SESSION_DIR_ENV = "PI_CODING_AGENT_SESSION_DIR"

#: A transcript written within this many seconds is treated as in use — the
#: window :data:`sx.service.ACTIVE_WINDOW_SECONDS` applies to whole sessions.
_LIVE_WINDOW_SECONDS = 90

#: How long a Pi project folder, and everything in it, must have gone unwritten
#: before it is offered for removal. Pi creates the folder when it starts in a
#: project (and when it lists sessions for ``/resume``), long before it writes
#: the first session file there; removing it under a running Pi makes every
#: later prompt fail until ``/new`` or ``/resume``.
PI_FOLDER_IDLE_SECONDS = 24 * 3600

#: An idle check gives up — and counts the folder as in use — past this many
#: entries.
_IDLE_SCAN_LIMIT = 20_000

#: Raw markers for the only lines a title needs. Pi writes compact JSON, so they
#: match its lines exactly, and a line carrying a multi-megabyte inline image is
#: skipped without being parsed.
_SESSION_INFO_MARK = b'"type":"session_info"'
_USER_MARK = b'"role":"user"'

#: A title longer than this is cut, with an ellipsis. Pi cuts at the terminal's
#: width instead.
_TITLE_LIMIT = 80

#: Session format versions Pi has written; a header without one is version 1.
_KNOWN_VERSIONS = (1, 2, 3)

#: The tree entry types Pi 1.5.0 writes (``SessionManager.append*``). A
#: header-less file whose first value is one of them is what Pi's append leaves
#: behind when the session file is gone.
PI_ENTRY_TYPES = frozenset(
    {
        "message",
        "thinking_level_change",
        "model_change",
        "usage",
        "compaction",
        "custom",
        "session_info",
        "custom_message",
        "context_edit",
        "label",
        "branch_summary",
    }
)

#: The longest first line read to tell whether a header-less file is debris.
#: Pi's first entry after its session file vanished can be a whole tool result.
_FRAGMENT_LINE_LIMIT = 16 * 1024 * 1024

#: The title prefix of a file listed although ``sx`` does not recognize its format.
UNRECOGNIZED_TITLE = "(unrecognized format)"

#: The :attr:`~sx.model.Session.extra` key marking such a file; it holds why.
UNRECOGNIZED_KEY = "unrecognized"

#: Why a file sx does not recognize never travels with a move.
_UNRECOGNIZED_STAYS = "sx does not recognize this file's format, so it never moves it (refused)"

#: Why nothing is deleted, moved or rewritten through a project folder that is a
#: symbolic link: whatever it points at lies outside the store's own folders.
_LINKED_FOLDER = (
    "its project folder is a symbolic link; sx never deletes, moves or rewrites anything "
    "through one (refused)"
)

#: What only Oh My Pi creates in an agent directory: its databases and its
#: terminal and blob stores. Pi's own files there are JSON (``auth.json``,
#: ``models.json``, ``settings.json``), ``pi-debug.log`` and plain directories.
OMP_AGENT_MARKERS = ("agent.db", "history.db", "models.db", "terminal-sessions", "blobs")

#: Oh My Pi's display name, which Pi yields a store to.
_OMP = "Oh My Pi"

#: The sessions-directory sample :func:`omp_markers` takes: this many folders,
#: newest first; the first line of at most this many transcripts in all, and of
#: at most this many per folder, newest first.
_MARKER_FOLDERS = 256
_MARKER_READS = 32
_MARKER_READS_PER_FOLDER = 2

#: How much of a transcript is read to see whether it starts with a title slot
#: (Oh My Pi's is 256 bytes).
_SLOT_PROBE_BYTES = 512

#: The characters JavaScript's ``String.prototype.trim`` removes.
_JS_SPACE = (
    " \t\n\v\f\r\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006"
    "\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
)

#: What Pi's session list turns into spaces in a title.
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")

#: A percent sign that does not start an escape, or an escaped ``/``: Node's
#: ``fileURLToPath`` throws on both.
_BAD_ESCAPE_RE = re.compile(r"%(?![0-9A-Fa-f]{2})|%2[fF]")

#: What URL parsing drops from the end of a URL: C0 controls and the space.
_URL_TRAILER = "".join(chr(code) for code in range(0x21))


class TranscriptKind(enum.Enum):
    """What a ``.jsonl`` file in a project folder holds (:meth:`PiFormatAdapter.transcript_kind`).

    Attributes:
        SESSION: A session: it starts with a session header the adapter reads.
        FRAGMENT: Debris the harness itself leaves — an empty file, or one
            whose first line is a tree entry, which is what Pi's append writes
            when the session file was deleted or moved under it. Offered for
            cleanup.
        UNRECOGNIZED: Anything else, a file that cannot be read included. It is
            listed and can be browsed as far as it parses, but it is never
            deleted, moved or offered for cleanup.
        MISSING: Nothing is there.
    """

    SESSION = "session"
    FRAGMENT = "fragment"
    UNRECOGNIZED = "unrecognized"
    MISSING = "missing"


# --- small helpers -----------------------------------------------------------


def _unknown_version(header: dict) -> str | None:
    """Return why ``header`` must not be deleted or rewritten, if its version is unknown.

    Args:
        header: A session header.

    Returns:
        A refusal reason when the header records a format version Pi never
        wrote, else ``None``.
    """
    version = header.get("version")
    if version is None or (not isinstance(version, bool) and version in _KNOWN_VERSIONS):
        return None
    return f"records session format version {version!r}, which sx does not know (refused)"


def _unrecognized_refusal(session: Session) -> str | None:
    """Return the refusal for a session listed as a file ``sx`` does not recognize.

    Args:
        session: A session about to be deleted or moved.

    Returns:
        The refusal reason, or ``None`` for a session ``sx`` recognized.
    """
    why = session.extra.get(UNRECOGNIZED_KEY)
    if not why:
        return None
    return (
        f"sx does not recognize this file's format ({why}), so it never deletes or moves it "
        "(refused)"
    )


def _truncate(text: str, limit: int) -> str:
    """Shorten ``text`` to ``limit`` characters, appending an ellipsis.

    Args:
        text: The string to shorten.
        limit: Maximum length before the ellipsis.

    Returns:
        The stripped text, cut at ``limit`` characters and ending in ``…`` when
        it was longer.
    """
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _one_line(text: str) -> str:
    """Collapse every run of whitespace, newlines included, into one space.

    Args:
        text: The text to flatten.

    Returns:
        The text on a single line.
    """
    return " ".join(text.split())


def _js_trim(text: str) -> str:
    """Trim ``text`` the way JavaScript's ``trim`` does.

    Args:
        text: The string to trim.

    Returns:
        The string without leading or trailing JavaScript whitespace.
    """
    return text.strip(_JS_SPACE)


def _node_normpath(path: str) -> str:
    """Normalize a path lexically, as Node's ``path.resolve`` does on POSIX.

    ``os.path.normpath`` keeps a leading ``//`` (POSIX leaves its meaning to the
    implementation); Node collapses it like any other repeated separator.

    Args:
        path: An absolute path.

    Returns:
        The normalized path. Symlinks are not resolved — Pi never resolves them
        either.
    """
    normalized = os.path.normpath(path)
    if normalized.startswith("//"):
        normalized = "/" + normalized.lstrip("/")
    return normalized


def _node_join(*parts: str) -> str:
    """Join path segments the way Node's ``path.join`` does on POSIX.

    Unlike :func:`os.path.join`, a later absolute segment does not discard the
    ones before it: ``join("/home/me", "/x")`` is ``/home/me/x``.

    Args:
        *parts: The segments; empty ones are skipped.

    Returns:
        The normalized joined path, or ``.`` when every segment is empty.
    """
    joined = "/".join(part for part in parts if part)
    return _node_normpath(joined) if joined else "."


def _file_url_path(value: str) -> str | None:
    """Return the path a ``file://`` URL names, as Node's ``fileURLToPath`` does on POSIX.

    Only a URL whose meaning is plain is read: an empty or ``localhost`` host,
    and a path with no query, fragment, backslash or control character, no
    escaped ``/`` and only escapes that decode to UTF-8. Trailing spaces and
    control characters are dropped, as URL parsing drops them.

    Args:
        value: A value that starts with ``file://``.

    Returns:
        The decoded path, not yet normalized; ``None`` when ``sx`` cannot be sure
        what Node makes of the URL.
    """
    rest = value[len("file://") :].rstrip(_URL_TRAILER)
    slash = rest.find("/")
    if slash < 0:
        return None
    host, path = rest[:slash], rest[slash:]
    if host.lower() not in ("", "localhost"):
        return None
    if any(char in "?#\\" or ord(char) < 0x20 or char == "\x7f" for char in path):
        return None
    if _BAD_ESCAPE_RE.search(path):
        return None
    try:
        decoded = unquote_to_bytes(path).decode("utf-8")
    except UnicodeDecodeError:
        return None
    return None if "\x00" in decoded else decoded


def _same_place(a: Path, b: Path) -> bool:
    """Return True if two paths name the same location, lexically or resolved.

    Args:
        a: One path.
        b: The other.

    Returns:
        True when they are equal after normalization or after resolving
        symlinks.
    """
    if _node_normpath(str(a)) == _node_normpath(str(b)):
        return True
    try:
        return a.resolve() == b.resolve()
    except (OSError, RuntimeError):
        return False


def _realpath(value: str | Path) -> str:
    """Return ``value`` with symlinks resolved, or unchanged if that fails.

    Args:
        value: A path, which need not exist.

    Returns:
        The resolved path as a string.
    """
    try:
        return os.path.realpath(value)
    except (OSError, ValueError):
        return str(value)


def _written_within(path: Path, seconds: float, now: float) -> bool:
    """Return True if ``path`` was modified within ``seconds`` of ``now``.

    A file that cannot be stat'ed counts as recently written: unknown is
    treated as in use, the fail-safe direction.

    Args:
        path: The file to test.
        seconds: The window.
        now: The current time, as from :func:`time.time`.

    Returns:
        True if the file is, or may be, in active use.
    """
    try:
        return now - path.stat().st_mtime < seconds
    except OSError:
        return True


def _newest_write(directory: Path, *, contents_only: bool = False) -> float | None:
    """Return when a directory, or anything under it, was last modified.

    Every entry is examined with ``lstat``; symbolic links are not followed.

    Args:
        directory: The directory.
        contents_only: Leave the directory's own modification time out — it
            changes whenever an entry is removed from it.

    Returns:
        The newest modification time, as from :func:`time.time` (``0.0`` for
        an empty directory when ``contents_only``); ``None`` when anything
        cannot be examined, or the tree holds more than 20,000 entries.
    """
    try:
        info = os.lstat(directory)
    except OSError:
        return None
    if not stat.S_ISDIR(info.st_mode):
        return None
    newest = 0.0 if contents_only else info.st_mtime
    pending = [os.fspath(directory)]
    seen = 0
    while pending:
        try:
            with os.scandir(pending.pop()) as entries:
                children = list(entries)
        except OSError:
            return None
        for child in children:
            seen += 1
            if seen > _IDLE_SCAN_LIMIT:
                return None
            try:
                child_info = child.stat(follow_symlinks=False)
            except OSError:
                return None
            newest = max(newest, child_info.st_mtime)
            if stat.S_ISDIR(child_info.st_mode):
                pending.append(child.path)
    return newest


def _js_truthy(value: object) -> bool:
    """Return whether a parsed JSON value is truthy by JavaScript's rules.

    Pi's readers branch on JavaScript truthiness, which differs from Python's
    for ``[]`` and ``{}`` (truthy in JavaScript) — so it is spelled out here.

    Args:
        value: A value produced by :func:`json.loads`.

    Returns:
        False for ``null``, ``false``, ``0`` and ``""``; True otherwise.
    """
    if value is None or value is False:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value != ""
    return True


def _parse_line(raw: bytes) -> object:
    """Parse one raw JSONL line.

    Args:
        raw: The line as read in binary mode.

    Returns:
        The parsed value, or ``None`` for a blank or malformed line.
    """
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return None


def _read_first_value(path: Path) -> object:
    """Return the first JSON value in a JSONL file, skipping what Pi skips.

    Blank and malformed lines are passed over, and so are values JavaScript
    treats as false, exactly as Pi's header reader does. Any other first value —
    an array, a number, an object that is not a header — is returned, so the
    caller rejects the file instead of reading past it.

    Args:
        path: A ``.jsonl`` file.

    Returns:
        That value, or ``None`` when there is none.

    Raises:
        OSError: The file cannot be read.
    """
    with path.open("rb") as fh:
        for raw in fh:
            value = _parse_line(raw)
            if _js_truthy(value):
                return value
    return None


def _first_value(path: Path) -> object:
    """Return :func:`_read_first_value`, or ``None`` when the file cannot be read.

    Args:
        path: A ``.jsonl`` file.

    Returns:
        The first value, or ``None``.
    """
    try:
        return _read_first_value(path)
    except OSError:
        return None


def _header_cwd(header: dict) -> str | None:
    """Return the working directory a session header records, if any.

    Args:
        header: A session header.

    Returns:
        The ``cwd`` value when it is a non-empty string, else ``None``.
    """
    cwd = header.get("cwd")
    return cwd if isinstance(cwd, str) and cwd else None


def _folder_transcripts(folder: Path) -> list[Path]:
    """Return the transcripts directly inside one project folder.

    Args:
        folder: A folder directly under the sessions directory.

    Returns:
        Its ``*.jsonl`` files, excluding names that start with ``.``, sorted.
    """
    try:
        return sorted(
            path
            for path in folder.iterdir()
            if path.name.endswith(".jsonl") and not path.name.startswith(".") and path.is_file()
        )
    except OSError:
        return []


def _folder_contents(folder: Path) -> tuple[str, int]:
    """Describe what a project folder without transcripts still contains.

    Deletion is recursive, so "no sessions" alone would understate a folder that
    still holds files.

    Args:
        folder: The folder to describe.

    Returns:
        A ``(reason, size_bytes)`` pair.
    """
    try:
        files = [path for path in folder.rglob("*") if path.is_file()]
    except OSError:
        files = []
    size = dir_size(folder)
    if not files:
        return "empty project folder", size
    detail = f"{len(files)} file(s), {human_size(size)}"
    hidden = sum(1 for path in files if path.name.endswith(".jsonl"))
    if hidden:
        detail = f"{hidden} nested or hidden transcript(s), {detail}"
    return f"no sessions; contains {detail}", size


def _content_text(content: object, *, joiner: str = "\n", images: bool = True) -> str:
    """Flatten Pi message content — a string, or text and image blocks — to text.

    Args:
        content: A message's ``content`` value, of any shape.
        joiner: Placed between blocks.
        images: Render each image block as ``[image]`` rather than dropping it.

    Returns:
        The text; empty when there is none or the shape is not recognized.
    """
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        elif kind == "image" and images:
            parts.append("[image]")
    return joiner.join(parts)


def _title_line(message: object) -> str:
    """Return a title from a user message: its first line, flattened and cut.

    Args:
        message: The ``message`` field of a ``message`` entry.

    Returns:
        Up to 60 characters of the first non-blank line, or an empty string
        when the message is not a user message or holds no text.
    """
    if not isinstance(message, dict) or message.get("role") != "user":
        return ""
    text = _content_text(message.get("content"), joiner=" ", images=False)
    for line in text.splitlines():
        if line.strip():
            return _truncate(_one_line(line), 60)
    return ""


def _first_user_text(message: object) -> str:
    """Return a user message's text as Pi's session list reads it for a title.

    That is Pi's ``extractTextContent``: a string content as it is, or the text
    of every text block joined with a space, images left out.

    Args:
        message: The ``message`` field of a ``message`` entry.

    Returns:
        The text, unmodified; empty when the message is not a user message or
        holds no text.
    """
    if not isinstance(message, dict) or message.get("role") != "user":
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return " ".join(
        block["text"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )


def _title_text(text: str) -> str:
    """Clean a title as Pi's session list does, then cut it for ``sx``'s list.

    Args:
        text: A session name or a first message.

    Returns:
        The text with every control character, newlines included, replaced by a
        space, trimmed, and cut at 80 characters.
    """
    return _truncate(_js_trim(_CONTROL_RE.sub(" ", text)), _TITLE_LIMIT)


def _stop_notice(message: dict, tool_calls: bool) -> str | None:
    """Return the notice Pi shows after an assistant turn that did not finish.

    Pi's ``AssistantMessageComponent`` shows it after any partial content: a
    truncation notice for ``stopReason: "length"``; for an aborted turn its
    ``errorMessage`` — unless that is absent or the generic "Request was
    aborted" — else "Operation aborted"; for a failed one ``Error:`` and the
    message, else "Unknown error". When the turn made tool calls, Pi shows the
    abort or the error on each tool call instead; ``sx``, which has no tool
    rows, shows it here — for an abort, "Operation aborted", as Pi's tool rows
    do when a session is reopened.

    Args:
        message: The ``message`` field of an assistant ``message`` entry.
        tool_calls: Whether the turn made tool calls.

    Returns:
        The notice, or ``None`` for a turn that finished.
    """
    reason = message.get("stopReason")
    error = message.get("errorMessage")
    detail = error.strip() if isinstance(error, str) else ""
    if reason == "length":
        return "Response was truncated before completion."
    if reason == "aborted":
        if detail and error != "Request was aborted" and not tool_calls:
            return detail
        return "Operation aborted"
    if reason == "error":
        return f"Error: {detail or 'Unknown error'}"
    return None


def _assistant_message(message: dict, timestamp: datetime | None) -> Message | None:
    """Render an assistant message: text, thinking, tool calls, and any stop notice.

    Args:
        message: The ``message`` field of a ``message`` entry.
        timestamp: The entry's timestamp.

    Returns:
        A :class:`Message`, or ``None`` when the message carries nothing to show.
        A turn that was cut short, aborted or failed always shows, with the
        notice Pi shows (see :func:`_stop_notice`) after its partial content.
    """
    content = message.get("content")
    texts: list[str] = []
    thoughts: list[str] = []
    tools: list[str] = []
    called = False
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                value = block.get("text")
                if isinstance(value, str):
                    texts.append(value)
            elif kind == "thinking":
                value = block.get("thinking")
                if isinstance(value, str):
                    thoughts.append(value)
            elif kind == "toolCall":
                called = True
                name = block.get("name")
                if isinstance(name, str) and name:
                    tools.append(name)
    text = "\n".join(part.strip() for part in texts if part.strip())
    notice = _stop_notice(message, called)
    if notice is not None:
        text = f"{text}\n\n{notice}" if text else notice
    thinking = "\n".join(part.strip() for part in thoughts if part.strip())
    if not (text or thinking or tools):
        return None
    return Message(
        Role.ASSISTANT,
        text=text,
        timestamp=timestamp,
        tool_summary="; ".join(tools) or None,
        thinking=thinking or None,
    )


def _displayed(display: object, content: object, timestamp: datetime | None) -> Message | None:
    """Render extension-injected content, but only if it is shown in Pi too.

    Args:
        display: The record's ``display`` flag.
        content: Its ``content`` value.
        timestamp: The entry's timestamp.

    Returns:
        An :attr:`Role.OTHER` message, or ``None`` when hidden or empty.
    """
    if not _js_truthy(display):
        return None
    text = _content_text(content)
    return Message(Role.OTHER, text=text, timestamp=timestamp) if text else None


# --- the transcript tree -------------------------------------------------------


def _parent_of(entry: dict) -> str | None:
    """Return an entry's ``parentId`` when it is a non-empty string, else ``None``."""
    parent = entry.get("parentId")
    return parent if isinstance(parent, str) and parent else None


def _walk_up(
    start: str, by_id: dict[str, dict], stop: dict[str, int]
) -> tuple[list[str], str | None]:
    """Follow ``parentId`` links from ``start`` until reaching an entry in ``stop``.

    Cycle-safe: an id already visited ends the walk, as does a missing,
    ``null`` or empty parent.

    Args:
        start: The id to start from.
        by_id: Every entry, by id.
        stop: Ids that end the walk without being included.

    Returns:
        ``(segment, fork)``: the ids walked, root-most first, and the id in
        ``stop`` that ended the walk — ``None`` when it ran out of parents.
    """
    segment: list[str] = []
    seen: set[str] = set()
    current: str | None = start
    while current is not None and current in by_id and current not in seen:
        if current in stop:
            segment.reverse()
            return segment, current
        seen.add(current)
        segment.append(current)
        current = _parent_of(by_id[current])
    segment.reverse()
    return segment, None


def _ancestors(ids: list[str], by_id: dict[str, dict]) -> set[str]:
    """Return every entry that is an ancestor of one of ``ids``.

    Each entry is visited once: a walk stops at an entry already found, whose
    own ancestors were found with it, and so a cycle ends a walk too.

    Args:
        ids: Entry ids.
        by_id: Every entry, by id.

    Returns:
        The ids of their ancestors.
    """
    found: set[str] = set()
    for entry_id in ids:
        current = _parent_of(by_id[entry_id])
        while current is not None and current in by_id and current not in found:
            found.add(current)
            current = _parent_of(by_id[current])
    return found


def _anchors(
    by_id: dict[str, dict], rendered: dict[str, Message | None]
) -> Callable[[str], str | None]:
    """Return a lookup of the nearest entry at or above an entry that shows something.

    Entries that show nothing — labels, usage, a thinking-level change — are
    looked through, so they never decide where a branch continues from.
    Answers are remembered, so every entry is walked past once.

    Args:
        by_id: Every entry, by id.
        rendered: Each entry's message, or ``None`` for one that shows nothing.

    Returns:
        A function from an entry id to that entry, when it shows a message, or
        to its nearest ancestor that does; ``None`` when there is none (the
        walk ran out of parents, or into a cycle).
    """
    memo: dict[str, str | None] = {}

    def anchor(entry_id: str) -> str | None:
        """Return the nearest entry at or above ``entry_id`` that shows a message."""
        chain: list[str] = []
        walked: set[str] = set()
        current: str | None = entry_id
        result: str | None = None
        while current is not None and current in by_id:
            if current in memo:
                result = memo[current]
                break
            if rendered[current] is not None:
                result = current
                break
            if current in walked:
                break
            walked.add(current)
            chain.append(current)
            current = _parent_of(by_id[current])
        for item in chain:
            memo[item] = result
        return result

    return anchor


def _running_counts(ids: list[str], rendered: dict[str, Message | None]) -> dict[str, int]:
    """Count the messages shown along a run of entries.

    Args:
        ids: Entry ids, in display order.
        rendered: Each entry's message, or ``None`` for one that shows nothing.

    Returns:
        For each id, how many messages are shown up to and including it.
    """
    counts: dict[str, int] = {}
    total = 0
    for entry_id in ids:
        if rendered[entry_id] is not None:
            total += 1
        counts[entry_id] = total
    return counts


def _fork_label(fork: str | None, owner: dict[str, int], position: dict[str, int]) -> str:
    """Describe where a branch forks, for its heading.

    Args:
        fork: The entry the branch continues from, or ``None``.
        owner: Branch number per placed entry; ``0`` is the current branch.
        position: Messages up to and including each placed entry, counted
            within its own branch.

    Returns:
        A phrase such as ``forks from the current branch after message 3``.
    """
    if fork is None:
        return "a separate root"
    number = owner[fork]
    where = "the current branch" if number == 0 else f"other branch {number}"
    count = position[fork]
    if count:
        return f"forks from {where} after message {count}"
    return f"forks from the start of {where}"


def _jump_label(anchor: str | None, linked: bool, counts: dict[str, int], current: bool) -> str:
    """Describe a jump in the file-order view, where the next message changes branch.

    Args:
        anchor: The nearest entry above the message that shows something, or
            ``None``.
        linked: Whether the message's parent is an entry in the file.
        counts: Messages up to and including each entry, in file order.
        current: Whether the message is on the current branch.

    Returns:
        A marker text such as ``branch — continues after message 5 (current branch)``.
    """
    which = "current branch" if current else "other branch"
    if anchor is not None:
        return f"branch — continues after message {counts[anchor]} ({which})"
    if linked:
        return f"branch — continues from the start ({which})"
    return f"branch — a separate root ({which})"


# --- rewriting headers ------------------------------------------------------------


def _header_transform(update: Callable[[dict], dict | None]) -> Callable[[dict], dict | None]:
    """Wrap ``update`` so a rewrite applies it to the session header alone.

    The header is the first record whose ``type`` is ``session`` — the first
    line of a Pi file, the second of an Oh My Pi file (after its title slot).
    Every other record is left alone, so its line is copied byte for byte.

    Args:
        update: Maps the header to a new replacement mapping, or to ``None`` for
            no change; it must not edit the header it is given.

    Returns:
        A transform for :func:`sx.util.rewrite_jsonl`.
    """
    seen = False

    def transform(obj: dict) -> dict | None:
        """Return the replacement header, or ``None`` for every other record."""
        nonlocal seen
        if seen or obj.get("type") != "session":
            return None
        seen = True
        return update(obj)

    return transform


def _relink(expected: str, target: str) -> Callable[[dict], dict | None]:
    """Build a header update that re-points ``parentSession``.

    Args:
        expected: The value the link must still have; anything else is left.
        target: The new value.

    Returns:
        A header update for :func:`_header_transform`.
    """

    def update(header: dict) -> dict | None:
        """Return the header with its fork link re-pointed, if it still matches."""
        if header.get("parentSession") != expected:
            return None
        return {**header, "parentSession": target}

    return update


def _half_moved(src: Path, dst: Path) -> bool:
    """Return True if ``src`` is a transcript a move cut short has already linked at ``dst``.

    A file is moved by linking it at its destination, then unlinking its old
    name (:func:`~sx.adapters.relocation.rename_no_clobber`). A crash between
    the two leaves one file under both names; finishing the move unlinks the
    old name, and loses nothing — the destination is the same file. A case-only
    rename on a case-insensitive file system, where both spellings name one
    entry, is not this (:func:`~sx.adapters.relocation.same_entry`).

    Args:
        src: A transcript's path before the move.
        dst: Where the move puts it.

    Returns:
        Whether both names are hard links to one regular file.
    """
    if not is_transcript(src) or src.name != dst.name:
        return False
    try:
        before, after = os.lstat(src), os.lstat(dst)
    except (OSError, ValueError):
        return False
    return (
        stat.S_ISREG(before.st_mode)
        and os.path.samestat(before, after)
        and not same_entry(src, dst)
    )


def _hand_over(exc: BaseException, partial: MoveResult) -> None:
    """Hang ``partial`` on ``exc`` as its ``partial_result``, for the op-log.

    The move service absorbs it into the record of the move ``exc`` cut short.

    Args:
        exc: What cut the move short.
        partial: What the move did on its way out that no unit's record holds.
    """
    try:
        exc.partial_result = partial
    except (AttributeError, TypeError):  # an exception that takes no attributes
        pass


def _final_locations(relocated: dict[Path, Path]) -> dict[Path, Path]:
    """Map each transcript's path before a move to where it actually ended up.

    ``relocated`` (:attr:`~sx.model.MoveResult.relocated`) lists each unit's
    relocations in the order the units ran. A transcript one unit moved and a
    later unit moved again is followed to its last location, and a path one
    unit vacated and another then reused is not mistaken for the transcript
    that left it.

    Args:
        relocated: Path before each unit ran -> path after it, in run order.

    Returns:
        Path before the move -> path after it.
    """
    final: dict[Path, Path] = {}
    origin: dict[Path, Path] = {}
    for before, after in relocated.items():
        start = origin.pop(before, before)
        final[start] = after
        origin[after] = start
    return final


# --- Pi's store location -----------------------------------------------------


def encode_session_folder(project: str | Path) -> str:
    r"""Return the folder name Pi keeps ``project``'s sessions in.

    This is Pi's ``getDefaultSessionDirPath``: the directory is normalized
    lexically (symlinks are not resolved), one leading separator is dropped,
    every ``/``, ``\`` and ``:`` becomes ``-``, and the result is wrapped in
    ``--``. Dots, spaces and underscores survive — unlike Claude Code's
    encoding — but the scheme is still lossy: ``/a-b`` and ``/a/b`` share a
    folder, so a folder name is never decoded.

    Args:
        project: An absolute project directory.

    Returns:
        The folder name, e.g. ``--Users-me-src-app--``.
    """
    text = _node_normpath(str(project))
    if text[:1] in ("/", "\\"):
        text = text[1:]
    for separator in ("/", "\\", ":"):
        text = text.replace(separator, "-")
    return f"--{text}--"


def agent_dir_override(base: Path) -> tuple[Path | None, str | None]:
    """Return the ``PI_CODING_AGENT_DIR`` override, or why it is ignored.

    Pi reads the value through ``normalizePath`` with its defaults
    (``utils/paths.js``): nothing is trimmed; ``~`` alone is the home directory
    and ``~/rest`` is Node's ``path.join(home, rest)``, so ``~//x`` is
    ``<home>/x``; a ``file://`` URL becomes the path it names. Any other value is
    used as given — a relative one, a value with leading spaces included,
    against whichever directory Pi is started in. ``sx`` follows an absolute
    result, but ignores a relative one rather than resolve it against its own
    working directory, which would let a store root — the delete allowlist —
    follow ``cwd``; a ``file://`` URL it cannot read exactly as Node does is
    ignored too. The home directory and a filesystem root are refused as well:
    the sessions directory beneath them (``~/sessions``, ``/sessions``) could be
    an ordinary user directory.

    Args:
        base: The home directory. Passed in rather than resolved here, so a
            test's monkeypatch of the caller's ``home`` is honored.

    Returns:
        ``(override, None)`` when the variable is usable, ``(None, reason)``
        when it is set but ignored, and ``(None, None)`` when it is unset or
        empty — Pi treats an empty value as unset.
    """
    value = os.environ.get(AGENT_DIR_ENV, "")
    if not value:
        return None, None
    if value == "~":
        value = str(base)
    elif value.startswith("~/"):
        value = _node_join(str(base), value[2:])
    elif value.startswith("file://"):
        path = _file_url_path(value)
        if path is None:
            return None, "a file: URL sx cannot read the way Pi does"
        value = path
    if not os.path.isabs(value):
        return None, "a relative path, which Pi resolves against wherever it is started"
    candidate = Path(_node_normpath(value))
    if candidate == Path(candidate.anchor):
        return None, "a filesystem root"
    if _same_place(candidate, base):
        return None, "the home directory"
    return candidate, None


# --- what only Oh My Pi writes -------------------------------------------------


def _omp_backup_owner(name: str) -> str | None:
    """Return the transcript an Oh My Pi crash backup restores, if ``name`` is one.

    Oh My Pi names them ``<transcript>.jsonl.<id>.bak``, ``<id>`` holding no dot.

    Args:
        name: A file name.

    Returns:
        The transcript's file name, or ``None``.
    """
    if not name.endswith(".bak"):
        return None
    stem = name[: -len(".bak")]
    dot = stem.rfind(".")
    if dot <= 0 or not stem[:dot].endswith(".jsonl"):
        return None
    return stem[:dot]


def _starts_with_title_slot(path: Path) -> bool:
    """Return True if a transcript's first line is Oh My Pi's title slot.

    That is a record ``{"type":"title","v":1,…}``; Pi never writes one.

    Args:
        path: A transcript.

    Returns:
        Whether it starts with a title slot; False when it cannot be read.
    """
    try:
        if not stat.S_ISREG(os.stat(path).st_mode):
            return False  # a pipe would block the read
        with path.open("rb") as fh:
            head = fh.read(_SLOT_PROBE_BYTES)
    except (OSError, ValueError):
        return False
    value = _parse_line(head.split(b"\n", 1)[0])
    if not isinstance(value, dict) or value.get("type") != "title":
        return False
    version = value.get("v")
    return isinstance(version, int) and not isinstance(version, bool) and version == 1


def _omp_session_evidence(sessions_dir: Path) -> str | None:
    """Return the first piece of Oh My Pi evidence a sample of a sessions directory shows.

    Args:
        sessions_dir: The sessions directory.

    Returns:
        A short description naming the file, or ``None``.
    """
    folders: list[tuple[float, str]] = []
    try:
        with os.scandir(sessions_dir) as entries:
            for entry in entries:
                if entry.name.startswith("."):
                    continue
                try:
                    info = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if stat.S_ISDIR(info.st_mode):
                    folders.append((info.st_mtime, entry.name))
    except OSError:
        return None
    folders.sort(reverse=True)
    reads = 0
    for _, name in folders[:_MARKER_FOLDERS]:
        folder = sessions_dir / name
        try:
            names = set(os.listdir(folder))
        except OSError:
            continue
        transcripts = sorted(
            (entry for entry in names if entry.endswith(".jsonl") and not entry.startswith(".")),
            reverse=True,
        )
        for transcript in transcripts:
            stem = transcript[: -len(".jsonl")]
            if stem not in names:
                continue
            try:
                artifacts = stat.S_ISDIR(os.lstat(folder / stem).st_mode)
            except OSError:
                artifacts = False
            if artifacts:
                return f"{name}/{stem}/ (an Oh My Pi artifacts directory)"
        # A backup counts only as a transcript does, by its first line: a name
        # alone — a backup's or a lock file's — is what a user's own copy
        # (``cp S.jsonl S.jsonl.$(date +%s).bak``) can look like too.
        backups = sorted((entry for entry in names if _omp_backup_owner(entry)), reverse=True)
        for candidate in [*transcripts, *backups][:_MARKER_READS_PER_FOLDER]:
            if reads >= _MARKER_READS:
                break
            reads += 1
            if _starts_with_title_slot(folder / candidate):
                return f"{name}/{candidate} (starts with Oh My Pi's title slot)"
    return None


def omp_markers(agent_dir: Path, sessions_dir: Path) -> list[str]:
    """Return what in a Pi-style store only Oh My Pi would have written.

    Pi and Oh My Pi can resolve to the same agent directory — both read
    ``PI_CODING_AGENT_DIR`` — and Oh My Pi's first run in any working directory
    renames every Pi-style folder in a shared sessions directory to its own
    scheme, so a store carrying Oh My Pi's marks is effectively Oh My Pi's.
    Pi's adapter leaves such a store alone. None of this is ever written by Pi.

    The agent directory is checked for :data:`OMP_AGENT_MARKERS` and for
    ``omp-*.log`` files. Only when it shows none is the sessions directory
    sampled, cheaply: its 256 newest folders are listed, and the first line of
    at most 32 of their newest transcripts (two per folder, crash backups
    after transcripts) is read. A folder shows Oh My Pi's hand when it holds a
    transcript or a crash backup (``<name>.jsonl.<id>.bak``) whose first line
    is Oh My Pi's title slot (``{"type":"title","v":1,…}``), or an artifacts
    directory named for a transcript beside it. A file's name alone never
    decides — not a backup's, not a lock file's (``<name>.jsonl.lock``): a
    user's own copy of a session can be named like one. Nor does a folder's
    name: Oh My Pi's buckets for paths outside home are named as Pi's are. The
    sample stops at the first piece of evidence.

    Args:
        agent_dir: The agent directory, e.g. ``~/.pi/agent``.
        sessions_dir: The sessions directory the store's transcripts are in.

    Returns:
        One short description per piece of evidence — every agent-directory
        marker found, else at most one file from the sessions directory —
        such as ``"agent.db"``. Empty when there is none. Never raises.
    """
    found: list[str] = []
    try:
        for name in OMP_AGENT_MARKERS:
            if os.path.lexists(agent_dir / name):
                found.append(name)
        try:
            with os.scandir(agent_dir) as entries:
                logs = sorted(
                    entry.name
                    for entry in entries
                    if entry.name.startswith("omp-") and entry.name.endswith(".log")
                )
        except OSError:
            logs = []
        if logs:
            more = len(logs) - 1
            found.append(f"{logs[0]} and {more} more log(s)" if more else logs[0])
        if not found:
            evidence = _omp_session_evidence(sessions_dir)
            if evidence is not None:
                found.append(evidence)
    except (OSError, ValueError):  # an embedded NUL, say: nothing more can be read
        pass
    return found


# --- the format ----------------------------------------------------------------


class PiFormatAdapter(EncodedFolderAdapter):
    """Base for harnesses that write Pi's session format: Pi and Oh My Pi.

    Subclasses implement :meth:`sessions_dir` and :meth:`folder_for`. The other
    hooks — :meth:`read_header`, :meth:`is_fragment`, :meth:`is_entry`,
    :meth:`title_for_file` (with :meth:`name_title` and :meth:`prompt_title`),
    :meth:`current_leaf`, :meth:`entry_message`, :meth:`header_update`,
    :meth:`destructive_refusal` and :meth:`busy_transcript` — default to the
    format's behavior and are overridden only where a harness's files, or the
    way it lists them, differ.

    Attributes:
        capabilities: Browse, orphans, delete and move.
        supports_branches: True — one file holds every branch of a session.
        folder_idle_seconds: How long a project folder, and everything in it,
            must have gone unwritten before it is offered for removal — as an
            empty folder, or with a project's last session. ``0`` (the default)
            adds nothing to the store-wide quiet window; Pi uses
            :data:`PI_FOLDER_IDLE_SECONDS`.
        fragment_entry_types: The entry types a header-less file may start with
            and still count as a fragment.
        fragment_line_limit: The longest first line, in bytes, that can make a
            header-less file a fragment; a longer one is not read.
        empty_folder_note: Added to the reason an empty project folder is
            offered for removal, when the harness keeps using the folder;
            ``None`` adds nothing.
    """

    capabilities = Capability.BROWSE | Capability.ORPHANS | Capability.DELETE | Capability.MOVE
    supports_branches = True
    folder_idle_seconds: int = 0
    fragment_entry_types: frozenset[str] = PI_ENTRY_TYPES
    fragment_line_limit: int = _FRAGMENT_LINE_LIMIT
    empty_folder_note: str | None = None

    # --- store layout ----------------------------------------------------

    @abstractmethod
    def sessions_dir(self) -> Path:
        """Return the resolved directory that holds the per-project folders."""
        raise NotImplementedError

    def store_roots(self) -> list[Path]:
        """Return the sessions directory — the only place this adapter writes.

        Returns:
            ``[sessions_dir()]``. The harness's configuration beside it
            (settings, trust decisions, credentials) is deliberately outside the
            delete and move allowlist.
        """
        return [self.sessions_dir()]

    def sessions_root(self) -> Path:
        """Return the directory holding the per-project folders."""
        return self.sessions_dir()

    def available(self) -> bool:
        """Return True when the sessions directory exists.

        Not the agent directory: other tools drop files into it (an extension
        under ``extensions/``, for instance) on machines where the harness has
        never been run.

        Returns:
            True if the harness has written sessions here, and this adapter does
            not leave the directory to another harness.
        """
        if self._yields_to() is not None:
            return False
        return self.sessions_dir().is_dir()

    def _yields_to(self) -> str | None:
        """Return the harness this adapter leaves its sessions directory to, if any.

        A directory one harness yields is not read by it at all: it would
        otherwise mistake the other's files for its own. An adapter that yields
        lists, plans and deletes nothing there.

        Returns:
            That harness's display name, or ``None``. Default: ``None``.
        """
        return None

    def store_owner(self) -> str | None:
        """Return the harness this adapter has handed its sessions directory to, if any.

        Shown by ``sx harnesses`` in place of "not installed": Pi and Oh My Pi
        can resolve to one store, and the adapter that leaves it to the other
        says whose it is.

        Returns:
            The other harness's display name — ``"Oh My Pi"`` for Pi, ``"Pi"``
            for Oh My Pi — when :meth:`_yields_to` names it, else ``None``.
        """
        return self._yields_to()

    def _yield_reason(self, owner: str) -> str:
        """Return the refusal for anything in a sessions directory left to ``owner``."""
        return (
            f"this sessions directory holds {owner}'s data; {self.display} leaves it alone "
            "(refused)"
        )

    def session_files(self) -> Iterator[Path]:
        """Yield each ``<sessions dir>/<folder>/*.jsonl``, one level deep, each file once.

        Deeper files are not sessions (Oh My Pi keeps sub-agent transcripts in a
        per-session directory), a file directly in the sessions directory
        belongs to no project folder, and a name starting with ``.`` is a
        temporary file rather than a transcript.

        A file reached through a symbolic link — in a project folder that is a
        link to another folder, or as a link itself — is yielded only when no
        path without a link reaches it, and then once, so a session is never
        listed twice that way. Two hard links in real folders are two names,
        and both are yielded: one may be what a move cut short between linking
        a transcript at its destination and unlinking its old name leaves, and
        a re-run finishes that (:func:`_half_moved`).

        Yields:
            Transcript paths, sorted.
        """
        try:
            folders = [
                path
                for path in self.sessions_dir().iterdir()
                if not path.name.startswith(".") and path.is_dir()
            ]
        except OSError:
            return
        found: list[Path] = []
        reached: set[object] = set()
        aliases: list[tuple[str, object, Path]] = []
        for folder in folders:
            linked = folder.is_symlink()
            for path in _folder_transcripts(folder):
                try:
                    info = os.stat(path)
                    key: object = (info.st_dev, info.st_ino)
                except (OSError, ValueError):
                    key = path  # cannot be told apart from anything: kept
                if linked or path.is_symlink():
                    aliases.append((str(path), key, path))
                else:
                    found.append(path)
                    reached.add(key)
        for _, key, path in sorted(aliases, key=lambda alias: alias[0]):
            if key not in reached:
                reached.add(key)
                found.append(path)
        yield from sorted(found)

    # --- format hooks ----------------------------------------------------

    def read_header(self, path: Path) -> dict | None:
        """Return a transcript's session header, or ``None`` if it has none.

        Mirrors Pi's session loader (``loadEntriesFromFile``): the first parsed
        JSON value must be an object with ``"type": "session"`` and a string
        ``id``. Pi cannot open any other file, and its session list passes over
        every file whose first value is not a ``session`` record.

        Args:
            path: A ``.jsonl`` file.

        Returns:
            The header, or ``None`` — also when the file cannot be read.
        """
        value = _first_value(path)
        if (
            isinstance(value, dict)
            and value.get("type") == "session"
            and isinstance(value.get("id"), str)
        ):
            return value
        return None

    def is_fragment(self, path: Path) -> bool:
        """Return True if a file without a session header is debris the harness left.

        Pi leaves two kinds: an empty file, created a moment before the first
        write (``openSync(file, "wx")``), and a file its append recreated after
        the session file was deleted or moved — whose first line is a tree
        entry: a string ``id``, an entry ``type`` Pi writes
        (:attr:`fragment_entry_types`) and a ``parentId`` that is a string or
        ``null``.

        Only the first line itself counts, never a later one: a session whose
        header line was damaged still holds its whole conversation after it,
        and is not debris. A first line that is blank, too long to vouch for
        (:attr:`fragment_line_limit`), not UTF-8 or not JSON makes the file
        one ``sx`` does not recognize.

        Args:
            path: A file :meth:`read_header` found no header in.

        Returns:
            Whether it is such a fragment.

        Raises:
            OSError: The file cannot be read.
        """
        if os.stat(path).st_size == 0:
            return True
        limit = self.fragment_line_limit
        with path.open("rb") as fh:
            raw = fh.readline(limit + 1)
        if len(raw) > limit:
            return False
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, RecursionError):
            return False
        if not isinstance(value, dict):
            return False
        entry_id = value.get("id")
        if not isinstance(entry_id, str) or not entry_id:
            return False
        if value.get("type") not in self.fragment_entry_types or "parentId" not in value:
            return False
        parent = value["parentId"]
        return parent is None or isinstance(parent, str)

    def _classify(self, path: Path) -> tuple[TranscriptKind, dict | None, str | None]:
        """Classify one ``.jsonl`` file, keeping what was read on the way.

        Args:
            path: A file in a project folder.

        Returns:
            ``(kind, header, why)``: the :class:`TranscriptKind`; the session
            header for a session; and, for a file ``sx`` does not recognize, why.
        """
        try:
            info = os.stat(path)
        except (FileNotFoundError, NotADirectoryError):
            return TranscriptKind.MISSING, None, None
        except OSError as exc:
            return TranscriptKind.UNRECOGNIZED, None, f"cannot be checked: {exc.strerror or exc}"
        except ValueError:
            return TranscriptKind.UNRECOGNIZED, None, "cannot be checked"
        if not stat.S_ISREG(info.st_mode):
            return TranscriptKind.UNRECOGNIZED, None, "is not a regular file"  # never opened
        header = self.read_header(path)
        if header is not None:
            return TranscriptKind.SESSION, header, None
        try:
            fragment = self.is_fragment(path)
        except (FileNotFoundError, NotADirectoryError):
            return TranscriptKind.MISSING, None, None
        except OSError as exc:
            return TranscriptKind.UNRECOGNIZED, None, f"cannot be read: {exc.strerror or exc}"
        if fragment:
            return TranscriptKind.FRAGMENT, None, None
        return TranscriptKind.UNRECOGNIZED, None, "does not start with a session header"

    def transcript_kind(self, path: Path) -> TranscriptKind:
        """Return what a ``.jsonl`` file in a project folder holds.

        A file is a session when :meth:`read_header` finds its header, a
        fragment when :meth:`is_fragment` says so, missing when nothing is
        there, and otherwise — a file that cannot be read included —
        unrecognized: listed, never deleted, moved or offered for cleanup.

        Args:
            path: A file in a project folder.

        Returns:
            Its :class:`TranscriptKind`.
        """
        return self._classify(path)[0]

    def is_entry(self, obj: dict) -> bool:
        """Return True if a parsed record is a tree entry rather than metadata.

        Args:
            obj: One parsed record.

        Returns:
            True for everything except the ``session`` header.
        """
        return obj.get("type") != "session"

    def title_for_file(self, path: Path, header: dict) -> str:
        """Return a session's title: its name, else its first prompt.

        The name is the latest ``session_info`` entry's, and an empty one clears
        it; the prompt is the first user message that gives a title. How each
        reads as a title is up to :meth:`name_title` and :meth:`prompt_title`.
        Only lines carrying the raw marker of one of those two records are
        parsed, so the multi-megabyte lines elsewhere in a file — tool output,
        inline base64 images — are skipped unparsed.

        Args:
            path: The transcript.
            header: Its session header (unused by Pi; Oh My Pi keeps a title in
                its own).

        Returns:
            The title, or ``(no messages)``.
        """
        name: str | None = None
        first_user: str | None = None
        try:
            with path.open("rb") as fh:
                for raw in fh:
                    wants_user = first_user is None and _USER_MARK in raw
                    if not wants_user and _SESSION_INFO_MARK not in raw:
                        continue
                    obj = _parse_line(raw)
                    if not isinstance(obj, dict):
                        continue
                    kind = obj.get("type")
                    if kind == "session_info":
                        value = obj.get("name")
                        name = (self.name_title(value) or None) if isinstance(value, str) else None
                    elif wants_user and kind == "message":
                        first_user = self.prompt_title(obj.get("message")) or None
        except OSError:
            pass
        return name or first_user or "(no messages)"

    def name_title(self, name: str) -> str:
        """Return a session name as a title.

        Args:
            name: A ``session_info`` entry's ``name``.

        Returns:
            The name, stripped; empty when it holds nothing.
        """
        return name.strip()

    def prompt_title(self, message: object) -> str:
        """Return the title a user message gives: its first line of text.

        Args:
            message: The ``message`` field of a ``message`` entry.

        Returns:
            Up to 60 characters of the first non-blank line of its text blocks,
            joined with a space, flattened; empty when it is not a user message
            or holds no text.
        """
        return _title_line(message)

    def current_leaf(self, entries: list[dict]) -> str | None:
        """Return the id of the entry the session resumes from.

        Pi persists no leaf pointer: on load its leaf is the last entry in the
        file, whichever branch that is on.

        Args:
            entries: The tree entries, in file order.

        Returns:
            The leaf id, or ``None`` for an empty session. An id that names no
            entry falls back to the last entry.
        """
        if not entries:
            return None
        leaf = entries[-1].get("id")
        return leaf if isinstance(leaf, str) else None

    def entry_message(self, entry: dict) -> Message | None:
        """Render one tree entry as a :class:`Message`, or ``None`` to skip it.

        Message entries map by role; compactions, branch summaries, model
        changes and displayed extension messages are shown as well. Bookkeeping
        — labels, names, usage, thinking levels, context edits, extension state,
        system prompts — is not. Malformed shapes are skipped, never raised on.

        Args:
            entry: One tree entry.

        Returns:
            The message, or ``None``.
        """
        kind = entry.get("type")
        timestamp = parse_ts(entry.get("timestamp"))
        if kind == "message":
            message = entry.get("message")
            return self._agent_message(message, timestamp) if isinstance(message, dict) else None
        if kind == "compaction":
            summary = entry.get("summary")
            text = "Context compacted"
            if isinstance(summary, str) and summary.strip():
                text = f"{text}\n\n{summary.strip()}"
            return Message(Role.SYSTEM, text=text, timestamp=timestamp)
        if kind == "branch_summary":
            summary = entry.get("summary")
            if isinstance(summary, str) and summary.strip():
                return Message(
                    Role.SYSTEM, text=f"Branch summary: {summary.strip()}", timestamp=timestamp
                )
            return None
        if kind == "model_change":
            parts = [
                value
                for value in (entry.get("provider"), entry.get("modelId"))
                if isinstance(value, str) and value
            ]
            if not parts:
                return None
            return Message(Role.SYSTEM, text="model → " + "/".join(parts), timestamp=timestamp)
        if kind == "custom_message":
            return _displayed(entry.get("display"), entry.get("content"), timestamp)
        return None

    @staticmethod
    def _agent_message(message: dict, timestamp: datetime | None) -> Message | None:
        """Render the ``message`` of a message entry by its role.

        Args:
            message: The ``message`` field.
            timestamp: The entry's timestamp.

        Returns:
            The message, or ``None`` for roles that are not shown.
        """
        role = message.get("role")
        content = message.get("content")
        if role == "user":
            text = _content_text(content)
            return Message(Role.USER, text=text, timestamp=timestamp) if text else None
        if role == "assistant":
            return _assistant_message(message, timestamp)
        if role == "toolResult":
            name = message.get("toolName")
            summary = f"{name} result" if isinstance(name, str) and name else "tool result"
            if _js_truthy(message.get("isError")):
                summary += " (error)"
            return Message(
                Role.TOOL,
                text=_truncate(_content_text(content), 200),
                timestamp=timestamp,
                tool_summary=summary,
            )
        if role == "bashExecution":
            command = message.get("command")
            summary = (
                "bash: " + _truncate(_one_line(command), 120)
                if isinstance(command, str) and command.strip()
                else "bash"
            )
            output = message.get("output")
            text = _truncate(output, 200) if isinstance(output, str) else ""
            code = message.get("exitCode")
            if isinstance(code, int) and not isinstance(code, bool):
                text = f"{text}\n(exit code {code})" if text else f"(exit code {code})"
            return Message(Role.TOOL, text=text, timestamp=timestamp, tool_summary=summary)
        if role == "custom" or role == "hookMessage":
            return _displayed(message.get("display"), content, timestamp)
        return None

    def header_update(
        self, header: dict, old: Path, new: Path, *, moved_from: Path | None
    ) -> dict | None:
        """Return the header a move should write, or ``None`` to leave it as is.

        Only the header's ``cwd`` is a structural record of the project
        directory; paths inside messages are history and stay untouched.

        Args:
            header: The transcript's current session header.
            old: The project directory being moved away from.
            new: The project directory being moved to.
            moved_from: The transcript's path before the move when the move also
                relocates it, else ``None``. Unused by Pi.

        Returns:
            A new replacement header, or ``None``; ``header`` itself is never
            edited.
        """
        updated = repoint(header.get("cwd", ""), old, new)
        if updated is None:
            return None
        return {**header, "cwd": updated}

    def destructive_refusal(self, session: Session, *, rewriting: bool) -> str | None:
        """Return why ``session`` must not be deleted or moved now, if it must not.

        A file listed in a format ``sx`` does not recognize is always refused.
        Otherwise the header is re-read at the moment of the operation: one that
        no longer parses as Pi's means the format changed under ``sx`` — it
        refuses rather than guesses — a different session id means the file was
        replaced, and a format version Pi never wrote refuses a delete and a
        move alike.

        Args:
            session: The session about to be deleted or moved.
            rewriting: True for a move, False for a delete. Pi refuses both for
                the same reasons.

        Returns:
            A refusal reason, or ``None`` when the operation may proceed. A file
            that is already gone is left to the guards to report.
        """
        reason = _unrecognized_refusal(session)
        if reason is not None:
            return reason
        path = session.primary_path
        if path is None or not os.path.lexists(path):
            return None
        header = self.read_header(path)
        if header is None:
            return "no longer starts with a session header; the format may have changed (refused)"
        if (header.get("id") or path.stem) != session.session_id:
            return "now holds a different session (refused)"
        return self.format_refusal(path, header)

    def format_refusal(self, path: Path, header: dict) -> str | None:
        """Return why a session file is in a format ``sx`` does not know, if it is.

        Such a session is listed, but nothing ever destroys it: a delete and a
        move refuse it, a move leaves it where it is, and orphan cleanup
        counts its project as unknown, so the folder holding it is never
        offered, or removed, as a dead project's.

        Args:
            path: A transcript with a session header.
            header: That header.

        Returns:
            A refusal reason — for Pi, a header recording a format version Pi
            never wrote — or ``None``.
        """
        return _unknown_version(header)

    def busy_transcript(self, path: Path) -> bool:
        """Return True if ``path`` must stay where it is on a move.

        Pi leaves no sign of a session it has open — no lock, no lease, nothing
        on disk ``sx`` can check — so this cannot tell whether Pi has a
        transcript open, and it does not look at modification times either. The
        move service's recency check (a session written in the last 90 seconds
        needs a typed confirmation) and :attr:`PiAdapter.open_session_caveat`
        are all there is for that. What stays is what a move must neither
        rewrite nor carry: a session whose header records a format version
        ``sx`` does not know, and a file ``sx`` does not recognize. A fragment,
        or a file already gone, may travel.

        Args:
            path: A transcript.

        Returns:
            Whether it must stay put.
        """
        kind, header, _ = self._classify(path)
        if kind is TranscriptKind.UNRECOGNIZED:
            return True
        return header is not None and self.format_refusal(path, header) is not None

    # --- discovery -------------------------------------------------------

    def discover(self) -> Iterator[Session]:
        """Yield a session for every transcript, and for every file sx does not recognize.

        Yields:
            Sessions with cheap metadata; none when the adapter leaves the
            sessions directory to another harness. A fragment is not a session.
        """
        if self._yields_to() is not None:
            return
        yield from super().discover()

    def _session_from_file(self, path: Path) -> Session | None:
        """Build a metadata-only :class:`Session` from one transcript.

        The project is the header's ``cwd`` — never the folder name, whose
        encoding collides. A session is an orphan only when
        :func:`~sx.fsguard.project_state` says its project is definitely gone,
        never when it merely cannot be checked. A file ``sx`` does not recognize
        is listed too, flagged (see :meth:`_unrecognized_session`).

        Args:
            path: A transcript.

        Returns:
            The session, or ``None`` for a fragment or a file that is gone.
        """
        kind, header, why = self._classify(path)
        if kind is TranscriptKind.UNRECOGNIZED:
            return self._unrecognized_session(path, why or "unrecognized")
        if header is None:
            return None
        project = _header_cwd(header)
        title = self.title_for_file(path, header)
        extra: dict = {}
        parent = header.get("parentSession")
        if isinstance(parent, str) and parent:
            extra["parent_session"] = parent
            title = f"{title} · fork"
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        return Session(
            harness=self.name,
            session_id=header["id"] or path.stem,
            project_path=project,
            title=title,
            created=parse_ts(header.get("timestamp")),
            modified=file_mtime(path),
            size_bytes=size,
            paths=[path],
            is_orphan=project is not None and project_state(project) == MISSING,
            extra=extra,
        )

    def _unrecognized_session(self, path: Path, why: str) -> Session:
        """List a file ``sx`` does not recognize, flagged so nothing destroys it.

        It belongs to no project, so no move ever selects it; its title says what
        it is; ``extra[UNRECOGNIZED_KEY]`` holds why, and every delete and move
        refuses it (:meth:`destructive_refusal`). It can still be opened, and
        shows whatever of it parses as entries.

        Args:
            path: The file.
            why: Why it is not recognized.

        Returns:
            The session.
        """
        try:
            size = os.lstat(path).st_size
        except OSError:
            size = 0
        return Session(
            harness=self.name,
            session_id=path.stem,
            project_path=None,
            title=f"{UNRECOGNIZED_TITLE} {path.name}",
            modified=file_mtime(path),
            size_bytes=size,
            paths=[path],
            extra={UNRECOGNIZED_KEY: why},
        )

    def recorded_project(self, path: Path) -> str | None:
        """Return the project directory a transcript's header records, if any."""
        header = self.read_header(path)
        return _header_cwd(header) if header is not None else None

    def unrecognized_warning(self) -> str | None:
        """Return a warning counting the files ``sx`` does not recognize, if any.

        Returns:
            One line for :meth:`store_warnings`, or ``None``.
        """
        count = sum(
            1
            for path in self.session_files()
            if self.transcript_kind(path) is TranscriptKind.UNRECOGNIZED
        )
        if not count:
            return None
        return (
            f"{count} file(s) in {short_path(self.sessions_dir())} do not look like "
            f'{self.display} sessions: they are listed as "{UNRECOGNIZED_TITLE} <file name>" and '
            "can be browsed, but sx never deletes or moves them."
        )

    # --- loading ---------------------------------------------------------

    def load(self, session: Session) -> list[Message]:
        """Return the branch the session was last on."""
        return self.load_transcript(session).current

    def load_transcript(self, session: Session) -> Transcript:
        """Load a transcript and its branches.

        Entries without a string ``id`` (version 1 files, or old entries copied
        under a newer header) cannot form a tree, so such a file is shown
        linearly, in file order.

        Args:
            session: The session to load.

        Returns:
            The :class:`~sx.model.Transcript`; ``branches``, ``tree`` and
            ``hidden`` are set only when the session branched.
        """
        path = session.primary_path
        if path is None:
            return Transcript()
        entries = [obj for obj in iter_jsonl(path) if self.is_entry(obj)]
        if any(not isinstance(entry.get("id"), str) for entry in entries):
            linear = [self.entry_message(entry) for entry in entries]
            return Transcript(current=[message for message in linear if message is not None])
        return self._tree(entries)

    def _tree(self, entries: list[dict]) -> Transcript:
        """Arrange id-linked entries into the current branch and the others.

        An entry that shows nothing — a label, usage, a thinking-level change —
        is transparent: it never makes a branch of its own, never splits one,
        and never marks a jump in the file-order view. Each entry is walked past
        a bounded number of times, so the work grows with the file, not with
        the square of it.

        Args:
            entries: Tree entries in file order, each with a string ``id``.

        Returns:
            The :class:`~sx.model.Transcript`.
        """
        by_id: dict[str, dict] = {}
        for entry in entries:
            by_id[entry["id"]] = entry  # a repeated id: the last one wins, as in Pi
        ordered = [entry for entry in entries if by_id[entry["id"]] is entry]
        if not ordered:
            return Transcript()
        rendered = {entry["id"]: self.entry_message(entry) for entry in ordered}

        leaf = self.current_leaf(entries)
        if not isinstance(leaf, str) or leaf not in by_id:
            leaf = ordered[-1]["id"]
        path, _ = _walk_up(leaf, by_id, {})
        owner: dict[str, int] = dict.fromkeys(path, 0)
        position = _running_counts(path, rendered)
        current = [rendered[i] for i in path if rendered[i] is not None]

        # A branch ends at a message nothing shown continues from; branches are
        # taken in the order their tips were written. Whatever is left after
        # that (an off-branch cycle has no tip) is walked newest first. Every
        # walk starts at a message and stops at the first entry already placed.
        shown = [entry["id"] for entry in ordered if rendered[entry["id"]] is not None]
        continued = _ancestors(shown, by_id)
        tips = [entry_id for entry_id in shown if entry_id not in continued]
        branches: list[Branch] = []
        for start in tips + shown[::-1]:
            if start in owner:
                continue
            segment, fork = _walk_up(start, by_id, owner)
            label = _fork_label(fork, owner, position)
            owner.update(dict.fromkeys(segment, len(branches) + 1))
            position.update(_running_counts(segment, rendered))
            messages = [rendered[i] for i in segment if rendered[i] is not None]
            branches.append(Branch(label=label, messages=messages))
        if not branches:
            return Transcript(current=current)

        on_path = set(path)
        counts = _running_counts([entry["id"] for entry in ordered], rendered)
        anchor = _anchors(by_id, rendered)
        tree: list[Message] = []
        last: str | None = None
        for entry in ordered:
            entry_id = entry["id"]
            message = rendered[entry_id]
            if message is None:
                continue
            parent = _parent_of(entry)
            linked = parent is not None and parent in by_id
            above = anchor(parent) if linked else None
            if last is not None and above != last:
                text = _jump_label(above, linked, counts, entry_id in on_path)
                tree.append(Message(Role.OTHER, text=text, marker=True))
            tree.append(message)
            last = entry_id
        hidden = sum(len(branch.messages) for branch in branches)
        return Transcript(current=current, branches=branches, tree=tree, hidden=hidden)

    # --- guards ----------------------------------------------------------

    def _refusal(self, session: Session, *, rewriting: bool) -> str | None:
        """Return why ``session`` must not be touched, including by yielding.

        Args:
            session: The session about to be deleted or moved.
            rewriting: True for a move, False for a delete.

        Returns:
            A refusal reason, or ``None``.
        """
        owner = self._yields_to()
        if owner is not None:
            return self._yield_reason(owner)
        path = session.primary_path
        if path is not None and path.parent.is_symlink():
            return _LINKED_FOLDER
        reason = _unrecognized_refusal(session)
        if reason is not None:
            return reason
        return self.destructive_refusal(session, rewriting=rewriting)

    def _through_link(self, unit: MoveUnit) -> Path | None:
        """Return the first path a unit would act on through a symlinked project folder.

        Args:
            unit: A unit about to run.

        Returns:
            The path, or ``None``. A project folder is one directly under the
            sessions directory; anywhere else — memory, say — is not judged.
        """
        root = self.sessions_dir()
        for src, _ in unit.relocations:
            for folder in (src, src.parent):
                if folder.parent == root and folder.is_symlink():
                    return src
        for path in unit.rewrites:
            if path.parent.parent == root and path.parent.is_symlink():
                return path
        return None

    def delete(self, session: Session, *, dry_run: bool = False) -> DeleteResult:
        """Delete a session's transcript, unless its header now forbids it.

        Pi keeps nothing beside the transcript — its own delete removes just the
        ``.jsonl`` — so nothing else goes with it, not even a file whose name
        starts the same way.

        Args:
            session: The session to remove.
            dry_run: If True, report what would happen and change nothing.

        Returns:
            A :class:`DeleteResult`.
        """
        reason = self._refusal(session, rewriting=False)
        if reason is not None:
            target = session.primary_path or Path(f"<{self.name}:{session.session_id}>")
            return DeleteResult(dry_run=dry_run, skipped={target: reason})
        return super().delete(session, dry_run=dry_run)

    # --- move ------------------------------------------------------------

    def plan_move(self, sessions: list[Session], old: Path, new: Path) -> MovePlan:
        """Plan a move, refusing sessions :meth:`destructive_refusal` objects to.

        Args:
            sessions: Sessions selected for the move.
            old: The project directory being moved away from.
            new: The project directory being moved to.

        Returns:
            A :class:`MovePlan` covering every selected session; refused ones
            appear in ``blocked`` and are otherwise left out of the work.
        """
        allowed: list[Session] = []
        refused: dict[Path, str] = {}
        for session in sessions:
            reason = self._refusal(session, rewriting=True)
            if reason is None:
                allowed.append(session)
            else:
                target = session.primary_path or Path(f"<{self.name}:{session.session_id}>")
                refused[target] = reason
        plan = super().plan_move(allowed, old, new)
        plan.sessions = list(sessions)
        plan.blocked.update(refused)
        return plan

    def _foreign_transcripts(self, folder: Path, selected: list[Session], old: Path) -> list[Path]:
        """Return transcripts that must not travel with ``folder``.

        Extends the base rule (another project's transcripts) with any that are
        not part of the move and either are files ``sx`` does not recognize or
        are reported by :meth:`busy_transcript`.

        Args:
            folder: The folder about to be relocated.
            selected: The sessions being moved out of it.
            old: The project directory being moved away from.

        Returns:
            The transcripts that stay put.
        """
        stay = super()._foreign_transcripts(folder, selected, old)
        chosen = {session.primary_path for session in selected}
        for path in sorted(folder.glob("*.jsonl")):
            if path in chosen or path in stay:
                continue
            if self.transcript_kind(path) is TranscriptKind.UNRECOGNIZED or self.busy_transcript(
                path
            ):
                stay.append(path)
        return sorted(stay)

    def _unit_check(
        self, unit: MoveUnit, plan: MovePlan, simulation: Namespace | None
    ) -> tuple[tuple[Path, str] | None, bool]:
        """Check a unit as the base does, and never let it carry an unrecognized file.

        :meth:`busy_transcript` already keeps such a file in place for Pi; this
        holds for every harness on this format, whatever its own
        ``busy_transcript`` says. Nor does a unit run through a project folder
        that has become a symbolic link since planning.

        Args:
            unit: The unit about to run.
            plan: The plan being executed.
            simulation: The dry run's namespace, or ``None`` to ask the disk.

        Returns:
            ``(problem, merge)``, as :meth:`EncodedFolderAdapter._unit_check`.
        """
        finishing = [pair for pair in unit.relocations if _half_moved(*pair)]
        rest = replace(unit, relocations=[p for p in unit.relocations if p not in finishing])
        problem, merge = super()._unit_check(rest if finishing else unit, plan, simulation)
        if problem is not None or merge:
            return problem, merge
        problem = self._finishing_problem(unit, finishing, simulation)
        if problem is not None:
            return problem, False
        linked = self._through_link(unit)
        if linked is not None:
            return (linked, _LINKED_FOLDER), False
        for path in self._carried(unit, simulation):
            if self.transcript_kind(path) is TranscriptKind.UNRECOGNIZED:
                return (path, _UNRECOGNIZED_STAYS), False
        return None, False

    def _finishing_problem(
        self,
        unit: MoveUnit,
        finishing: list[tuple[Path, Path]],
        simulation: Namespace | None,
    ) -> tuple[Path, str] | None:
        """Check the half-done links a unit finishes (:func:`_half_moved`).

        The base check refuses any destination that exists, so each such pair
        is checked here instead, as a relocation and the session it carries
        would be: both names inside the store, the destination folder a real
        one, and the transcript neither refused nor busy.

        Args:
            unit: The unit about to run.
            finishing: Its relocations that finish a half-done link.
            simulation: The dry run's namespace, or ``None`` to ask the disk.

        Returns:
            The path and reason the unit cannot run, or ``None``.
        """
        roots = self.store_roots()
        exempt = (self.sessions_root(), *roots)
        sessions = {session.primary_path: session for session in unit.sessions}
        for src, dst in finishing:
            if not is_within(src, roots) or not is_within(dst, roots):
                return src, "outside store root (refused)"
            refusal = self._folder_refusal(dst.parent, simulation, exempt)
            if refusal is not None:
                return src, refusal
            session = sessions.get(src)
            reason = self.destructive_refusal(session, rewriting=True) if session else None
            if reason is not None:
                return src, reason
            if self.busy_transcript(src):
                return src, "in use, or not safe to rewrite (refused)"
        return None

    def _plan_entry(self, src: Path, dst: Path, plan: MovePlan, roots: list[Path]) -> None:
        """Record one relocation as the base does, or plan to finish a half-done one.

        A transcript a move cut short already linked at its destination
        (:func:`_half_moved`) — re-pointed, then linked, but its old name not
        yet unlinked — is planned to move there, though its destination
        exists: finishing it unlinks the old name, and the file is already
        where it goes.

        Args:
            src: The path to move.
            dst: Where it goes.
            plan: The plan to record into.
            roots: The adapter's store roots, for the containment guard.
        """
        if _half_moved(src, dst) and is_within(src, roots) and is_within(dst, roots):
            plan.relocations.append((src, dst))
            return
        super()._plan_entry(src, dst, plan, roots)

    def _rename(self, src: Path, dst: Path, simulation: Namespace | None) -> str | None:
        """Rename as the base does; a dry run also finishes a half-done link.

        On disk, :func:`~sx.adapters.relocation.rename_no_clobber` already
        finishes one — the same file at the destination — by unlinking the
        old name. A dry run's namespace only knows the destination exists.

        Args:
            src: The path to move.
            dst: Where it goes.
            simulation: The dry run's namespace, or ``None`` for a real move.

        Returns:
            ``None`` on success, otherwise why nothing was moved.
        """
        if simulation is not None and simulation.lexists(src) and _half_moved(src, dst):
            simulation.rename(src, dst)
            return None
        return super()._rename(src, dst, simulation)

    def move_transform(
        self, path: Path, plan: MovePlan, *, moved_from: Path | None
    ) -> Callable[[dict], dict | None]:
        """Return the transform that re-points one transcript: its header alone.

        Every other line is copied byte for byte. The header update returns a
        new mapping and never edits the record it is given, so a unit that
        fails later can put the transcript back exactly.

        Args:
            path: The transcript about to be rewritten, at its current path.
            plan: The move being carried out.
            moved_from: ``path`` when the same move also relocates the
                transcript, else ``None``.

        Returns:
            A transform for :func:`sx.util.rewrite_jsonl`.
        """
        return _header_transform(self._header_mover(plan, moved_from))

    def _header_mover(
        self, plan: MovePlan, moved_from: Path | None
    ) -> Callable[[dict], dict | None]:
        """Bind :meth:`header_update` to one plan and one transcript.

        Args:
            plan: The plan being executed.
            moved_from: The transcript's path if the plan relocates it.

        Returns:
            A header update for :func:`_header_transform`.
        """

        def update(header: dict) -> dict | None:
            """Return the moved header, or ``None`` if nothing changes."""
            return self.header_update(header, plan.old, plan.new, moved_from=moved_from)

        return update

    def move(self, plan: MovePlan, *, dry_run: bool = False) -> MoveResult:
        """Re-point and relocate the transcripts, then their forks' links.

        A fork's header records the absolute path of the transcript it was
        forked from (``parentSession``), so relocating that transcript would
        orphan the link. After the move, every header in the store pointing at
        a relocated transcript is re-pointed at where it actually ended up —
        :attr:`~sx.model.MoveResult.relocated`, which also covers a folder
        merged into one that appeared after planning, and a transcript moved
        on by a later step of the same move.

        A move cut short — by Ctrl-C, or an error — still re-points the links
        of every unit that finished, before the interruption goes on; what
        that re-pointing did, which no unit's record holds, goes with the
        exception as its ``partial_result`` (a :class:`MoveResult`), for the
        op-log to absorb. And a link an earlier, interrupted run of the same
        move left dangling is repaired (:meth:`_stranded_links`) — even by a
        run with nothing left to relocate.

        Args:
            plan: The plan produced by :meth:`plan_move`.
            dry_run: If True, report what would happen and change nothing.

        Returns:
            A :class:`MoveResult`.
        """
        owner = self._yields_to()
        if owner is not None:
            refused = {self.sessions_dir(): self._yield_reason(owner)}
            return MoveResult(dry_run=dry_run, skipped=refused)
        # Resolved while the old paths still exist; afterwards they cannot be.
        origins = self._carried_origins(plan)
        state = self._thread_state()
        state.progress = None
        try:
            result = super().move(plan, dry_run=dry_run)
        except BaseException as exc:
            partial, state.progress = state.progress, None
            links = _final_locations(partial.relocated) if partial is not None else {}
            if links:
                self._repoint_on_the_way_out(exc, links, origins, partial, dry_run=dry_run)
            raise
        state.progress = None
        links = _final_locations(result.relocated)
        stranded, resolved = self._stranded_links(plan, links)
        links.update(stranded)
        origins.update(resolved)
        if links:
            self._repoint_fork_links(links, origins, result, dry_run=dry_run)
        return result

    def _repoint_on_the_way_out(
        self,
        exc: BaseException,
        links: dict[Path, Path],
        origins: dict[Path, str],
        partial: MoveResult,
        *,
        dry_run: bool,
    ) -> None:
        """Re-point the fork links of the units a move cut short had finished.

        What it does is recorded in a result of its own, hung on the exception
        that cut the move short as ``partial_result`` before anything is
        re-pointed — so whatever stops it part-way, it is reported — and on a
        second interruption too: the units' records already hold everything
        else. Any other error is swallowed: the interruption is what to report.

        Args:
            exc: What cut the move short.
            links: Old transcript path -> where it ended up, for the units that
                finished.
            origins: Each old path with symlinks resolved, taken before the move.
            partial: The move's running result, which lists what the finished
                units rewrote.
            dry_run: If True, count without writing.
        """
        step = MoveResult(dry_run=dry_run)
        _hand_over(exc, step)
        try:
            self._repoint_fork_links(
                links, origins, step, dry_run=dry_run, rewritten=partial.rewritten
            )
        except Exception:  # noqa: BLE001, S110 - the interruption is what to report
            pass
        except BaseException as again:  # interrupted again: report what was done all the same
            _hand_over(again, step)
            raise

    def _run_unit(
        self,
        unit: MoveUnit,
        plan: MovePlan,
        result: MoveResult,
        simulation: Namespace | None,
    ) -> None:
        """Run one unit as the base does, keeping the move's running result at hand.

        A move cut short is re-raised before :meth:`move` gets its result
        back, so the result every finished unit was folded into is kept here,
        for those units' fork links to be re-pointed on the way out.

        Args:
            unit: The unit to run.
            plan: The plan being executed.
            result: The whole move's result (mutated in place).
            simulation: The dry run's namespace, or ``None`` for a real move.
        """
        self._thread_state().progress = result
        super()._run_unit(unit, plan, result, simulation)

    def _thread_state(self) -> threading.local:
        """Return this adapter's per-thread scratch state, made on first use.

        Returns:
            A :class:`threading.local` kept on the adapter, so two threads
            running operations on one adapter never see each other's state.
        """
        state = self.__dict__.get("_sx_thread_state")
        if state is None:
            state = self.__dict__.setdefault("_sx_thread_state", threading.local())
        return state

    def _stranded_links(
        self, plan: MovePlan, links: dict[Path, Path]
    ) -> tuple[dict[Path, Path], dict[Path, str]]:
        """Find fork links an earlier, interrupted run of the same move left dangling.

        Only a link naming a path where nothing is, inside a folder of this
        store, qualifies — and only when a transcript of the same name now
        records a directory under ``new`` whose counterpart under ``old`` that
        folder is named for: the transcript the link named, which the earlier
        run moved before it was cut short.

        Args:
            plan: The plan that just ran.
            links: What this run relocated itself (path before -> after),
                which needs no repair.

        Returns:
            ``(links, origins)`` to add to what :meth:`_repoint_fork_links` is
            given: each dangling link, as a path, -> the transcript it names
            now; and that path with symlinks resolved.
        """
        root_real = _realpath(self.sessions_dir())
        headers: list[dict] = []
        placed: dict[tuple[str, str], Path] = {}
        for path in self.session_files():
            header = self.read_header(path)
            if header is None:
                continue
            headers.append(header)
            cwd = _header_cwd(header)
            if cwd is None or is_under(cwd, plan.old) or not is_under(cwd, plan.new):
                continue
            before = repoint(cwd, plan.new, plan.old)
            if before is None:
                continue
            for name in self.folder_names(before):
                if name != path.parent.name:
                    placed.setdefault((name, path.name), path)
        found: dict[Path, Path] = {}
        resolved: dict[Path, str] = {}
        if not placed:
            return found, resolved
        for header in headers:
            link = header.get("parentSession")
            if not isinstance(link, str) or not link or Path(link) in links:
                continue
            try:
                if os.path.lexists(link):
                    continue
            except (OSError, ValueError):
                continue
            real = Path(_realpath(link))
            if str(real.parent.parent) != root_real:
                continue
            target = placed.get((real.parent.name, real.name))
            if target is not None:
                found[Path(link)] = target
                resolved[Path(link)] = str(real)
        return found, resolved

    @staticmethod
    def _carried_origins(plan: MovePlan) -> dict[Path, str]:
        """Resolve, before anything moves, every transcript a plan may relocate.

        A fork may name its parent through another spelling of the same path —
        a symbolic link on the way. Once the parent has moved, its old path no
        longer exists to be resolved, so it is resolved here first: each
        transcript relocated on its own, and each transcript inside a folder
        relocated (or merged) whole.

        Args:
            plan: The plan about to run.

        Returns:
            Transcript path before the move -> that path with symlinks resolved.
        """
        sources = {source for source, _ in plan.relocations}
        for unit in plan.units:
            sources.update(source for source, _ in unit.relocations)
        found: dict[Path, str] = {}
        for source in sorted(sources):
            if is_transcript(source):
                found[source] = _realpath(source)
                continue
            try:
                names = sorted(os.listdir(source))
            except (OSError, ValueError):
                continue
            for name in names:
                path = source / name
                if is_transcript(path):
                    found[path] = _realpath(path)
        return found

    def _repoint_fork_links(
        self,
        links: dict[Path, Path],
        origins: dict[Path, str],
        result: MoveResult,
        *,
        dry_run: bool,
        rewritten: list[Path] | None = None,
    ) -> None:
        """Re-point ``parentSession`` in every header that names a moved transcript.

        A link matches by its exact text or by the location it resolves to. A
        transcript written in the last 90 seconds, or that
        :meth:`busy_transcript` says must stay as it is, is left alone and
        listed as skipped — unless this move just rewrote it, which is what made
        it recent — and so is one in a project folder that is a symbolic link.
        A link is never pointed at a path where nothing is, and a link that
        changed after it was read here is left as it now is.

        Args:
            links: Old transcript path to where it ended up, for transcripts
                relocated.
            origins: Each old path with symlinks resolved, taken before the move.
            result: The result to record outcomes in (mutated in place).
            dry_run: If True, count without writing.
            rewritten: What this move rewrote, when ``result`` is not the move's
                own result; ``None`` takes ``result.rewritten``.
        """
        by_text = {str(old): new for old, new in links.items()}
        by_real = {origins[old]: new for old, new in links.items() if old in origins}
        done = result.rewritten if rewritten is None else rewritten
        touched = set(done) | {links.get(path, path) for path in done}
        now = time.time()
        for path in self.session_files():
            header = self.read_header(path)
            parent = header.get("parentSession") if header is not None else None
            if not isinstance(parent, str) or not parent:
                continue
            target = by_text.get(parent) or by_real.get(_realpath(parent))
            if target is None:
                continue
            if not dry_run and not os.path.lexists(target):
                result.skipped[path] = (
                    f"its fork link was not re-pointed: nothing is at {target} after the "
                    "move (refused)"
                )
                continue
            if path.parent.is_symlink():
                result.skipped[path] = f"its fork link was not re-pointed: {_LINKED_FOLDER}"
                continue
            if path not in touched:
                if self.busy_transcript(path):
                    result.skipped[path] = (
                        "in use, or not safe to rewrite; its fork link was not re-pointed (refused)"
                    )
                    continue
                if _written_within(path, _LIVE_WINDOW_SECONDS, now):
                    result.skipped[path] = (
                        f"written in the last {_LIVE_WINDOW_SECONDS} seconds; "
                        "its fork link was not re-pointed (refused)"
                    )
                    continue
            transform = _header_transform(_relink(parent, str(target)))
            changed, error = rewrite_jsonl(path, transform, dry_run=dry_run)
            if error is not None:
                result.skipped[path] = error
                continue
            if changed:
                result.fields_updated += changed
                if path not in touched:
                    result.rewritten.append(path)

    # --- orphans ---------------------------------------------------------

    def find_orphans(self) -> list[Orphan]:
        """Find idle empty folders, fragments and dead projects.

        A folder whose transcripts all record a working directory that is
        definitely gone is reported as dead — but never when any transcript's
        directory is unknown, a fragment, a file ``sx`` does not recognize and
        a session in a format it does not know (:meth:`format_refusal`)
        included, and never for a directory that merely cannot be checked
        (an unmounted volume, a parent ``sx`` cannot traverse). An empty folder
        is reported only once it and everything in it have gone unwritten for
        :attr:`folder_idle_seconds`. A file ``sx`` does not recognize is never
        reported. Symlinked folders are skipped: their contents lie outside the
        store.

        Returns:
            The orphans; every path lies inside :meth:`store_roots`.
        """
        if self._yields_to() is not None:
            return []
        try:
            folders = sorted(
                path
                for path in self.sessions_dir().iterdir()
                if not path.name.startswith(".") and not path.is_symlink() and path.is_dir()
            )
        except OSError:
            return []
        orphans: list[Orphan] = []
        for folder in folders:
            orphans.extend(self._folder_orphans(folder))
        return orphans

    def _folder_orphans(self, folder: Path) -> list[Orphan]:
        """Return the orphans in one project folder.

        Args:
            folder: A folder directly under the sessions directory.

        Returns:
            Its orphans, possibly none.
        """
        transcripts = _folder_transcripts(folder)
        if not transcripts:
            if self._idle_reason(folder) is not None:
                return []  # the harness may have just created it
            reason, size = _folder_contents(folder)
            if self.empty_folder_note:
                reason = f"{reason}; {self.empty_folder_note}"
            return [Orphan(self.name, OrphanKind.EMPTY, [folder], reason, size)]
        orphans: list[Orphan] = []
        cwds: list[str] = []
        unknown = False
        for path in transcripts:
            kind, header, _ = self._classify(path)
            if kind is TranscriptKind.FRAGMENT:
                unknown = True
                orphans.append(
                    Orphan(
                        self.name,
                        OrphanKind.STRAY_TEMP,
                        [path],
                        "no session header — a fragment, not a session",
                        dir_size(path),
                    )
                )
                continue
            cwd = _header_cwd(header) if header is not None else None
            if cwd is None or self.format_refusal(path, header) is not None:
                # No project, a file sx does not recognize, one gone, or a
                # session in a format sx does not know: whose it is stays open.
                unknown = True
            else:
                cwds.append(cwd)
        if unknown or not cwds:
            return orphans
        if any(project_state(cwd) != MISSING for cwd in cwds):
            return orphans
        gone = sorted(set(cwds))
        reason = (
            f"project directory no longer exists: {gone[0]}"
            if len(gone) == 1
            else "project directories no longer exist: " + ", ".join(gone)
        )
        orphans.append(
            Orphan(self.name, OrphanKind.DEAD_PROJECT, [folder], reason, dir_size(folder))
        )
        return orphans

    def _idle_reason(self, folder: Path, *, contents_only: bool = False) -> str | None:
        """Return why a project folder has not been idle long enough to remove.

        Args:
            folder: A project folder.
            contents_only: Judge only what is inside it — right after ``sx``
                deleted a session there, which changed the folder itself.

        Returns:
            A refusal reason when the folder, or anything in it, was written in
            the last :attr:`folder_idle_seconds` or cannot be examined in full;
            ``None`` when it is idle, or the adapter asks for no idle time.
        """
        seconds = self.folder_idle_seconds
        if seconds <= 0:
            return None
        newest = _newest_write(folder, contents_only=contents_only)
        if newest is None:
            return "cannot be examined in full, so it may be in use (refused)"
        if time.time() - newest < seconds:
            hours = seconds / 3600
            span = f"{hours:g} hours" if hours >= 1 else f"{seconds} seconds"
            return (
                f"was written in the last {span}: {self.display} creates a project's folder "
                "when it starts there, before it saves a session (refused)"
            )
        return None

    def _orphan_revalidation(self, orphan: Orphan) -> dict[Path, str]:
        """Ask again, at the moment of deletion, whether each path is still an orphan.

        On top of the base checks: nothing is deleted from a sessions directory
        this adapter now leaves to another harness; a fragment must still be a
        fragment; an empty folder must still have been idle for
        :attr:`folder_idle_seconds`; and a dead project's folder must still
        hold nothing but sessions whose recorded directories are definitely
        gone.

        Args:
            orphan: The orphan about to be deleted.

        Returns:
            Path -> refusal reason, for the paths that must stay.
        """
        owner = self._yields_to()
        if owner is not None:
            return {path: self._yield_reason(owner) for path in orphan.paths}
        refused = super()._orphan_revalidation(orphan)
        root = self.sessions_root()
        for path in orphan.paths:
            if path in refused or not os.path.lexists(path):
                continue
            reason: str | None = None
            if orphan.kind is OrphanKind.STRAY_TEMP and is_transcript(path):
                kind = self.transcript_kind(path)
                if kind is TranscriptKind.SESSION:
                    reason = "now holds a session (refused)"
                elif kind is TranscriptKind.UNRECOGNIZED:
                    reason = "is no longer a fragment: sx does not recognize it now (refused)"
            elif path.parent == root and not path.is_symlink() and path.is_dir():
                if orphan.kind is OrphanKind.EMPTY:
                    reason = self._idle_reason(path)
                elif orphan.kind is OrphanKind.DEAD_PROJECT:
                    reason = self._dead_folder_refusal(path)
            if reason is not None:
                refused[path] = reason
        return refused

    def _dead_folder_refusal(self, folder: Path) -> str | None:
        """Return why a dead project's folder must stay after all, or ``None``.

        Args:
            folder: A folder offered as a dead project's.

        Returns:
            A refusal reason when it now holds anything but sessions, in a
            format ``sx`` knows, whose recorded directories are definitely
            gone.
        """
        for path in _folder_transcripts(folder):
            kind, header, _ = self._classify(path)
            if kind is TranscriptKind.MISSING:
                continue
            if header is None:
                return f"{path.name} is not a session sx can place in a project (refused)"
            problem = self.format_refusal(path, header)
            if problem is not None:
                return f"{path.name}: {problem}"
            cwd = _header_cwd(header)
            if cwd is None:
                return f"{path.name} records no project directory (refused)"
            if project_state(cwd) != MISSING:
                return f"{path.name}'s project directory is there, or cannot be checked (refused)"
        return None

    # --- project-scoped state --------------------------------------------

    def project_leftovers(self, project: str) -> ProjectLeftovers | None:
        """Offer the project's session folder once its last session is gone.

        Only when every transcript in it records ``project`` (or a directory
        inside it): a folder shared with a colliding project, or holding a file
        whose project cannot be read, is not offered. Nor is a folder anything
        in which — the last session included — was written in the last
        :attr:`folder_idle_seconds`: the harness may have that session open.
        No configuration is touched — not even Pi's trust decisions.

        Args:
            project: The project directory.

        Returns:
            The folder as a :class:`ProjectLeftovers`, or ``None``.
        """
        if self._yields_to() is not None:
            return None
        folder = self.sessions_dir() / self.folder_for(project)
        if folder.is_symlink() or not folder.is_dir():
            return None
        try:
            transcripts = [path for path in folder.iterdir() if path.name.endswith(".jsonl")]
        except OSError:
            return None
        for path in transcripts:
            cwd = self.recorded_project(path)
            if cwd is None or not is_under(cwd, project):
                return None
        if self._idle_reason(folder) is not None:
            return None
        return ProjectLeftovers(project_path=project, paths=[folder], size_bytes=dir_size(folder))

    def delete_project_leftovers(
        self, leftovers: ProjectLeftovers, *, dry_run: bool = False
    ) -> DeleteResult:
        """Remove the folder :meth:`project_leftovers` offered, if still a leftover.

        The offer is made before the last session is deleted and acted on after,
        possibly minutes later, so the folder is checked again first (see
        :meth:`_still_in_use`): a session started there meanwhile must not go
        with it.

        Args:
            leftovers: The plan produced by :meth:`project_leftovers`.
            dry_run: If True, report what would happen and change nothing.

        Returns:
            A :class:`DeleteResult`.
        """
        owner = self._yields_to()
        expected = self.sessions_dir() / self.folder_for(leftovers.project_path)
        allowed: list[Path] = []
        refused: dict[Path, str] = {}
        for path in leftovers.paths:
            if owner is not None:
                refused[path] = self._yield_reason(owner)
            elif path != expected:
                refused[path] = "not this project's session folder (refused)"
            elif (reason := self._still_in_use(path)) is not None:
                refused[path] = reason
            else:
                allowed.append(path)
        result = self._delete_paths(allowed, dry_run=dry_run)
        result.skipped.update(refused)
        return result

    def _still_in_use(self, folder: Path) -> str | None:
        """Return why a leftover folder must stay, or ``None`` if it may go.

        It must hold no session and no file ``sx`` does not recognize; a
        fragment written in the last 90 seconds keeps it too (the harness still
        writing to a session just deleted). And everything in it must have gone
        unwritten for :attr:`folder_idle_seconds` — the folder itself aside,
        since deleting the last session just changed it.

        Args:
            folder: The project folder about to be removed.

        Returns:
            A refusal reason, or ``None``. A missing folder is left for the
            delete guard to report.
        """
        if not os.path.lexists(folder):
            return None
        try:
            transcripts = sorted(p for p in folder.iterdir() if p.name.endswith(".jsonl"))
        except OSError as exc:
            return f"cannot read: {exc}"
        now = time.time()
        for path in transcripts:
            kind = self.transcript_kind(path)
            if kind is TranscriptKind.SESSION:
                return f"still holds a session ({path.name}) (refused)"
            if kind is TranscriptKind.UNRECOGNIZED:
                return f"holds {path.name}, which sx does not recognize (refused)"
            if kind is TranscriptKind.FRAGMENT and _written_within(path, _LIVE_WINDOW_SECONDS, now):
                return (
                    f"{path.name} was written in the last {_LIVE_WINDOW_SECONDS} seconds (refused)"
                )
        return self._idle_reason(folder, contents_only=True)


# --- Pi ----------------------------------------------------------------------------


class PiAdapter(PiFormatAdapter):
    """Adapter for Pi coding agent sessions.

    Sessions live at ``<agent dir>/sessions/--<encoded cwd>--/<ts>_<id>.jsonl``,
    where the agent directory is ``PI_CODING_AGENT_DIR`` when set and usable,
    else ``~/.pi/agent``. A store that carries Oh My Pi's marks
    (:func:`omp_markers`) is left to Oh My Pi entirely.

    Attributes:
        name: Machine identifier (``"pi"``).
        display: Human-readable harness name.
        open_session_caveat: Pi marks nothing about the sessions it has open,
            so the confirmations say what deleting or moving one does to it.
        folder_idle_seconds: :data:`PI_FOLDER_IDLE_SECONDS` — Pi creates a
            project's folder when it starts there.
        empty_folder_note: An empty folder offered for removal is still the
            one Pi uses when it starts in that project: Pi creates the folder
            only when it is missing, so an old one is in use as soon as Pi
            starts there.
    """

    name = "pi"
    display = "Pi"
    open_session_caveat = (
        "Pi leaves no sign of which sessions it has open, so close a session in Pi before "
        "deleting or moving it: otherwise Pi's next turn fails if the session's folder is "
        "gone, or is written to a header-less file that Pi never lists if only the "
        "session's file is gone."
    )
    folder_idle_seconds = PI_FOLDER_IDLE_SECONDS
    empty_folder_note = (
        "Pi uses this folder when it starts in that project, so if Pi is running there, "
        "its next prompt fails once the folder is gone"
    )

    def agent_dir(self) -> Path:
        """Return Pi's agent directory, honoring ``PI_CODING_AGENT_DIR``.

        ``home`` is resolved here, in this module's namespace and at call time,
        so a test that redirects it keeps every derived path inside its sandbox.

        Returns:
            The override when usable, else ``~/.pi/agent``.
        """
        override, _ = agent_dir_override(home())
        return override if override is not None else home() / ".pi" / "agent"

    def sessions_dir(self) -> Path:
        """Return ``<agent dir>/sessions``."""
        return self.agent_dir() / "sessions"

    def folder_for(self, project: str) -> str:
        """Return the folder name Pi uses for ``project``."""
        return encode_session_folder(project)

    def name_title(self, name: str) -> str:
        """Return a session name as Pi's session list shows it.

        Pi shows the name in place of the first message (``session-selector.js``):
        each control character becomes a space and the result is trimmed. ``sx``
        cuts it at 80 characters, where Pi cuts it at the terminal's width.

        Args:
            name: A ``session_info`` entry's ``name``.

        Returns:
            The title; empty when nothing is left, which clears the name.
        """
        return _title_text(name)

    def prompt_title(self, message: object) -> str:
        """Return the title a user message gives, as Pi's session list shows it.

        That is the whole text of the message, not just its first line — a
        string content as it is, or its text blocks joined with a space, images
        left out (``extractTextContent``) — with each control character,
        newlines included, turned into a space and the result trimmed. ``sx``
        cuts it at 80 characters, where Pi cuts it at the terminal's width, and
        passes over a prompt that is blank once cleaned, which Pi would show as
        an empty title.

        Args:
            message: The ``message`` field of a ``message`` entry.

        Returns:
            The title; empty when it is not a user message or holds no text.
        """
        return _title_text(_first_user_text(message))

    def omp_evidence(self) -> list[str]:
        """Return what in Pi's store only Oh My Pi would have written.

        Returns:
            :func:`omp_markers` for Pi's agent and sessions directories.
        """
        return omp_markers(self.agent_dir(), self.sessions_dir())

    def _yields_to(self) -> str | None:
        """Return ``Oh My Pi`` when Pi's store carries its marks.

        Whichever profile ``sx``'s Oh My Pi adapter reads, and whether or not it
        reads this directory at all: a store Oh My Pi has written to is
        effectively its own. A store with no Oh My Pi marks is Pi's.

        Returns:
            ``"Oh My Pi"``, or ``None``.
        """
        return _OMP if self.omp_evidence() else None

    def store_notes(self) -> list[str]:
        """Return where Pi's store is, when ``PI_CODING_AGENT_DIR`` moved it."""
        override, _ = agent_dir_override(home())
        if override is None:
            return []
        sessions = short_path(override / "sessions")
        return [f"{AGENT_DIR_ENV} is set; reading Pi sessions from {sessions}"]

    def store_warnings(self) -> list[str]:
        """Return problems with where Pi's sessions are, or may be, written.

        Returns:
            One line each for an ignored ``PI_CODING_AGENT_DIR``, a custom
            session directory (the environment variable or ``sessionDir`` in
            Pi's settings), and either a store Pi leaves to Oh My Pi or the
            files in it ``sx`` does not recognize.
        """
        warnings: list[str] = []
        _, ignored = agent_dir_override(home())
        if ignored is not None:
            warnings.append(f"{AGENT_DIR_ENV} is {ignored}; ignoring it and reading ~/.pi/agent.")
        custom = os.environ.get(SESSION_DIR_ENV, "")
        if custom:
            warnings.append(
                f"{SESSION_DIR_ENV} is set ({custom}): Pi writes new sessions there, and sx "
                "does not read custom session directories yet, so they are not listed."
            )
        settings = self.agent_dir() / "settings.json"
        configured = self._settings_session_dir(settings)
        if configured is not None:
            warnings.append(
                f"sessionDir is set in {short_path(settings)} ({configured}): Pi writes "
                "sessions there, and sx does not read custom session directories yet, so "
                "they are not listed."
            )
        evidence = self.omp_evidence()
        if evidence:
            warnings.append(
                f"{short_path(self.sessions_dir())} holds Oh My Pi's data "
                f"({', '.join(evidence)}), so Pi lists, moves and deletes nothing there; "
                "its sessions are listed under Oh My Pi when Oh My Pi reads that directory."
            )
        else:
            unrecognized = self.unrecognized_warning()
            if unrecognized is not None:
                warnings.append(unrecognized)
        return warnings

    @staticmethod
    def _settings_session_dir(settings: Path) -> str | None:
        """Return the ``sessionDir`` set in Pi's global settings, if any.

        Args:
            settings: ``<agent dir>/settings.json``.

        Returns:
            The configured value, or ``None`` when unset or unreadable.
        """
        try:
            data = json.loads(settings.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError, UnicodeError, RecursionError):
            return None
        value = data.get("sessionDir") if isinstance(data, dict) else None
        return value if isinstance(value, str) and value else None
