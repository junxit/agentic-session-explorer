"""Normalized domain model shared across every harness adapter.

Each adapter translates its harness's idiosyncratic on-disk format into these
types. Because the model is harness-agnostic, the transcript viewer, the
Markdown exporter, and the delete flow are each written once and work for all
harnesses — present and future.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from sx.util import human_size


class Role(enum.Enum):
    """Who produced a message turn.

    Attributes:
        USER: A human prompt.
        ASSISTANT: A model response.
        TOOL: Output from a tool/function call.
        SYSTEM: System or environment context.
        OTHER: Anything that does not map cleanly to the above.
    """

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"
    OTHER = "other"


class Capability(enum.Flag):
    """What an adapter supports, so the UI can degrade gracefully.

    Attributes:
        NONE: No capabilities.
        BROWSE: Sessions can be discovered and their transcripts loaded.
        ORPHANS: The adapter can detect orphaned folders/files.
        DELETE: Sessions and orphans can be deleted.
    """

    NONE = 0
    BROWSE = enum.auto()
    ORPHANS = enum.auto()
    DELETE = enum.auto()


@dataclass(slots=True)
class Message:
    """A single normalized turn in a transcript.

    Attributes:
        role: Who produced this turn.
        text: Human-readable text content (already flattened from blocks/parts).
        timestamp: When the turn occurred, if the source recorded it.
        tool_summary: One-line summary of a tool call/result, when applicable
            (e.g. ``"tool: Bash(git status)"``). ``None`` for plain messages.
        thinking: Reasoning/thinking content, kept separate from ``text`` so the
            viewer can dim or hide it.
    """

    role: Role
    text: str = ""
    timestamp: datetime | None = None
    tool_summary: str | None = None
    thinking: str | None = None


@dataclass(slots=True)
class Session:
    """A single chat session, normalized across harnesses.

    Discovery populates the cheap metadata fields (everything except the full
    transcript). The transcript itself is loaded lazily via
    :meth:`HarnessAdapter.load` only when a session is opened.

    Attributes:
        harness: Adapter name that owns this session (e.g. ``"claude"``).
        session_id: Stable identifier within the harness.
        project_path: Decoded absolute project/cwd this session belongs to, or
            ``None`` if it could not be resolved.
        title: Human-friendly label for lists.
        created: Session start time, if known.
        modified: Last-modified time (usually file mtime).
        message_count: Number of transcript turns, if cheaply known.
        size_bytes: Total on-disk size of the session's file(s).
        paths: Every file/dir that constitutes this session — what delete and
            export operate on. The primary transcript is ``paths[0]``.
        is_orphan: True if the owning project path no longer exists.
        extra: Adapter-specific scratch data (kept out of the shared surface).
    """

    harness: str
    session_id: str
    project_path: str | None = None
    title: str = ""
    created: datetime | None = None
    modified: datetime | None = None
    message_count: int | None = None
    size_bytes: int = 0
    paths: list[Path] = field(default_factory=list)
    is_orphan: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def primary_path(self) -> Path | None:
        """The main transcript file, or ``None`` if this session has no files."""
        return self.paths[0] if self.paths else None

    @property
    def project_name(self) -> str:
        """Short, display-friendly name of the owning project."""
        if self.project_path:
            return Path(self.project_path).name or self.project_path
        return "(unknown project)"


class OrphanKind(enum.Enum):
    """Why something was flagged as an orphan.

    Attributes:
        DEAD_PROJECT: Session/folder whose decoded project path no longer exists.
        EMPTY: An empty directory or zero-message session file.
        STRAY_TEMP: Leftover temp/scratch file (e.g. ``projects.json.*.tmp``).
    """

    DEAD_PROJECT = "dead_project"
    EMPTY = "empty"
    STRAY_TEMP = "stray_temp"


@dataclass(slots=True)
class Orphan:
    """An orphaned artifact that is a candidate for cleanup.

    Attributes:
        harness: Adapter that detected it.
        kind: Why it is considered orphaned.
        paths: Files/dirs that make up this orphan (deleted together).
        reason: Human-readable explanation shown in the UI.
        size_bytes: Total on-disk size.
    """

    harness: str
    kind: OrphanKind
    paths: list[Path]
    reason: str
    size_bytes: int = 0


@dataclass(slots=True)
class DeleteResult:
    """Outcome of a delete operation.

    Attributes:
        removed: Paths that were successfully removed.
        freed_bytes: Total bytes freed.
        skipped: Mapping of path -> reason for anything intentionally not removed
            (e.g. blocked by a guard).
        dry_run: True if this was a preview and nothing was actually removed.
        note: Optional human-readable detail about removals that are not files —
            e.g. database rows. ``None`` for ordinary file-only deletes.
    """

    removed: list[Path] = field(default_factory=list)
    freed_bytes: int = 0
    skipped: dict[Path, str] = field(default_factory=dict)
    dry_run: bool = False
    note: str | None = None

    @property
    def refused(self) -> dict[Path, str]:
        """Targets actively refused or errored, excluding merely-absent ones.

        A path that simply did not exist is not a failure worth alarming the user
        about; a path blocked by the store-root guard, or one that errored, is.
        """
        return {
            path: reason
            for path, reason in self.skipped.items()
            if "does not exist" not in reason
        }

    @property
    def failed(self) -> bool:
        """True if nothing happened and at least one target was refused.

        ``note`` counts as something happening: for database-backed harnesses the
        real work is row removal, which has no path to report in ``removed``.
        """
        return not self.removed and not self.note and bool(self.refused)

    def summary(self) -> str:
        """One-line human summary of what this result actually did.

        Used for both the confirmation preview and the post-action notification so
        the two can never disagree about the outcome.
        """
        bits: list[str] = []
        if self.removed:
            bits.append(f"{len(self.removed)} path(s) · {human_size(self.freed_bytes)}")
        if self.note:
            bits.append(self.note)
        if self.refused:
            bits.append(f"{len(self.refused)} refused")
        return " · ".join(bits) if bits else "nothing to remove"
