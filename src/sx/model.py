"""Normalized domain model shared across every harness adapter.

Each adapter translates its harness's idiosyncratic on-disk format into these
types. Because the model is harness-agnostic, the transcript viewer, the
Markdown exporter, and the delete flow are each written once and work for all
harnesses — present and future.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import ClassVar

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
        MOVE: Sessions can be re-pointed at a new project directory.
    """

    NONE = 0
    BROWSE = enum.auto()
    ORPHANS = enum.auto()
    DELETE = enum.auto()
    MOVE = enum.auto()


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
        marker: True for a separator ``sx`` inserted itself — where a branch
            starts, for instance — rather than a turn the harness recorded.
            Renderers style it as a divider and leave it out of message counts.
    """

    role: Role
    text: str = ""
    timestamp: datetime | None = None
    tool_summary: str | None = None
    thinking: str | None = None
    marker: bool = False


class BranchMode(enum.Enum):
    """How a session that branched is presented, in the viewer and in exports.

    Some harnesses (Pi, Oh My Pi) keep every attempt in the same file: retrying
    or editing an earlier prompt starts a new branch and leaves the abandoned one
    in place. The user picks how much of that to see.

    Attributes:
        CURRENT: The branch the session was last on, plus a count of what the
            other branches hold.
        APPENDIX: The current branch, followed by each other branch.
        TREE: Every entry in the order it was written, with markers where the
            conversation jumps to another branch.
    """

    CURRENT = "current"
    APPENDIX = "appendix"
    TREE = "tree"

    def label(self) -> str:
        """Return a short human-readable name for the mode."""
        return {
            BranchMode.CURRENT: "current branch",
            BranchMode.APPENDIX: "current branch + other branches",
            BranchMode.TREE: "whole tree",
        }[self]

    def next(self) -> BranchMode:
        """Return the mode after this one, wrapping around."""
        members = list(BranchMode)
        return members[(members.index(self) + 1) % len(members)]


@dataclass(slots=True)
class Branch:
    """One side branch of a session tree, from its fork point to its tip.

    Attributes:
        label: Where the branch forks, for its heading.
        messages: The branch's own messages; the history it shares with the
            branch it forks from is not repeated.
    """

    label: str
    messages: list[Message] = field(default_factory=list)


@dataclass(slots=True)
class Transcript:
    """A loaded transcript, with any branches it has.

    Every harness can produce one: a linear transcript simply has no
    ``branches``, and then every :class:`BranchMode` presents the same thing.

    Attributes:
        current: The branch the session was last on — what the harness shows
            when the session is resumed.
        branches: The other branches, in the order their tips were written.
        tree: Every message in file order, with branch markers; empty when the
            session never branched.
        hidden: How many messages lie on other branches.
    """

    current: list[Message] = field(default_factory=list)
    branches: list[Branch] = field(default_factory=list)
    tree: list[Message] = field(default_factory=list)
    hidden: int = 0

    @property
    def shown_branches(self) -> list[Branch]:
        """The other branches that hold at least one message.

        A branch with nothing to show is never listed, numbered or counted:
        it would only add an empty divider.
        """
        return [branch for branch in self.branches if branch.messages]

    @property
    def branched(self) -> bool:
        """True if the session has anything off its current branch."""
        return any(branch.messages for branch in self.branches)

    def messages(self, mode: BranchMode) -> list[Message]:
        """Flatten the transcript for display or export in ``mode``.

        A divider separates two runs of messages, so one with nothing before it
        — the current branch is empty, or the file opens on another branch —
        is left out rather than drawn as a rule under the header. Exports would
        otherwise open with two rules in a row.

        Args:
            mode: How to present the other branches.

        Returns:
            The messages to render, with divider markers where branches start.
        """
        if not self.branched or mode is BranchMode.CURRENT:
            return list(self.current)
        if mode is BranchMode.TREE:
            return _between_messages(self.tree)
        out = list(self.current)
        shown = self.shown_branches
        for number, branch in enumerate(shown, start=1):
            if out:
                out.append(
                    Message(
                        Role.OTHER,
                        text=f"Other branch {number} of {len(shown)} — {branch.label}",
                        marker=True,
                    )
                )
            out.extend(branch.messages)
        return out

    def describe(self, mode: BranchMode) -> str | None:
        """Return a one-line description of what ``mode`` shows, if it matters.

        Branches are counted only when they hold messages — the current branch
        included, which is empty when the session resumes from its start.

        Args:
            mode: The presentation being described.

        Returns:
            ``None`` for a session that never branched; otherwise a sentence for
            the viewer's header line and the export's metadata.
        """
        if not self.branched:
            return None
        count = len(self.shown_branches)
        current = any(not message.marker for message in self.current)
        if mode is BranchMode.CURRENT:
            return (
                f"current branch only — {self.hidden} message(s) on "
                f"{count} other branch(es) not shown"
            )
        if mode is BranchMode.APPENDIX:
            if not current:
                return f"current branch (empty), then {count} other branch(es)"
            return f"current branch, then {count} other branch(es)"
        total = count + (1 if current else 0)
        noun = "branch" if total == 1 else "branches"
        return f"whole tree in the order it was written — {total} {noun}"


def _between_messages(messages: list[Message]) -> list[Message]:
    """Return ``messages`` with every divider that separates nothing removed.

    Leading and trailing markers go, and a run of markers keeps only its last
    one — the jump the run ends on is the one that leads to what follows.

    Args:
        messages: Messages and markers in display order.

    Returns:
        A new list in which each marker sits between two messages.
    """
    out: list[Message] = []
    for message in messages:
        if message.marker:
            if not out:
                continue
            if out[-1].marker:
                out[-1] = message
                continue
        out.append(message)
    while out and out[-1].marker:
        out.pop()
    return out


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
        STALE_SESSION: Artifact keyed to a session id that no longer exists —
            state the harness wrote beside a conversation that has since been
            deleted.
    """

    DEAD_PROJECT = "dead_project"
    EMPTY = "empty"
    STRAY_TEMP = "stray_temp"
    STALE_SESSION = "stale_session"


@dataclass(slots=True)
class Orphan:
    """An orphaned artifact that is a candidate for cleanup.

    Attributes:
        harness: Adapter that detected it.
        kind: Why it is considered orphaned.
        paths: Files/dirs that make up this orphan (deleted together).
        reason: Human-readable explanation shown in the UI.
        size_bytes: Total on-disk size.
        fingerprints: What each path looked like when it was listed, from
            :func:`sx.fsguard.survey` — taken at scan time so the delete can
            refuse anything that changed since. ``None`` for an orphan built
            outside a scan, which is held to the store-root guards only.
    """

    harness: str
    kind: OrphanKind
    paths: list[Path]
    reason: str
    size_bytes: int = 0
    fingerprints: dict[Path, str] | None = None


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


@dataclass(slots=True)
class MoveUnit:
    """One atomic piece of a move: its rewrites and relocations happen together or not at all.

    A transcript re-pointed at the new directory while it stays in the folder
    named for the old one is invisible to the harness, and a transcript
    relocated with its old directory still recorded is offered later as a dead
    project's. So every rewrite belongs to the relocation that carries its file,
    and a unit that cannot finish is rolled back.

    Attributes:
        kind: :attr:`FOLDER` (a project folder renamed whole), :attr:`SESSION`
            (one transcript and the entries that travel with it),
            :attr:`ENTRY` (anything else moved on its own, such as a memory
            folder) or :attr:`REWRITE` (a transcript re-pointed where it is).
        relocations: ``(source, destination)`` renames, in the order they run;
            a session's transcript comes last, after its sidecars.
        rewrites: Transcripts re-pointed before the renames, at their paths
            before the move.
        sessions: The sessions whose transcripts this unit re-points, checked
            again just before it runs.
    """

    FOLDER: ClassVar[str] = "folder"
    SESSION: ClassVar[str] = "session"
    ENTRY: ClassVar[str] = "entry"
    REWRITE: ClassVar[str] = "rewrite"

    kind: str
    relocations: list[tuple[Path, Path]] = field(default_factory=list)
    rewrites: list[Path] = field(default_factory=list)
    sessions: list[Session] = field(default_factory=list)

    def label(self) -> str:
        """Return a short description of the unit for logs and messages."""
        if self.relocations:
            source, destination = self.relocations[-1]
            return f"{self.kind} {source} -> {destination}"
        if self.rewrites:
            return f"{self.kind} {self.rewrites[0]}"
        return self.kind


@dataclass(slots=True)
class MovePlan:
    """What re-pointing one harness at a new project directory would involve.

    A plan is produced before anything is touched, so the confirmation can state
    the full extent of the work — including the parts that will be refused.

    Attributes:
        harness: Adapter that produced this plan.
        old: The project directory being moved away from.
        new: The project directory being moved to.
        sessions: Sessions the move applies to.
        rewrites: Files whose recorded project path will be rewritten in place.
        relocations: ``(source, destination)`` pairs to be moved on disk.
        live: Sessions written within the active window; rewriting these races
            the harness, so they are called out and gated behind a typed phrase.
        blocked: Mapping of path -> reason for work that cannot proceed (a name
            collision at the destination, a target outside the store roots).
        note: Human-readable detail about non-file work, such as database rows.
        include_config: Also re-point project state the harness keeps outside its
            session store. Honored only by adapters that have such state (today,
            Claude's ``.claude.json`` and ``history.jsonl``); opt-in,
            because those files belong to a harness that may be running.
        units: The atomic pieces ``rewrites`` and ``relocations`` are grouped
            into, in the order they run (encoded-folder harnesses only). A
            subclass that drops a pair from ``relocations`` or a path from
            ``rewrites`` after planning drops the whole unit it belongs to.
        on_unit: Called with each unit and its own result as soon as it has
            run, so the op-log records progress as it happens and a move cut
            short still reports what it did. Set by
            :class:`~sx.service.MoveService`; never by adapters.
    """

    harness: str
    old: Path
    new: Path
    sessions: list[Session] = field(default_factory=list)
    rewrites: list[Path] = field(default_factory=list)
    relocations: list[tuple[Path, Path]] = field(default_factory=list)
    live: list[Session] = field(default_factory=list)
    blocked: dict[Path, str] = field(default_factory=dict)
    note: str | None = None
    include_config: bool = False
    units: list[MoveUnit] = field(default_factory=list)
    on_unit: Callable[[MoveUnit, MoveResult], None] | None = field(
        default=None, repr=False, compare=False
    )
    #: Planning scratch: the store as the relocations planned so far leave it.
    _namespace: object | None = field(default=None, repr=False, compare=False)

    @property
    def empty(self) -> bool:
        """True if this harness has nothing to do for the move."""
        return not (self.rewrites or self.relocations or self.blocked or self.note)


@dataclass(slots=True)
class MoveResult:
    """Outcome of re-pointing one harness at a new project directory.

    Deliberately shaped like :class:`DeleteResult` so the preview and the
    post-action message are rendered from the same accessors and cannot disagree
    about what happened.

    Attributes:
        moved: ``(source, destination)`` pairs that were relocated, exactly as
            each rename ran — so a folder merged entry by entry is listed entry
            by entry.
        rewritten: Files whose contents were rewritten, at their paths before
            the move.
        fields_updated: How many recorded paths were changed in total.
        skipped: Mapping of path -> reason for anything not acted on.
        dry_run: True if this was a preview and nothing was changed.
        note: Optional detail about non-file work (e.g. database rows).
        relocated: Each transcript that moved, from its path before the move
            to where it actually ended up — whether its folder was renamed,
            merged, or the transcript moved alone.
    """

    moved: list[tuple[Path, Path]] = field(default_factory=list)
    rewritten: list[Path] = field(default_factory=list)
    fields_updated: int = 0
    skipped: dict[Path, str] = field(default_factory=dict)
    dry_run: bool = False
    note: str | None = None
    relocated: dict[Path, Path] = field(default_factory=dict)

    @property
    def refused(self) -> dict[Path, str]:
        """Targets actively refused or errored, excluding merely-absent ones."""
        return {
            path: reason
            for path, reason in self.skipped.items()
            if "does not exist" not in reason
        }

    @property
    def failed(self) -> bool:
        """True if nothing was changed and at least one target was refused."""
        return (
            not self.moved
            and not self.rewritten
            and not self.note
            and bool(self.refused)
        )

    def absorb(self, other: MoveResult) -> None:
        """Add what ``other`` did to this result, as one more step of the same move.

        Args:
            other: The result of one step — a :class:`MoveUnit`, say. Its
                note is not carried over: a note describes a whole move.
        """
        self.rewritten.extend(other.rewritten)
        self.fields_updated += other.fields_updated
        self.moved.extend(other.moved)
        self.relocated.update(other.relocated)
        self.skipped.update(other.skipped)

    def summary(self) -> str:
        """One-line human summary of what this result actually did."""
        bits: list[str] = []
        if self.rewritten:
            bits.append(
                f"{len(self.rewritten)} file(s) re-pointed"
                f" · {self.fields_updated} path(s) updated"
            )
        if self.moved:
            bits.append(f"{len(self.moved)} relocated")
        if self.note:
            bits.append(self.note)
        if self.refused:
            bits.append(f"{len(self.refused)} refused")
        return " · ".join(bits) if bits else "nothing to move"


@dataclass(slots=True)
class ProjectLeftovers:
    """Project-scoped state that outlives the sessions it was written beside.

    A project's memory, its store folder, and the harness's per-project settings
    belong to the *directory*, not to any one conversation. Deleting a session
    must never take them, but deleting the **last** session leaves them with
    nothing to belong to — so the confirmation offers them, and states what they
    are either way.

    Attributes:
        project_path: The project directory this state belongs to.
        memory_files: Memory documents found for the project. Reported on every
            delete so the user knows they are being kept.
        paths: Files/directories the optional cleanup would remove.
        config_notes: Human-readable descriptions of non-path edits the cleanup
            would make (a settings key, prompt-history rows).
        size_bytes: Total on-disk size of ``paths``.
    """

    project_path: str
    memory_files: list[Path] = field(default_factory=list)
    paths: list[Path] = field(default_factory=list)
    config_notes: list[str] = field(default_factory=list)
    size_bytes: int = 0

    @property
    def empty(self) -> bool:
        """True if there is nothing project-scoped left to clean up."""
        return not (self.paths or self.config_notes)

    def summary(self) -> str:
        """One-line description of what the optional cleanup would remove."""
        bits: list[str] = []
        if self.memory_files:
            bits.append(f"{len(self.memory_files)} memory file(s)")
        if self.paths:
            bits.append(f"{len(self.paths)} path(s) · {human_size(self.size_bytes)}")
        bits.extend(self.config_notes)
        return " · ".join(bits) if bits else "nothing project-scoped"
