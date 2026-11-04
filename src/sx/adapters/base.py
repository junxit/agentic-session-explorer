"""Adapter contract and the reusable JSONL-folder base implementation.

Every harness is represented by a subclass of :class:`HarnessAdapter`. Most
harnesses store sessions as a folder of JSONL files, so the bulk of that work
lives in :class:`JsonlFolderAdapter`; a concrete harness usually overrides only
a few small hooks.

Harnesses that are not installed still ship as adapter subclasses; their
:meth:`HarnessAdapter.available` returns ``False`` and the UI grays them out
until their store appears on disk.
"""

from __future__ import annotations

import errno
import functools
import os
import shutil
import stat
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from sx.fsguard import MISSING, project_state, survey
from sx.model import (
    Capability,
    DeleteResult,
    Message,
    MovePlan,
    MoveResult,
    MoveUnit,
    Orphan,
    OrphanKind,
    ProjectLeftovers,
    Session,
    Transcript,
)
from sx.util import (
    dir_size,
    file_mtime,
    is_under,
    is_within,
    iter_jsonl,
    read_first_line,
    repoint,
    rewrite_jsonl,
)

from .relocation import (
    Namespace,
    Recorder,
    canonical,
    is_transcript,
    landed,
    read_records,
    rename_no_clobber,
    same_entry,
)

#: An orphan path written to within this many seconds is refused: the harness
#: may be creating it right now — Pi makes its project folder at startup, well
#: before the first session file lands in it.
ORPHAN_QUIET_SECONDS = 90

#: Why a session's other files stay when one of them cannot move.
_STAYS_WHOLE = "stays with the rest of its session, which cannot move (refused)"

#: Why a transcript is not re-pointed when the folder holding it cannot move.
_FOLDER_STAYS = "its folder cannot move, so it is not re-pointed (refused)"


def _same_path(a: Path, b: Path) -> bool:
    """Return True if two paths resolve to the same location."""
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return False


def _regular_file(path: Path, simulation: object | None) -> bool:
    """Return True if ``path`` itself is a regular file — not a link to one.

    Args:
        path: A transcript about to be rewritten.
        simulation: The dry run's namespace, or ``None`` for a real move. A
            dry run's earlier steps exist only in it, so a path missing from
            the disk is left to it.

    Returns:
        Whether ``lstat`` finds a regular file there.
    """
    try:
        return stat.S_ISREG(os.lstat(path).st_mode)
    except FileNotFoundError:
        return simulation is not None
    except (OSError, ValueError):
        return False


def _what_stopped(exc: BaseException) -> str:
    """Describe what cut a unit short, for its record: Ctrl-C, or an error."""
    if isinstance(exc, Exception):
        return f"failed ({exc!r})"
    return "interrupted"


class HarnessAdapter(ABC):
    """Abstract base every harness adapter implements.

    Subclasses declare a :attr:`name`, a :attr:`display` label, the
    :attr:`capabilities` they support, and the set of :meth:`store_roots` they
    own. The four operations — discover, load, find_orphans, delete — form the
    contract the registry, CLI, and TUI depend on.

    Attributes:
        name: Stable machine identifier (e.g. ``"claude"``).
        display: Human-friendly label (e.g. ``"Claude Code"``).
        capabilities: Bitwise flags describing supported operations.
        supports_branches: True when :meth:`load_transcript` can return branches.
        open_session_caveat: A sentence shown in delete and move confirmations,
            for harnesses that keep a session open without leaving any sign of
            it on disk — no lock, no lease, nothing ``sx`` can check — so a
            session in use cannot be told from an idle one. ``None`` (the
            default) for harnesses whose open sessions ``sx`` can detect, or
            that cannot be hurt by it.
        fingerprints_orphans: True when every orphan :meth:`find_orphans`
            returns is fingerprinted as it is listed, so :meth:`delete_orphan`
            refuses whatever changed since, or was written in the last
            :data:`ORPHAN_QUIET_SECONDS`. Set by every adapter whose orphans are
            real store data; an orphan without a fingerprint is held to the
            store-root guards only.
    """

    name: str = "base"
    display: str = "Base"
    capabilities: Capability = Capability.NONE
    #: True when :meth:`load_transcript` can return branches. The UI only offers
    #: a choice of :class:`~sx.model.BranchMode` — and only loads a transcript
    #: up front to find out whether it branched — for adapters that set this.
    supports_branches: bool = False
    #: See the class docstring. Read by the UI with ``getattr``.
    open_session_caveat: str | None = None
    #: See the class docstring.
    fingerprints_orphans: bool = False

    def __init_subclass__(cls, **kwargs) -> None:
        """Fingerprint the orphans of every subclass's own ``find_orphans``.

        Wrapping here means an adapter's scan is fingerprinted however it is
        written — including adapters whose ``find_orphans`` predates the
        fingerprint — while the delete-time re-check stays in one place.

        Args:
            **kwargs: Passed on to :meth:`object.__init_subclass__`.
        """
        super().__init_subclass__(**kwargs)
        scan = cls.__dict__.get("find_orphans")
        if scan is None or getattr(scan, "_fingerprinted", False):
            return

        @functools.wraps(scan)
        def find_orphans(self, *args, **kw):
            """Run the adapter's scan, then fingerprint what it found."""
            return self.fingerprint_orphans(scan(self, *args, **kw))

        find_orphans._fingerprinted = True
        cls.find_orphans = find_orphans

    @abstractmethod
    def store_roots(self) -> list[Path]:
        """Return the root directories this adapter owns.

        These bound every filesystem operation: discovery only scans within
        them, and delete refuses to touch anything outside them.
        """
        raise NotImplementedError

    def protected_paths(self) -> list[Path]:
        """Return paths that must never be removed, even from inside a store root.

        Some store roots have to be wide enough to reach a handful of
        session-keyed entries while also containing unrelated bulk the harness
        depends on. Rather than narrow the root until it no longer covers what
        the cascade needs, the exceptions are named here and refused outright.

        Returns:
            Paths (files or directories) that are off limits. Default: none.
        """
        return []

    def available(self) -> bool:
        """Return True if this harness appears installed (any store root exists)."""
        return any(root.exists() for root in self.store_roots())

    @abstractmethod
    def discover(self) -> Iterator[Session]:
        """Yield sessions with cheap metadata only (no full transcript parse)."""
        raise NotImplementedError

    @abstractmethod
    def load(self, session: Session) -> list[Message]:
        """Parse and return the full transcript for ``session``."""
        raise NotImplementedError

    def load_transcript(self, session: Session) -> Transcript:
        """Return the transcript for ``session``, including any branches.

        The default wraps :meth:`load`: a harness that stores conversations
        linearly has a current branch and nothing else.

        Args:
            session: The session to load.

        Returns:
            A :class:`~sx.model.Transcript`.
        """
        return Transcript(current=self.load(session))

    def store_notes(self) -> list[str]:
        """Return informational lines about where this harness's store was found.

        Shown by ``sx harnesses`` — for instance, that an environment variable
        relocated the store. Default: none.
        """
        return []

    def store_warnings(self) -> list[str]:
        """Return warnings about the store location the user should act on.

        Shown by ``sx harnesses`` and in the TUI's error rows — for instance, that
        an override was ignored, or that sessions may be written somewhere this
        tool does not read. Default: none.
        """
        return []

    def store_owner(self) -> str | None:
        """Return the harness this adapter has handed its store to, if any.

        Shown by ``sx harnesses``. Two harnesses can share one store on disk;
        the adapter that steps aside for the other says so here.

        Returns:
            The display name of the other harness this adapter has handed its
            store to, when it has; otherwise ``None`` (the default).
        """
        return None

    def find_orphans(self) -> list[Orphan]:
        """Return orphaned artifacts for cleanup. Default: none."""
        return []

    def correlated_paths(self, session: Session) -> list[Path]:
        """Return extra paths tied to ``session`` (cascade delete). Default: none."""
        return []

    def last_activity(self, session: Session) -> datetime | None:
        """Return when this session was last written, for the live-delete guard.

        The default reads the primary transcript's mtime, which is correct for
        file-per-session harnesses. Adapters whose content does not live in that
        file (a shared database, for instance) must override this — otherwise the
        guard silently never engages for them.

        Args:
            session: The session to probe.

        Returns:
            The last-write time, or ``None`` if it cannot be determined (the
            caller treats unknown as active).
        """
        path = session.primary_path
        if path is None:
            return None
        return file_mtime(path)

    # --- guards a destructive operation re-checks -------------------------

    def destructive_refusal(self, session: Session, *, rewriting: bool) -> str | None:
        """Return why ``session`` must not be deleted or moved now, if it must not.

        Asked again at the moment of the operation — a move asks just before
        each unit runs, not only while planning — so a session that a harness
        opened in the meantime is caught. Default: no objection.

        Args:
            session: The session about to be deleted or moved.
            rewriting: True for a move, which rewrites the file; False for a
                delete.

        Returns:
            A refusal reason ending in ``(refused)``, or ``None``.
        """
        return None

    def busy_transcript(self, path: Path) -> bool:
        """Return True if ``path`` must stay where it is on a move.

        Asked for every transcript a move would carry or rewrite, while
        planning and again just before each unit runs. Default: never.

        Args:
            path: A transcript.

        Returns:
            Whether it is in use or unsafe to move.
        """
        return False

    # --- move ------------------------------------------------------------

    def sessions_for_project(
        self,
        project: Path,
        sessions: list[Session] | None = None,
    ) -> list[Session]:
        """Return the sessions belonging to ``project`` or any directory inside it.

        A session recorded in a subdirectory of the project (``/proj/docs``) is
        part of that project and must travel with it, so containment rather than
        equality decides membership.

        Args:
            project: The project directory to select for.
            sessions: An already-discovered pool to filter; when omitted the
                adapter discovers its own.

        Returns:
            The matching sessions.
        """
        pool = list(self.discover()) if sessions is None else sessions
        return [
            s
            for s in pool
            if s.harness == self.name and s.project_path and is_under(s.project_path, project)
        ]

    def stranded_sessions(
        self,
        old: Path,
        new: Path,
        sessions: list[Session] | None = None,
    ) -> list[Session]:
        """Return sessions an interrupted ``old`` -> ``new`` move left half done.

        Re-running a move must finish it. A session whose recorded project was
        already re-pointed under ``new`` is no longer selected by
        :meth:`sessions_for_project`, so harnesses that also relocate files
        report here the ones still sitting where ``old`` keeps them. Default:
        none — a harness that only rewrites in place has nothing to finish.

        Args:
            old: The project directory being moved away from.
            new: The project directory being moved to.
            sessions: An already-discovered pool; when omitted the adapter
                discovers its own.

        Returns:
            The sessions to hand to :meth:`plan_move` along with the selected
            ones.
        """
        return []

    def plan_move(self, sessions: list[Session], old: Path, new: Path) -> MovePlan:
        """Describe what re-pointing ``sessions`` from ``old`` to ``new`` would do.

        The default plan is empty, so an adapter that does not implement moving
        (every dormant harness) reports honestly that it has no work rather than
        appearing to succeed.

        Args:
            sessions: Sessions selected for the move.
            old: The project directory being moved away from.
            new: The project directory being moved to.

        Returns:
            A :class:`MovePlan`.
        """
        return MovePlan(harness=self.name, old=old, new=new, sessions=list(sessions))

    def move(self, plan: MovePlan, *, dry_run: bool = False) -> MoveResult:
        """Carry out a :class:`MovePlan`. Default: do nothing.

        A move that an error or Ctrl-C stops part-way lets the exception
        through. Before it does, it may attach the work it did that no unit
        reported through :attr:`MovePlan.on_unit` — Pi re-pointing fork links
        on its way out — to the exception as ``partial_result``, a
        :class:`MoveResult`; the service absorbs it into the record of the
        move, next to what the units reported.
        """
        return MoveResult(dry_run=dry_run)

    # --- delete ----------------------------------------------------------

    def delete(self, session: Session, *, dry_run: bool = False) -> DeleteResult:
        """Permanently delete a session's files (and correlated paths).

        Refuses, via :func:`sx.util.is_within`, to remove anything outside this
        adapter's :meth:`store_roots`.

        Args:
            session: The session to remove.
            dry_run: If True, report what *would* be removed without removing it.

        Returns:
            A :class:`DeleteResult` describing what was removed or skipped.
        """
        targets = list(session.paths) + self.correlated_paths(session)
        return self._delete_paths(targets, dry_run=dry_run)

    def delete_orphan(self, orphan: Orphan, *, dry_run: bool = False) -> DeleteResult:
        """Permanently delete an orphan's files, unless they changed since the scan.

        Each path is checked again at the moment of deletion — the preview
        included, so it says what will really happen. A path is refused when it
        changed since it was listed (:meth:`_orphan_drift`), when it was written
        in the last :data:`ORPHAN_QUIET_SECONDS`, when it no longer lies inside
        the store roots, or when the adapter's own re-check
        (:meth:`_orphan_revalidation`) finds it is no longer an orphan. What
        is left goes through the same guards as any delete.

        Args:
            orphan: The orphan to remove, as :meth:`find_orphans` listed it.
            dry_run: If True, report what would be removed without removing it.

        Returns:
            A :class:`DeleteResult`.
        """
        refused: dict[Path, str] = {}
        for path in orphan.paths:
            reason = self._orphan_drift(orphan, path)
            if reason is not None:
                refused[path] = reason
        if len(refused) < len(orphan.paths):
            for path, reason in self._orphan_revalidation(orphan).items():
                refused.setdefault(path, reason)
        allowed = [path for path in orphan.paths if path not in refused]
        result = self._delete_paths(allowed, dry_run=dry_run)
        result.skipped.update(refused)
        return result

    def fingerprint_orphans(self, orphans: list[Orphan]) -> list[Orphan]:
        """Fingerprint each orphan path that has none yet, if this adapter does.

        Called on the result of every :meth:`find_orphans`; adapters may call
        it earlier, right where an orphan is built, to narrow the window
        between deciding that something is debris and recording what it held.

        Args:
            orphans: The orphans a scan found.

        Returns:
            The same orphans, as a list.
        """
        found = list(orphans)
        if not self.fingerprints_orphans:
            return found
        for orphan in found:
            if orphan.fingerprints is None:
                orphan.fingerprints = {path: survey(path).fingerprint for path in orphan.paths}
        return found

    def _orphan_drift(self, orphan: Orphan, path: Path) -> str | None:
        """Return why one orphan path must not be deleted now, judged from disk.

        The reusable delete-time re-check: an adapter that overrides
        :meth:`delete_orphan` calls this for each path before its own checks.

        Args:
            orphan: The orphan as it was listed.
            path: One of its paths.

        Returns:
            ``"outside store root (refused)"`` when the path resolves outside
            the store roots now; ``"changed since it was listed (refused)"``
            when its fingerprint differs from the scan's; a "was modified in the
            last … seconds (refused)" reason when anything at or under it was
            written within :data:`ORPHAN_QUIET_SECONDS`; else ``None``. A path
            that is gone, or that the scan did not fingerprint, is left to the
            delete guards.
        """
        if not os.path.lexists(path):
            return None
        if not is_within(path, self.store_roots()):
            return "outside store root (refused)"
        recorded = orphan.fingerprints
        if recorded is None or path not in recorded:
            return None
        now = survey(path)
        if now.fingerprint != recorded[path]:
            return "changed since it was listed (refused)"
        quiet_since = time.time_ns() - ORPHAN_QUIET_SECONDS * 1_000_000_000
        if now.newest_ns is not None and now.newest_ns > quiet_since:
            return f"was modified in the last {ORPHAN_QUIET_SECONDS} seconds (refused)"
        return None

    def _orphan_revalidation(self, orphan: Orphan) -> dict[Path, str]:
        """Return the orphan's paths that are no longer orphans, with why.

        Fingerprints catch any change after the scan; this asks the adapter's
        own question again — is the project still gone, does the folder still
        hold no session — for what the scan itself might have raced.
        Default: nothing to add.

        Args:
            orphan: The orphan about to be deleted.

        Returns:
            Path -> refusal reason, for the paths that must stay.
        """
        return {}

    # --- project-scoped state --------------------------------------------

    def project_leftovers(self, project: str) -> ProjectLeftovers | None:
        """Describe this harness's project-scoped state for ``project``.

        Project-scoped state — memory, per-project settings, trust decisions —
        belongs to the directory rather than to any one session, so deleting a
        session never touches it. It is offered for removal only when the last
        session for that project is deleted.

        Args:
            project: The project directory.

        Returns:
            A :class:`ProjectLeftovers`, or ``None`` when this harness keeps no
            project-scoped state.
        """
        return None

    def delete_project_leftovers(
        self, leftovers: ProjectLeftovers, *, dry_run: bool = False
    ) -> DeleteResult:
        """Remove the state described by :meth:`project_leftovers`. Default: none."""
        return DeleteResult(dry_run=dry_run)

    def delete_paths(self, paths: list[Path], *, dry_run: bool = False) -> DeleteResult:
        """Remove specific paths through this adapter's guards.

        The public entry point for callers that already know exactly what they
        want removed — the memory browser, for instance — so they inherit the
        store-root allowlist, the protected-path refusals and the honest result
        reporting instead of reimplementing them.

        Args:
            paths: Files/dirs to remove.
            dry_run: If True, report what would be removed without removing it.

        Returns:
            A :class:`DeleteResult`.
        """
        return self._delete_paths(paths, dry_run=dry_run)

    def _delete_paths(self, paths: list[Path], *, dry_run: bool) -> DeleteResult:
        """Remove a list of paths, guarding each against the store roots.

        Existence is judged with ``lstat``: only ``ENOENT``/``ENOTDIR`` means a
        path is gone. Any other error — a parent ``sx`` cannot traverse — is a
        refusal, so it is reported rather than passed off as already deleted.
        A symbolic link is removed as a link, never followed into its target.

        Args:
            paths: Files/dirs to remove.
            dry_run: If True, only compute the result; remove nothing.

        Returns:
            A :class:`DeleteResult`. Paths outside the store roots are recorded
            in ``skipped`` and never touched.
        """
        roots = self.store_roots()
        protected = self.protected_paths()
        result = DeleteResult(dry_run=dry_run)
        seen: set[Path] = set()
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            try:
                info = os.lstat(path)
            except OSError as exc:
                if exc.errno in (errno.ENOENT, errno.ENOTDIR):
                    result.skipped[path] = "does not exist"
                else:
                    result.skipped[path] = f"cannot check: {exc} (refused)"
                continue
            except ValueError as exc:
                result.skipped[path] = f"cannot check: {exc} (refused)"
                continue
            if not is_within(path, roots):
                result.skipped[path] = "outside store root (refused)"
                continue
            if any(_same_path(path, root) for root in roots):
                # A store root is "within itself", so the guard above would let
                # it through and rmtree the entire harness store.
                result.skipped[path] = "target is a store root (refused)"
                continue
            if protected and is_within(path, protected):
                # Inside a store root, but explicitly off limits — bulk the
                # harness owns that happens to share a directory with the
                # session-keyed entries the cascade needs to reach.
                result.skipped[path] = "protected path (refused)"
                continue
            link = stat.S_ISLNK(info.st_mode)
            size = info.st_size if link else dir_size(path)
            if dry_run:
                result.removed.append(path)
                result.freed_bytes += size
                continue
            try:
                if stat.S_ISDIR(info.st_mode):
                    shutil.rmtree(path)
                else:
                    path.unlink()
                result.removed.append(path)
                result.freed_bytes += size
            except OSError as exc:
                result.skipped[path] = f"error: {exc}"
        return result


class JsonlFolderAdapter(HarnessAdapter):
    """Base for harnesses that store sessions as JSONL files in folders.

    Subclasses typically override only:
      * :meth:`session_files` — where the ``.jsonl`` files are;
      * :meth:`parse_line` — how to turn one JSON object into a :class:`Message`;
      * :meth:`group_key` — the project path a file belongs to;
      * :meth:`title_for` / :meth:`session_id_for` — display niceties.

    The default :attr:`capabilities` cover browse + orphan detection + delete.
    """

    capabilities = (
        Capability.BROWSE | Capability.ORPHANS | Capability.DELETE | Capability.MOVE
    )
    fingerprints_orphans = True

    # --- discovery -------------------------------------------------------

    def session_files(self) -> Iterator[Path]:
        """Yield every session ``.jsonl`` file under the store roots.

        Default implementation globs ``*.jsonl`` recursively in each root.
        Override for harnesses with a more specific layout.
        """
        for root in self.store_roots():
            if not root.exists():
                continue
            yield from sorted(root.rglob("*.jsonl"))

    def discover(self) -> Iterator[Session]:
        """Yield a :class:`Session` per session file, with cheap metadata."""
        for path in self.session_files():
            session = self._session_from_file(path)
            if session is not None:
                yield session

    def _session_from_file(self, path: Path) -> Session | None:
        """Build a metadata-only :class:`Session` from one file.

        Reads only the first line (for grouping/id) and stats the file; the full
        transcript is left for :meth:`load`. A session is an orphan only when
        :func:`~sx.fsguard.project_state` says its project is definitely
        missing — never when it merely cannot be checked.

        Args:
            path: A session file.

        Returns:
            The session, or ``None`` when the file is not one.
        """
        first = read_first_line(path)
        project = self.group_key(path, first)
        session_id = self.session_id_for(path, first)
        size = 0
        try:
            size = path.stat().st_size
        except OSError:
            pass
        session = Session(
            harness=self.name,
            session_id=session_id,
            project_path=project,
            title=self.title_for(path, first),
            modified=file_mtime(path),
            size_bytes=size,
            paths=[path],
            is_orphan=bool(project) and project_state(project) == MISSING,
        )
        return session

    # --- per-harness hooks ----------------------------------------------

    def group_key(self, path: Path, first: dict | None) -> str | None:
        """Return the project path a session file belongs to. Override me."""
        return None

    def session_id_for(self, path: Path, first: dict | None) -> str:
        """Return a stable id for the session. Default: filename stem."""
        return path.stem

    def title_for(self, path: Path, first: dict | None) -> str:
        """Return a display title. Default: filename stem."""
        return path.stem

    def parse_line(self, obj: dict) -> Message | None:
        """Turn one JSON object into a :class:`Message`, or ``None`` to skip.

        Override me — this is the heart of each harness's transcript format.
        """
        return None

    # --- loading ---------------------------------------------------------

    def load(self, session: Session) -> list[Message]:
        """Parse the full transcript for ``session`` via :meth:`parse_line`."""
        messages: list[Message] = []
        if session.primary_path is None:
            return messages
        for obj in iter_jsonl(session.primary_path):
            msg = self.parse_line(obj)
            if msg is not None:
                messages.append(msg)
        return messages

    # --- move ------------------------------------------------------------

    def repoint_record(self, obj: dict, old: Path, new: Path) -> dict | None:
        """Return ``obj`` with its recorded project path re-pointed, or ``None``.

        This is the one place a harness declares *which field* holds the working
        directory. Only that structural field may be touched: free text elsewhere
        in a record (tool output, a pasted diff, the user's own prose) is
        historical record, and rewriting paths inside it would corrupt the
        transcript rather than move it.

        Args:
            obj: One parsed transcript record.
            old: The project directory being moved away from.
            new: The project directory being moved to.

        Returns:
            A replacement record, or ``None`` to leave the line untouched.
        """
        return None

    def move_transform(
        self, path: Path, plan: MovePlan, *, moved_from: Path | None
    ) -> Callable[[dict], dict | None]:
        """Return the record transform that re-points one transcript.

        The hook a harness overrides when a move rewrites more than one field
        per record — a session header that also records where the file used to
        be, say. The transform is handed to :func:`sx.util.rewrite_jsonl`, so it
        must return a new mapping (or ``None``) and never edit the one it is
        given: a move records each change it makes so a failed relocation can
        put the transcript back exactly.

        Args:
            path: The transcript about to be rewritten, at its current path.
            plan: The move being carried out.
            moved_from: ``path`` itself when the same move also relocates the
                transcript, else ``None``.

        Returns:
            A transform for :func:`sx.util.rewrite_jsonl`. The default applies
            :meth:`repoint_record` to every record.
        """
        old, new = plan.old, plan.new

        def transform(obj: dict) -> dict | None:
            """Re-point one record's working directory, if it has one."""
            return self.repoint_record(obj, old, new)

        return transform

    def plan_move(self, sessions: list[Session], old: Path, new: Path) -> MovePlan:
        """Plan a move as an in-place rewrite of each session's own files.

        Harnesses whose files also have to be relocated (their location encodes
        the project path) extend this.

        Args:
            sessions: Sessions selected for the move.
            old: The project directory being moved away from.
            new: The project directory being moved to.

        Returns:
            A :class:`MovePlan` listing the files to rewrite.
        """
        plan = MovePlan(harness=self.name, old=old, new=new, sessions=list(sessions))
        roots = self.store_roots()
        seen: set[Path] = set()
        for session in sessions:
            for path in session.paths:
                if path in seen:
                    continue
                seen.add(path)
                if not path.exists():
                    continue
                if not is_within(path, roots):
                    plan.blocked[path] = "outside store root (refused)"
                    continue
                plan.rewrites.append(path)
        return plan

    def move(self, plan: MovePlan, *, dry_run: bool = False) -> MoveResult:
        """Rewrite every file in ``plan``, re-pointing its recorded directory.

        Args:
            plan: The plan produced by :meth:`plan_move`.
            dry_run: If True, count what would change without writing.

        Returns:
            A :class:`MoveResult`.
        """
        result = MoveResult(dry_run=dry_run)
        result.skipped.update(plan.blocked)
        self._rewrite_all(plan, result, dry_run=dry_run)
        return result

    def _rewrite_all(self, plan: MovePlan, result: MoveResult, *, dry_run: bool) -> None:
        """Rewrite every file in ``plan.rewrites`` through :meth:`move_transform`.

        Args:
            plan: The plan being executed.
            result: The result to record outcomes in (mutated in place).
            dry_run: If True, count without writing.
        """
        for path in plan.rewrites:
            transform = self.move_transform(path, plan, moved_from=None)
            changed, error = rewrite_jsonl(path, transform, dry_run=dry_run)
            if error is not None:
                result.skipped[path] = error
                continue
            if changed:
                result.rewritten.append(path)
                result.fields_updated += changed


# --- the moving parts of an encoded-folder move ---------------------------------


@dataclass(slots=True)
class _Candidate:
    """Something a move wants to relocate, before it is planned in detail.

    Attributes:
        kind: :attr:`MoveUnit.FOLDER` or :attr:`MoveUnit.SESSION`.
        folder: The folder it comes from.
        destination: The folder it goes to.
        sessions: The selected sessions it carries.
    """

    kind: str
    folder: Path
    destination: Path
    sessions: list[Session]


class _UnitFailed(Exception):
    """Raised inside a unit to abandon it; carries the reasons, by path."""

    def __init__(self, reasons: dict[Path, str]) -> None:
        """Store the reasons."""
        super().__init__(reasons)
        self.reasons = reasons


class EncodedFolderAdapter(JsonlFolderAdapter):
    """Base for harnesses that keep one folder per project, named after its path.

    Claude Code, Pi and Oh My Pi store a project's transcripts in a folder whose
    *name* is derived from the project directory. Re-pointing such a session is
    therefore two operations — rewrite the directory recorded in the transcript,
    then relocate the files into the folder the new directory encodes to — and
    the relocation half lives here so every such harness moves the same way.

    The two halves are never decided apart. A move is cut into
    :class:`~sx.model.MoveUnit` pieces — a folder renamed whole, or one session's
    transcript with its sidecars — and each piece re-points and relocates
    together or not at all: a transcript re-pointed but left in the old folder
    is invisible to the harness, and one relocated with its old directory still
    recorded looks like a dead project's.

    The encodings are one-way and lossy (``/a-b`` and ``/a/b`` produce the same
    name), so folder names are never decoded. Candidate directories are encoded
    and compared instead, and a folder is only moved whole when nothing in it
    belongs to another project.

    Subclasses implement :meth:`sessions_root` and :meth:`folder_for`, and
    override :meth:`folder_names`, :meth:`session_sidecars`,
    :meth:`move_transform`, :meth:`destructive_refusal` and
    :meth:`busy_transcript` when their layout or format needs it.
    """

    def sessions_root(self) -> Path:
        """Return the directory holding the per-project folders."""
        raise NotImplementedError

    def folder_for(self, project: str) -> str:
        """Return the folder name the harness uses today for ``project``."""
        raise NotImplementedError

    def folder_names(self, project: str) -> set[str]:
        """Return every folder name that may hold ``project``'s sessions.

        Defaults to the current name only. Harnesses that changed their encoding
        and migrate lazily add the older names, so a folder created under the old
        scheme still counts as belonging to the project.

        Args:
            project: A project directory.

        Returns:
            The folder names, current one included.
        """
        return {self.folder_for(project)}

    def session_sidecars(self, session: Session) -> list[Path]:
        """Return per-session entries that travel with the transcript.

        Args:
            session: The session being relocated.

        Returns:
            Existing paths beside the transcript that belong to this session
            alone. Default: none.
        """
        return []

    def recorded_project(self, path: Path) -> str | None:
        """Return the project directory recorded in one transcript, if readable."""
        return self.group_key(path, read_first_line(path))

    # --- orphans ---------------------------------------------------------

    def _orphan_revalidation(self, orphan: Orphan) -> dict[Path, str]:
        """Ask again whether each project folder or fragment is still an orphan.

        An empty folder that now holds a transcript, a dead project's folder
        holding a transcript whose project is back (or cannot be checked), and
        a fragment that now records a project are all refused.

        Args:
            orphan: The orphan about to be deleted.

        Returns:
            Path -> refusal reason, for the paths that must stay.
        """
        refused: dict[Path, str] = {}
        root = self.sessions_root()
        for path in orphan.paths:
            if path.parent == root and not path.is_symlink() and path.is_dir():
                reason = self._folder_revalidation(path, orphan.kind)
            elif orphan.kind is OrphanKind.STRAY_TEMP and is_transcript(path):
                project = self.recorded_project(path) if path.is_file() else None
                reason = "now holds a session (refused)" if project else None
            else:
                reason = None
            if reason is not None:
                refused[path] = reason
        return refused

    def _folder_revalidation(self, folder: Path, kind: OrphanKind) -> str | None:
        """Return why a project folder offered for cleanup must stay after all.

        Args:
            folder: A folder directly under :meth:`sessions_root`.
            kind: Why it was offered.

        Returns:
            A refusal reason, or ``None``.
        """
        try:
            transcripts = sorted(p for p in folder.iterdir() if is_transcript(p))
        except OSError as exc:
            return f"cannot read: {exc} (refused)"
        if kind is OrphanKind.EMPTY and transcripts:
            return f"now holds a session ({transcripts[0].name}) (refused)"
        if kind is not OrphanKind.DEAD_PROJECT:
            return None
        for path in transcripts:
            project = self.recorded_project(path)
            if project is not None and project_state(project) != MISSING:
                return f"{path.name}'s project directory is there, or cannot be checked (refused)"
        return None

    # --- resuming --------------------------------------------------------

    def stranded_sessions(
        self,
        old: Path,
        new: Path,
        sessions: list[Session] | None = None,
    ) -> list[Session]:
        """Return sessions already re-pointed under ``new`` but still in ``old``'s folders.

        An interrupted move — or a version of ``sx`` that rewrote a transcript
        and then could not relocate it — leaves the harness unable to see the
        session: its header names the new directory, its file sits in the folder
        named for the old one. These are handed to :meth:`plan_move`, which
        relocates them without rewriting them again.

        Args:
            old: The project directory being moved away from.
            new: The project directory being moved to.
            sessions: An already-discovered pool; when omitted the adapter
                discovers its own.

        Returns:
            Sessions recorded under ``new`` (and not under ``old``) whose file is
            in a folder named for the matching directory under ``old``.
        """
        pool = list(self.discover()) if sessions is None else sessions
        root = self.sessions_root()
        found: list[Session] = []
        for session in pool:
            project = session.project_path
            path = session.primary_path
            if session.harness != self.name or not project or path is None:
                continue
            if is_under(project, old) or not is_under(project, new):
                continue
            folder = path.parent
            if folder.parent != root or folder.name in self.folder_names(project):
                continue
            before = repoint(project, new, old)
            if before is not None and folder.name in self.folder_names(before):
                found.append(session)
        return found

    @staticmethod
    def _target_dir(session: Session, old: Path, new: Path) -> str | None:
        """Return the directory ``session`` is recorded under once the move is done.

        Args:
            session: A selected session.
            old: The project directory being moved away from.
            new: The project directory being moved to.

        Returns:
            Its re-pointed project; its own project when an interrupted move
            already re-pointed it under ``new``; ``None`` when it has none.
        """
        project = session.project_path
        if not project:
            return None
        moved = repoint(project, old, new)
        if moved is not None:
            return moved
        return project if is_under(project, new) else None

    @staticmethod
    def _origin_dir(session: Session, old: Path, new: Path) -> str | None:
        """Return the directory ``session``'s folder was named for before the move.

        Args:
            session: A selected session.
            old: The project directory being moved away from.
            new: The project directory being moved to.

        Returns:
            Its project, or for a stranded session the directory under ``old``
            its project was re-pointed from.
        """
        project = session.project_path
        if project and not is_under(project, old) and is_under(project, new):
            return repoint(project, new, old)
        return project

    # --- planning --------------------------------------------------------

    def plan_move(self, sessions: list[Session], old: Path, new: Path) -> MovePlan:
        """Plan the rewrite *and* the relocation an encoded-folder layout requires.

        A folder is relocated whole whenever its name matches the directory being
        moved and every transcript in it belongs to that directory — which also
        carries along whatever else the folder holds (Claude's ``memory/``, each
        session's sidecar) without enumerating it. Transcripts that merely
        record some other directory first, whose path encodes elsewhere, travel
        with it as they always have.

        Sessions whose folder is named for some other directory, or whose folder
        also holds another project's transcripts (two paths that encode alike),
        are relocated individually, transcript and sidecars together; a shared
        folder and everything else in it stays where it is.

        Everything is planned against the store as the earlier steps of the
        same move will leave it: a folder that is itself moving is vacated
        before anything moves into it, and a destination an earlier step
        creates is merged into entry by entry. Each step becomes a
        :class:`~sx.model.MoveUnit`; a unit that cannot be planned whole —
        a name taken at the destination, say — is refused whole, and its
        transcripts are not re-pointed either.

        Args:
            sessions: Sessions selected for the move. A session an interrupted
                move already re-pointed under ``new`` (see
                :meth:`stranded_sessions`) is relocated without being
                rewritten again.
            old: The project directory being moved away from.
            new: The project directory being moved to.

        Returns:
            A :class:`MovePlan` covering rewrites, relocations and refusals.
        """
        plan = super().plan_move(sessions, old, new)
        namespace = self._namespace_of(plan)
        root = self.sessions_root()
        roots = self.store_roots()
        rewrites = set(plan.rewrites)

        by_folder: dict[Path, list[Session]] = {}
        for session in sessions:
            path = session.primary_path
            if path is not None:
                by_folder.setdefault(path.parent, []).append(session)

        units: list[MoveUnit] = []
        candidates: list[_Candidate] = []
        strays = shared = foreign_total = 0
        for folder, members in sorted(by_folder.items()):
            movable = [s for s in members if s.primary_path in rewrites]
            # A selected session whose own rewrite is refused (it lies outside
            # the store, say) never moves, so its folder cannot move whole.
            pinned = [s.primary_path for s in members if s.primary_path in plan.blocked]
            if not movable:
                continue
            if folder.parent != root:
                units.extend(self._rewrite_unit(s) for s in movable)  # unfamiliar layout
                continue
            dirs = [str(old)] + [
                d for s in movable for d in (s.project_path, self._origin_dir(s, old, new)) if d
            ]
            implied = self._folder_project(folder, dirs)
            if implied is not None and is_under(implied, old):
                stay = set(self._foreign_transcripts(folder, movable, old)) | set(pinned)
                if not stay:
                    target = repoint(implied, old, new)
                    destination = root / self.folder_for(target)
                    if destination == folder:
                        # Old and new encode alike: nothing to relocate.
                        units.extend(self._rewrite_unit(s) for s in movable)
                    else:
                        candidates.append(
                            _Candidate(MoveUnit.FOLDER, folder, destination, movable)
                        )
                    continue
                foreign_total += len(stay)
                for session in movable:
                    candidate = self._session_candidate(session, folder, old, new)
                    if candidate is None:
                        units.append(self._rewrite_unit(session))
                    else:
                        candidates.append(candidate)
                        shared += 1
                continue
            for session in movable:
                candidate = self._session_candidate(session, folder, old, new)
                if candidate is None:
                    units.append(self._rewrite_unit(session))
                else:
                    candidates.append(candidate)
                    strays += 1

        ordered, cyclic = self._ordered(candidates)
        for candidate in cyclic:
            for session in candidate.sessions:
                self._refuse_rewrite(
                    plan,
                    session.primary_path,
                    "would trade places with another folder in the same move (refused)",
                )
        merged_into: set[Path] = set()
        for candidate in ordered:
            if candidate.kind == MoveUnit.FOLDER:
                found = self._plan_folder(candidate, plan, roots)
            else:
                found = self._plan_session(candidate, plan, roots)
            if any(unit.kind != MoveUnit.FOLDER for unit in found):
                merged_into.add(candidate.destination)  # entries go into a folder there
            self._apply(found, namespace)
            units.extend(found)

        plan.units = units
        kept = {path for unit in units for path in unit.rewrites}
        plan.rewrites = [path for path in plan.rewrites if path in kept]

        notes: list[str] = []
        if strays:
            notes.append(
                f"{strays} session(s) live in a folder named for another "
                "directory and move individually"
            )
        if shared:
            notes.append(
                f"{shared} session(s) move individually: their folder also holds "
                f"{foreign_total} session(s) that stay put (another project's, in "
                "use, or not safe to rewrite), so the folder and everything else in "
                "it stay in place"
            )
        waiting = sum(
            1
            for destination in merged_into
            for name in namespace.preexisting(destination)
            if is_transcript(destination / name)
        )
        if waiting:
            notes.append(
                f"{waiting} session(s) already at the destination; a later inverse "
                "move will move them too"
            )
        if notes:
            plan.note = "; ".join(notes)
        return plan

    @staticmethod
    def _namespace_of(plan: MovePlan) -> Namespace:
        """Return the plan's planning namespace, starting one if it has none."""
        namespace = plan._namespace
        if not isinstance(namespace, Namespace):
            namespace = Namespace()
            plan._namespace = namespace
        return namespace

    @staticmethod
    def _rewrite_unit(session: Session) -> MoveUnit:
        """Return the unit that re-points a session's transcript where it is."""
        path = session.primary_path
        return MoveUnit(MoveUnit.REWRITE, rewrites=[path] if path else [], sessions=[session])

    def _session_candidate(
        self, session: Session, folder: Path, old: Path, new: Path
    ) -> _Candidate | None:
        """Return the relocation one session needs on its own, if any.

        Args:
            session: A selected session.
            folder: The folder holding it.
            old: The project directory being moved away from.
            new: The project directory being moved to.

        Returns:
            A session candidate, or ``None`` when the session is already in the
            folder its re-pointed project encodes to (or records no project).
        """
        target = self._target_dir(session, old, new)
        if target is None:
            return None
        destination = self.sessions_root() / self.folder_for(target)
        if destination == folder:
            return None
        return _Candidate(MoveUnit.SESSION, folder, destination, [session])

    @staticmethod
    def _ordered(candidates: list[_Candidate]) -> tuple[list[_Candidate], list[_Candidate]]:
        """Order candidates so a folder is vacated before anything moves into it.

        A folder merged into a bucket that is itself relocated later in the same
        move would otherwise ride along with it, one level too far. Whole
        folders go before single sessions where the dependencies allow, so a
        session that strayed into another folder merges into its project's
        renamed folder rather than forcing that folder to merge.

        Args:
            candidates: The relocations a move wants, in folder order.

        Returns:
            ``(ordered, cyclic)``: the candidates in a safe order, and those
            caught in a cycle (two folders trading names), which cannot move.
        """
        vacating = {
            c.folder: index for index, c in enumerate(candidates) if c.kind == MoveUnit.FOLDER
        }
        waits_for: dict[int, int] = {}
        for index, candidate in enumerate(candidates):
            blocker = vacating.get(candidate.destination)
            if blocker is not None and blocker != index:
                waits_for[index] = blocker
        done: set[int] = set()
        ordered: list[_Candidate] = []

        def rank(index: int) -> tuple[int, int]:
            """Sort whole folders first, then single sessions, each in folder order."""
            return (0 if candidates[index].kind == MoveUnit.FOLDER else 1, index)

        pending = sorted(range(len(candidates)), key=rank)
        progress = True
        while pending and progress:
            progress = False
            for index in pending:
                blocker = waits_for.get(index)
                if blocker is None or blocker in done:
                    ordered.append(candidates[index])
                    done.add(index)
                    pending.remove(index)
                    progress = True
                    break
        return ordered, [candidates[index] for index in pending]

    def _plan_folder(self, candidate: _Candidate, plan: MovePlan, roots: list[Path]) -> list[MoveUnit]:
        """Plan one folder's relocation and cut it into units.

        Args:
            candidate: The folder and where it goes.
            plan: The plan to record into.
            roots: The adapter's store roots, for the containment guard.

        Returns:
            The units; a folder renamed whole is one unit.
        """
        mark = len(plan.relocations)
        self._plan_relocation(candidate.folder, candidate.destination, plan, roots)
        pairs = plan.relocations[mark:]
        return self._folder_units(
            candidate.folder, candidate.destination, pairs, candidate.sessions, plan
        )

    def _folder_units(
        self,
        folder: Path,
        destination: Path,
        pairs: list[tuple[Path, Path]],
        sessions: list[Session],
        plan: MovePlan,
    ) -> list[MoveUnit]:
        """Group what planning one folder produced into atomic units.

        A single ``folder -> destination`` pair is one unit carrying every
        selected rewrite. Entry-by-entry pairs — a merge, or a harness that
        moves its folder around files that must stay — are grouped per session:
        its transcript and sidecars form one unit, and if any of them cannot
        move, none of them does and the transcript is not re-pointed. Anything
        else (a memory folder, say) moves as its own unit, ahead of the
        sessions, so a move interrupted halfway still leaves sessions behind for
        a re-run to find.

        Args:
            folder: The folder being relocated.
            destination: Where it goes.
            pairs: The relocations planning it appended; anything it planned
                no relocation for (a name taken at the destination, a file the
                harness keeps in place) is missing from them.
            sessions: The selected sessions in the folder.
            plan: The plan, adjusted in place for refused groups.

        Returns:
            The units, in the order they run.
        """
        selected = {s.primary_path: s for s in sessions if s.primary_path is not None}
        rewrites = set(plan.rewrites)
        if pairs == [(folder, destination)]:
            paths = sorted(selected)
            return [
                MoveUnit(
                    MoveUnit.FOLDER,
                    relocations=[(folder, destination)],
                    rewrites=[path for path in paths if path in rewrites],
                    sessions=[selected[path] for path in paths],
                )
            ]
        if not pairs:
            for path in selected:
                self._refuse_rewrite(plan, path, _FOLDER_STAYS)
            return []
        owners = self._entry_owners(folder, selected)
        groups: dict[Path, list[tuple[Path, Path]]] = {}
        units: list[MoveUnit] = []
        for pair in pairs:
            owner = owners.get(pair[0])
            if owner is None:
                units.append(MoveUnit(MoveUnit.ENTRY, relocations=[pair]))
            else:
                groups.setdefault(owner, []).append(pair)
        for transcript in sorted(set(owners.values())):
            members = [path for path, owner in owners.items() if owner == transcript]
            planned = groups.get(transcript, [])
            sources = {src for src, _ in planned}
            session = selected.get(transcript)
            if all(path in sources for path in members):
                units.append(
                    MoveUnit(
                        MoveUnit.SESSION,
                        relocations=sorted(planned, key=lambda pair: pair[0] == transcript),
                        rewrites=[transcript] if transcript in rewrites and session else [],
                        sessions=[session] if session else [],
                    )
                )
            elif planned or session is not None:
                self._refuse_group(plan, transcript, members, planned)
        return units

    def _entry_owners(self, folder: Path, selected: dict[Path, Session]) -> dict[Path, Path]:
        """Map each transcript in ``folder``, and each of its sidecars, to the transcript.

        Args:
            folder: A project folder.
            selected: The selected sessions in it, by transcript.

        Returns:
            Entry path -> the transcript it belongs to.
        """
        try:
            transcripts = {p for p in folder.iterdir() if is_transcript(p)}
        except OSError:
            transcripts = set()
        owners: dict[Path, Path] = {}
        for transcript in sorted(transcripts | set(selected)):
            session = selected.get(transcript) or Session(
                harness=self.name, session_id=transcript.stem, paths=[transcript]
            )
            owners[transcript] = transcript
            for sidecar in self.session_sidecars(session):
                owners.setdefault(sidecar, transcript)
        return owners

    def _plan_session(
        self, candidate: _Candidate, plan: MovePlan, roots: list[Path]
    ) -> list[MoveUnit]:
        """Plan one session's transcript and sidecars into another folder, as one unit.

        Args:
            candidate: The session, its folder and its destination folder.
            plan: The plan to record into.
            roots: The adapter's store roots, for the containment guard.

        Returns:
            The unit, or nothing when any of its files cannot move — its
            destination folder is a symbolic link, say.
        """
        session = candidate.sessions[0]
        path = session.primary_path
        mark = len(plan.relocations)
        members = [*self.session_sidecars(session), path]
        refusal = (
            self._folder_refusal(
                candidate.destination, self._namespace_of(plan), (self.sessions_root(), *roots)
            )
            if is_within(candidate.destination, roots)
            else None  # left to _plan_entry, which refuses it as outside the store
        )
        if refusal is not None:
            plan.blocked.setdefault(path, refusal)
            self._refuse_group(plan, path, members, [])
            return []
        for member in members:
            self._plan_entry(member, candidate.destination / member.name, plan, roots)
        pairs = plan.relocations[mark:]
        if len(pairs) == len(members):
            return [
                MoveUnit(
                    MoveUnit.SESSION,
                    relocations=pairs,
                    rewrites=[path] if path in plan.rewrites else [],
                    sessions=[session],
                )
            ]
        self._refuse_group(plan, path, members, pairs)
        return []

    @staticmethod
    def _refuse_group(
        plan: MovePlan, transcript: Path, members: list[Path], pairs: list[tuple[Path, Path]]
    ) -> None:
        """Keep a whole session in place because one of its files cannot move.

        Args:
            plan: The plan to adjust.
            transcript: The session's transcript.
            members: The transcript and its sidecars.
            pairs: The relocations already planned for them, now withdrawn.
        """
        withdrawn = set(pairs)
        plan.relocations = [pair for pair in plan.relocations if pair not in withdrawn]
        for member in members:
            plan.blocked.setdefault(member, _STAYS_WHOLE)
        if transcript in plan.rewrites:
            plan.rewrites.remove(transcript)

    @staticmethod
    def _refuse_rewrite(plan: MovePlan, path: Path | None, reason: str) -> None:
        """Withdraw a transcript's rewrite and record why.

        Args:
            plan: The plan to adjust.
            path: The transcript.
            reason: The refusal.
        """
        if path is None:
            return
        plan.blocked.setdefault(path, reason)
        if path in plan.rewrites:
            plan.rewrites.remove(path)

    @staticmethod
    def _folder_refusal(
        folder: Path, namespace: Namespace | None, exempt: tuple[Path, ...]
    ) -> str | None:
        """Return why nothing may be moved into ``folder``, if nothing may.

        A destination folder must be a real directory, or not exist yet. A
        symbolic link would carry whatever moves into it to wherever it points
        — another project's folder, say — where the harness will never look
        for it.

        Args:
            folder: The folder something is about to be moved into.
            namespace: The plan's or the dry run's view of the store, or
                ``None`` to ask the disk.
            exempt: Folders never judged — the sessions root and the store
                roots, which a user may keep behind a link.

        Returns:
            A refusal reason, or ``None``.
        """
        if folder in exempt:
            return None
        if namespace is not None:
            if not namespace.lexists(folder) or namespace.is_dir(folder):
                return None
        else:
            try:
                info = os.lstat(folder)
            except FileNotFoundError:
                return None
            except (OSError, ValueError) as exc:
                return f"cannot check the destination folder {folder}: {exc} (refused)"
            if stat.S_ISDIR(info.st_mode):
                return None
        return f"the destination folder {folder} is a symbolic link or not a directory (refused)"

    @staticmethod
    def _apply(units: list[MoveUnit], namespace: Namespace) -> None:
        """Record planned units in the namespace, so later steps see their effect."""
        for unit in units:
            for src, dst in unit.relocations:
                if not namespace.lexists(dst.parent):
                    namespace.mkdir(dst.parent)
                namespace.rename(src, dst)

    def _folder_project(self, folder: Path, candidates: list[str]) -> str | None:
        """Return the project directory a store folder is named for.

        Each candidate is *encoded* and compared with the folder name, which is
        exact; the name itself is never decoded.

        Args:
            folder: A folder directly under :meth:`sessions_root`.
            candidates: Directories that might have produced the folder name.

        Returns:
            The matching candidate, or ``None`` when the folder is named for some
            other directory (a project renamed after the folder was created, for
            instance).
        """
        for candidate in candidates:
            if candidate and folder.name in self.folder_names(candidate):
                return candidate
        return None

    def _foreign_transcripts(
        self, folder: Path, selected: list[Session], old: Path
    ) -> list[Path]:
        """Return transcripts in ``folder`` that belong to a colliding project.

        Two project paths that differ only in punctuation encode to the same
        folder, so a folder named for the project being moved can still hold
        another project's sessions. Moving that folder whole would carry them
        off to a folder named for a directory they have nothing to do with.

        A transcript is foreign only when its recorded project encodes to this
        same folder name and is not under ``old`` — a real collision. One whose
        project encodes elsewhere (it recorded some other directory first) or
        cannot be read is not counted: it cannot be attributed to anyone else
        here, and it has always traveled with its folder. Subclasses extend
        this with transcripts that must stay for other reasons (one the harness
        has open, for instance).

        Args:
            folder: The folder about to be relocated.
            selected: The sessions being moved out of it.
            old: The project directory being moved away from.

        Returns:
            The transcripts that must not travel with the folder.
        """
        chosen = {s.primary_path for s in selected}
        foreign: list[Path] = []
        for path in sorted(folder.glob("*.jsonl")):
            if path in chosen:
                continue
            project = self.recorded_project(path)
            if (
                project
                and folder.name in self.folder_names(project)
                and not is_under(project, old)
            ):
                foreign.append(path)
        return foreign

    def _plan_relocation(
        self, src: Path, dst: Path, plan: MovePlan, roots: list[Path]
    ) -> None:
        """Plan moving a whole project folder, merging when the target will exist.

        A destination folder already exists whenever the harness has been run at
        the new path — or when an earlier step of the same move creates it. Its
        entries are then merged one by one, and any name already taken is
        refused rather than overwritten — a move must never destroy a
        transcript that is already there.

        Args:
            src: The folder to move.
            dst: Where it should end up.
            plan: The plan to record into.
            roots: The adapter's store roots, for the containment guard.
        """
        if src == dst:
            return
        namespace = self._namespace_of(plan)
        if not namespace.lexists(dst):
            self._plan_entry(src, dst, plan, roots)
            return
        if not is_within(dst, roots):
            plan.blocked[src] = "outside store root (refused)"
            return
        if not namespace.is_dir(dst):
            plan.blocked[dst] = "destination exists and is not a directory (refused)"
            return
        names = namespace.names(src)
        if not isinstance(names, set):
            plan.blocked[src] = f"cannot read: {src}"
            return
        for name in sorted(names):
            self._plan_entry(src / name, dst / name, plan, roots)

    def _plan_entry(self, src: Path, dst: Path, plan: MovePlan, roots: list[Path]) -> None:
        """Record one ``src -> dst`` relocation, or why it cannot happen.

        Args:
            src: The path to move.
            dst: Where it goes.
            plan: The plan to record into; its namespace decides what will exist.
            roots: The adapter's store roots, for the containment guard.
        """
        if not is_within(src, roots) or not is_within(dst, roots):
            plan.blocked[src] = "outside store root (refused)"
            return
        namespace = self._namespace_of(plan)
        if not namespace.readable(dst.parent):
            plan.blocked[src] = f"cannot read the destination folder {dst.parent} (refused)"
            return
        if namespace.lexists(dst):
            plan.blocked[src] = (
                f"{dst} is this same file under another spelling (refused)"
                if same_entry(src, dst)
                else f"already exists at destination: {dst} (refused)"
            )
            return
        plan.relocations.append((src, dst))

    # --- executing -------------------------------------------------------

    def move(self, plan: MovePlan, *, dry_run: bool = False) -> MoveResult:
        """Carry out a plan one atomic unit at a time.

        Each unit is checked again right before it runs — its destination still
        free and its destination folder a real directory, its files still
        inside the store, its transcripts still regular files and none of them
        busy, and none of its sessions refused by :meth:`destructive_refusal` —
        then its transcripts are re-pointed and its files renamed. If any step
        fails, or Ctrl-C cuts it short, the unit is put back: renames undone,
        rewrites reversed. A folder whose destination appeared since planning
        is merged into it instead; one whose "destination" is the folder itself
        under another spelling — a case-only rename on a case-insensitive file
        system — is renamed in place.

        Args:
            plan: The plan produced by :meth:`plan_move`.
            dry_run: If True, report what would happen and change nothing.

        Returns:
            A :class:`MoveResult` describing what actually happened. The plan's
            note describes the work, not work done, so a real move that changed
            nothing leaves it out — otherwise a move refused from end to end
            would not count as :attr:`~sx.model.MoveResult.failed`.
        """
        result = MoveResult(dry_run=dry_run)
        result.skipped.update(plan.blocked)
        simulation = Namespace() if dry_run else None
        for unit in self._execution_units(plan, result):
            self._run_unit(unit, plan, result, simulation)
        if plan.note and (dry_run or result.moved or result.rewritten):
            result.note = plan.note
        if not dry_run:
            self._prune_empty_folders(plan, result.moved)
        return result

    def _execution_units(self, plan: MovePlan, result: MoveResult) -> list[MoveUnit]:
        """Return the units to run, reconciled with any later change to the plan.

        A subclass may withdraw relocations or rewrites after planning; the
        unit they belong to is then refused whole. Relocations or rewrites a
        subclass added on its own become units of their own. The order keeps
        every folder vacated before anything moves into it.

        Relocations a subclass planned on its own — Oh My Pi's memory folders,
        kept apart from the session folders — run ahead of the session units,
        in their own order: they carry no session, so if they ran last, a move
        interrupted between the two would leave them with no session behind
        for a re-run to find.

        Args:
            plan: The plan being executed.
            result: Where refusals are recorded.

        Returns:
            The units, in the order they run.
        """
        listed = set(plan.relocations)
        rewrites = set(plan.rewrites)
        covered_pairs: set[tuple[Path, Path]] = set()
        covered_rewrites: set[Path] = set()
        units: list[MoveUnit] = []
        for unit in plan.units:
            covered_pairs.update(unit.relocations)
            covered_rewrites.update(unit.rewrites)
            members = [src for src, _ in unit.relocations] + list(unit.rewrites)
            withdrawn = [src for src, dst in unit.relocations if (src, dst) not in listed]
            withdrawn += [path for path in unit.rewrites if path not in rewrites]
            if withdrawn:
                reason = next(
                    (plan.blocked[path] for path in members if path in plan.blocked),
                    _STAYS_WHOLE,
                )
                for path in members:
                    result.skipped.setdefault(path, reason)
                continue
            units.append(unit)
        own: list[MoveUnit] = []
        for pair in plan.relocations:
            if pair not in covered_pairs:
                own.append(MoveUnit(MoveUnit.ENTRY, relocations=[pair]))
                covered_pairs.add(pair)
        units = own + units
        for path in plan.rewrites:
            if path not in covered_rewrites:
                units.append(MoveUnit(MoveUnit.REWRITE, rewrites=[path]))
                covered_rewrites.add(path)
        return self._vacate_first(units, result)

    @staticmethod
    def _vacate_first(units: list[MoveUnit], result: MoveResult) -> list[MoveUnit]:
        """Order units so nothing moves into a path before it is vacated.

        Args:
            units: The units, in plan order.
            result: Where units caught in a cycle are refused.

        Returns:
            The units in a safe order, plan order kept wherever it is safe.
        """
        vacated: dict[Path, int] = {}
        for index, unit in enumerate(units):
            for src, _ in unit.relocations:
                vacated.setdefault(src, index)
        waits: dict[int, set[int]] = {}
        for index, unit in enumerate(units):
            for _, dst in unit.relocations:
                for entered in (dst, dst.parent):
                    blocker = vacated.get(entered)
                    if blocker is not None and blocker != index:
                        waits.setdefault(index, set()).add(blocker)
        done: set[int] = set()
        ordered: list[MoveUnit] = []
        pending = list(range(len(units)))
        progress = True
        while pending and progress:
            progress = False
            for index in pending:
                if waits.get(index, set()) <= done:
                    ordered.append(units[index])
                    done.add(index)
                    pending.remove(index)
                    progress = True
                    break
        for index in pending:
            for src, _ in units[index].relocations:
                result.skipped.setdefault(
                    src, "would trade places with another path in the same move (refused)"
                )
        return ordered

    def _run_unit(
        self,
        unit: MoveUnit,
        plan: MovePlan,
        result: MoveResult,
        simulation: Namespace | None,
    ) -> None:
        """Run one unit and fold its outcome into ``result``, then report it.

        Args:
            unit: The unit to run.
            plan: The plan being executed; its ``on_unit`` hears about the unit.
            result: The whole move's result (mutated in place).
            simulation: The dry run's namespace, or ``None`` for a real move.
        """
        outcome = MoveResult(dry_run=result.dry_run)
        try:
            self._attempt(unit, plan, outcome, simulation)
        except BaseException as exc:
            # Whatever escaped, the unit's record must not read "nothing to do".
            outcome.skipped.setdefault(
                self._unit_key(unit), f"{_what_stopped(exc)} before it finished (refused)"
            )
            raise
        finally:
            result.absorb(outcome)
            observer = plan.on_unit
            if observer is not None:
                try:
                    observer(unit, outcome)
                except Exception:  # noqa: BLE001, S110 - logging must never break a move
                    pass

    def _unit_key(self, unit: MoveUnit) -> Path:
        """Return the path a unit's own refusal is recorded under."""
        if unit.relocations:
            return unit.relocations[0][0]
        if unit.rewrites:
            return unit.rewrites[0]
        return Path(f"<{self.name}>")

    def _attempt(
        self,
        unit: MoveUnit,
        plan: MovePlan,
        outcome: MoveResult,
        simulation: Namespace | None,
        *,
        merged: bool = False,
    ) -> None:
        """Check one unit, then re-point and relocate it — or put everything back.

        Args:
            unit: The unit to run.
            plan: The plan being executed.
            outcome: This unit's result (mutated in place).
            simulation: The dry run's namespace, or ``None`` for a real move.
            merged: True for a piece of a folder already re-planned as a merge,
                which is never merged again: a destination that still calls
                for one changed while the move ran.
        """
        problem, merge = self._unit_check(unit, plan, simulation)
        if merge:
            if merged:
                folder, destination = unit.relocations[0]
                outcome.skipped[folder] = (
                    f"its destination {destination} changed while the move ran (refused)"
                )
                return
            for part in self._merge_units(unit, plan, outcome, simulation):
                self._attempt(part, plan, outcome, simulation, merged=True)
            return
        if problem is not None:
            path, reason = problem
            outcome.skipped[path] = reason
            return
        carried = self._carried(unit, simulation)
        rewrites: list[tuple[Path, int, list[tuple[int, dict, dict]]]] = []
        moves: list[tuple[Path, Path]] = []
        created: list[Path] = []
        # The rewrite and the rename under way, if one is: Ctrl-C can land
        # after either has changed the disk and before it is recorded above.
        pending: list[tuple[Path, list[tuple[int, dict, dict]]]] = []
        renaming: tuple[Path, Path] | None = None
        try:
            for path in unit.rewrites:
                changed, error, undo = self._rewrite_transcript(
                    path, unit, plan, dry_run=outcome.dry_run, pending=pending
                )
                if error is not None:
                    raise _UnitFailed({path: error})
                if changed:
                    rewrites.append((path, changed, undo))
                pending.clear()
            for src, dst in unit.relocations:
                renaming = (src, dst)
                error = self._make_parent(dst, created, simulation)
                if error is None:
                    error = self._rename(src, dst, simulation)
                if error is not None:
                    raise _UnitFailed({src: error})
                moves.append((src, dst))
                renaming = None
        except _UnitFailed as failure:
            outcome.skipped.update(failure.reasons)
            self._put_back(moves, rewrites, created, outcome, simulation)
            return
        except BaseException as exc:
            # Interrupted (Ctrl-C) or crashed mid-unit: put it back, then let
            # the interruption through. A re-run finishes whatever is left.
            try:
                if simulation is None:
                    self._settle(pending, renaming, rewrites, moves)
            finally:
                self._abandon(unit, exc, moves, rewrites, created, outcome, simulation)
            raise
        outcome.rewritten.extend(path for path, _, _ in rewrites)
        outcome.fields_updated += sum(changed for _, changed, _ in rewrites)
        outcome.moved.extend(moves)
        outcome.relocated.update(
            {before: after for before, after in carried.items() if before != after}
        )

    def _settle(
        self,
        pending: list[tuple[Path, list[tuple[int, dict, dict]]]],
        renaming: tuple[Path, Path] | None,
        rewrites: list[tuple[Path, int, list[tuple[int, dict, dict]]]],
        moves: list[tuple[Path, Path]],
    ) -> None:
        """Count the step an interruption cut short among those to put back, if it took effect.

        A rewrite swapped in, or a rename finished, a moment before Ctrl-C is
        on disk though the unit never recorded it; a half-done link is taken
        back (:func:`~sx.adapters.relocation.landed`).

        Args:
            pending: The rewrite under way, with the changes it recorded.
            renaming: The rename under way.
            rewrites: The unit's rewrites so far (mutated in place).
            moves: The unit's renames so far (mutated in place).
        """
        for path, changes in pending:
            done = {done_path for done_path, _, _ in rewrites}
            if changes and path not in done and self._repointed(path, changes):
                rewrites.append((path, len(changes), list(changes)))
        if renaming is not None and renaming not in moves and landed(*renaming):
            moves.append(renaming)

    def _abandon(
        self,
        unit: MoveUnit,
        exc: BaseException,
        moves: list[tuple[Path, Path]],
        rewrites: list[tuple[Path, int, list[tuple[int, dict, dict]]]],
        created: list[Path],
        outcome: MoveResult,
        simulation: Namespace | None,
    ) -> None:
        """Put back a unit something stopped part-way, and say truthfully how far that got.

        The unit's record is written once the put-back has returned. If the
        put-back is itself interrupted, whatever it had not undone yet is
        reported as done, which it is.

        Args:
            unit: The unit.
            exc: What stopped it.
            moves: Its renames, as :meth:`_put_back` takes them.
            rewrites: Its rewrites, as :meth:`_put_back` takes them.
            created: The folders it created.
            outcome: Its result (mutated in place).
            simulation: The dry run's namespace, or ``None`` for a real move.
        """
        key = self._unit_key(unit)
        stopped = _what_stopped(exc)
        marks = (len(outcome.moved), len(outcome.rewritten))
        try:
            self._put_back(moves, rewrites, created, outcome, simulation)
        except BaseException:
            self._still_done(moves, rewrites, outcome, simulation)
            outcome.skipped[key] = (
                f"{stopped}, then interrupted again while being put back: what it had "
                "not put back yet is still done; re-run the same move to finish it"
            )
            raise
        stuck = (len(outcome.moved), len(outcome.rewritten)) != marks
        done = f"{stopped} before it finished, and put back{' except as noted' if stuck else ''} (refused)"
        earlier = outcome.skipped.get(key)
        outcome.skipped[key] = f"{done}; {earlier}" if earlier else done

    def _still_done(
        self,
        moves: list[tuple[Path, Path]],
        rewrites: list[tuple[Path, int, list[tuple[int, dict, dict]]]],
        outcome: MoveResult,
        simulation: Namespace | None,
    ) -> None:
        """Report what an interrupted put-back left undone: it is on disk.

        :meth:`_put_back` empties its lists as each step is undone, so only the
        step it was in the middle of can be in doubt; that one is judged from
        the disk.

        Args:
            moves: The renames not yet undone.
            rewrites: The rewrites not yet undone.
            outcome: The unit's result (mutated in place).
            simulation: The dry run's namespace, or ``None`` for a real move.
        """
        reason = "was not put back: the put-back was interrupted; re-run the same move to finish it"
        for pair in moves:
            src, dst = pair
            if pair in outcome.moved or (simulation is None and not landed(src, dst)):
                continue
            outcome.moved.append(pair)
            outcome.skipped.setdefault(dst, reason)
        last = len(rewrites) - 1
        for index, (path, changed, undo) in enumerate(rewrites):
            if path in outcome.rewritten:
                continue
            if simulation is None and index == last and not self._repointed(path, undo):
                continue  # its undo finished just before the interruption
            outcome.rewritten.append(path)
            outcome.fields_updated += changed
            outcome.skipped.setdefault(path, reason)

    @staticmethod
    def _repointed(path: Path, changes: list[tuple[int, dict, dict]]) -> bool:
        """Return True if ``path`` still holds a record as a rewrite re-pointed it.

        Args:
            path: A transcript.
            changes: What the rewrite changed, as :class:`Recorder` keeps it.

        Returns:
            Whether any of the new records is in the file.
        """
        expected = {canonical(new) for _, new, _ in changes}
        return any(canonical(obj) in expected for obj in read_records(path))

    def _unit_check(
        self, unit: MoveUnit, plan: MovePlan, simulation: Namespace | None
    ) -> tuple[tuple[Path, str] | None, bool]:
        """Check a unit against the store as it is right now.

        Args:
            unit: The unit about to run.
            plan: The plan being executed.
            simulation: The dry run's namespace, or ``None`` to ask the disk.

        Returns:
            ``(problem, merge)``: the path and reason the unit cannot run, or
            ``None``; and True when it is a folder whose destination now exists,
            which is merged into instead.
        """
        roots = self.store_roots()
        exempt = (self.sessions_root(), *roots)

        def exists(path: Path) -> bool:
            """Return True if anything is at ``path``, as the dry run or the disk sees it."""
            return simulation.lexists(path) if simulation else os.path.lexists(path)

        for path in unit.rewrites:
            if not is_within(path, roots):
                return (path, "outside store root (refused)"), False
            if not exists(path):
                return (path, "does not exist"), False
            if not _regular_file(path, simulation):
                # Swapped for a link since planning: rewriting it would put a
                # re-pointed copy in its place, a second copy of the session.
                return (path, "is no longer a regular file (refused)"), False
        for src, dst in unit.relocations:
            if not is_within(src, roots) or not is_within(dst, roots):
                return (src, "outside store root (refused)"), False
            if not exists(src):
                return (src, "does not exist"), False
            refusal = self._folder_refusal(dst.parent, simulation, exempt)
            if refusal is not None:
                return (src, refusal), False
            if simulation is None and same_entry(src, dst):
                # A case-only rename on a case-insensitive file system: the
                # destination "exists" because it is the source itself.
                if src.name == dst.name:
                    return (src, f"{dst} is this same file under another spelling (refused)"), False
                continue
            if exists(dst):
                if unit.kind == MoveUnit.FOLDER and self._mergeable(dst, roots, simulation):
                    return None, True
                return (src, f"already exists at destination: {dst} (refused)"), False
        for session in unit.sessions:
            if session.primary_path not in unit.rewrites:
                continue
            reason = self.destructive_refusal(session, rewriting=True)
            if reason is not None:
                return (session.primary_path, reason), False
        for path in self._carried(unit, simulation):
            if self.busy_transcript(path):
                return (path, "in use, or not safe to rewrite (refused)"), False
        if unit.kind == MoveUnit.FOLDER:
            folder = unit.relocations[0][0]
            stay = self._foreign_transcripts(folder, unit.sessions, plan.old)
            if stay:
                reason = (
                    f"its folder now also holds {len(stay)} session(s) that must stay "
                    "where they are (refused)"
                )
                return (folder, reason), False
        return None, False

    @staticmethod
    def _mergeable(path: Path, roots: list[Path], simulation: Namespace | None) -> bool:
        """Return True if an existing destination is a real directory inside the store."""
        if not is_within(path, roots):
            return False
        if simulation is not None:
            return simulation.is_dir(path)
        try:
            return stat.S_ISDIR(os.lstat(path).st_mode)
        except OSError:
            return False

    def _merge_units(
        self,
        unit: MoveUnit,
        plan: MovePlan,
        outcome: MoveResult,
        simulation: Namespace | None,
    ) -> list[MoveUnit]:
        """Re-plan a folder unit as a merge into a destination that now exists.

        The folder is planned again on a scratch plan, through the same hooks
        planning uses, against the store as it is now (or as the dry run has
        left it).

        Args:
            unit: A folder unit whose destination exists.
            plan: The plan being executed.
            outcome: Where refusals of the merge's own entries are recorded.
            simulation: The dry run's namespace, or ``None`` for a real move.

        Returns:
            The units the merge is made of.
        """
        folder, destination = unit.relocations[0]
        scratch = MovePlan(
            harness=plan.harness,
            old=plan.old,
            new=plan.new,
            sessions=list(unit.sessions),
            rewrites=list(unit.rewrites),
            _namespace=simulation if simulation is not None else Namespace(),
        )
        self._plan_relocation(folder, destination, scratch, self.store_roots())
        parts = self._folder_units(
            folder, destination, list(scratch.relocations), unit.sessions, scratch
        )
        outcome.skipped.update(scratch.blocked)
        return parts

    def _carried(self, unit: MoveUnit, simulation: Namespace | None) -> dict[Path, Path]:
        """Map every transcript a unit carries to where it will be afterwards.

        Args:
            unit: The unit.
            simulation: The dry run's namespace, or ``None`` to list the disk.

        Returns:
            Transcript path before the move -> path after it; a rewritten
            transcript that does not move maps to itself.
        """
        found: dict[Path, Path] = {}
        for src, dst in unit.relocations:
            if is_transcript(src):
                found[src] = dst
            elif unit.kind == MoveUnit.FOLDER:
                if simulation is not None:
                    names = simulation.names(src)
                    names = names if isinstance(names, set) else set()
                else:
                    try:
                        names = set(os.listdir(src))
                    except OSError:
                        names = set()
                for name in sorted(names):
                    if is_transcript(src / name):
                        found[src / name] = dst / name
        for path in unit.rewrites:
            found.setdefault(path, path)
        return found

    def _legacy_rewrites(self) -> bool:
        """Return True for a subclass that customizes ``_rewrite_all`` but not the hook.

        Such an adapter predates :meth:`move_transform`; its own rewrite is
        used, one transcript at a time, and undone from a snapshot.

        Returns:
            Whether :meth:`_rewrite_transcript` must go through ``_rewrite_all``.
        """
        cls = type(self)
        return (
            cls._rewrite_all is not JsonlFolderAdapter._rewrite_all
            and cls.move_transform is JsonlFolderAdapter.move_transform
        )

    def _rewrite_transcript(
        self,
        path: Path,
        unit: MoveUnit,
        plan: MovePlan,
        *,
        dry_run: bool,
        pending: list[tuple[Path, list[tuple[int, dict, dict]]]] | None = None,
    ) -> tuple[int, str | None, list[tuple[int, dict, dict]]]:
        """Re-point one transcript, remembering how to undo it.

        Args:
            path: The transcript, at its path before the move.
            unit: The unit it belongs to.
            plan: The plan being executed.
            dry_run: If True, count without writing.
            pending: Given the transcript and the changes being recorded
                before the rewrite starts, so that a rewrite Ctrl-C cuts short
                just after it was swapped in can still be undone.

        Returns:
            ``(records changed, error, undo)``, where ``undo`` lists ``(call
            number, new record, original record)`` for :meth:`_undo_rewrite`.
        """
        destination = next((dst for src, dst in unit.relocations if src == path), None)
        if destination is None:
            folder = next((dst for src, dst in unit.relocations if src == path.parent), None)
            destination = folder / path.name if folder is not None else None
        moved_from = path if destination is not None else None
        if self._legacy_rewrites():
            before = [] if dry_run else read_records(path)
            sub = replace(
                plan,
                rewrites=[path],
                relocations=[(path, destination)] if destination is not None else [],
                units=[],
                on_unit=None,
            )
            partial = MoveResult(dry_run=dry_run)
            self._rewrite_all(sub, partial, dry_run=dry_run)
            if path in partial.skipped:
                return 0, partial.skipped[path], []
            changed = partial.fields_updated
            if dry_run or not changed:
                return changed, None, []
            after = read_records(path)
            undo = [
                (index, after[index], before[index])
                for index in range(min(len(before), len(after)))
                if canonical(after[index]) != canonical(before[index])
            ]
            return changed, None, undo
        recorder = Recorder(self.move_transform(path, plan, moved_from=moved_from))
        if pending is not None and not dry_run:
            pending.append((path, recorder.changes))
        changed, error = rewrite_jsonl(path, recorder, dry_run=dry_run)
        if error is not None:
            return 0, error, []
        return changed, None, ([] if dry_run else recorder.changes)

    @staticmethod
    def _undo_rewrite(path: Path, undo: list[tuple[int, dict, dict]]) -> str | None:
        """Put back the records a rewrite replaced.

        Each record is matched by content at or after the position the rewrite
        changed, and restored to the record it replaced; lines the harness
        appended meanwhile are left alone.

        Args:
            path: The transcript, where it is now.
            undo: What :meth:`_rewrite_transcript` recorded.

        Returns:
            ``None`` when every record was restored, otherwise why not.
        """
        if not undo:
            return None
        expected = [canonical(new) for _, new, _ in undo]
        state = {"next": 0, "calls": 0}

        def restore(obj: dict) -> dict | None:
            """Return the original of a record this move changed, else ``None``."""
            call = state["calls"]
            state["calls"] += 1
            position = state["next"]
            if (
                position < len(undo)
                and call >= undo[position][0]
                and canonical(obj) == expected[position]
            ):
                state["next"] += 1
                return undo[position][2]
            return None

        _, error = rewrite_jsonl(path, restore)
        if error is not None:
            return error
        if state["next"] != len(undo):
            return f"restored {state['next']} of {len(undo)} re-pointed record(s)"
        return None

    @staticmethod
    def _make_parent(dst: Path, created: list[Path], simulation: Namespace | None) -> str | None:
        """Create ``dst``'s folder when it does not exist yet (a new destination).

        Args:
            dst: Where something is about to be renamed to.
            created: Collects the folders created, for a rollback.
            simulation: The dry run's namespace, or ``None`` for a real move.

        Returns:
            ``None`` when the folder exists or was made, otherwise why not.
        """
        parent = dst.parent
        if simulation is not None:
            if not simulation.lexists(parent):
                simulation.mkdir(parent)
                created.append(parent)
            return None
        if os.path.lexists(parent):
            return None
        try:
            os.mkdir(parent)
        except FileExistsError:
            return None
        except OSError as exc:
            return f"cannot create {parent}: {exc} (refused)"
        created.append(parent)
        return None

    @staticmethod
    def _rename(src: Path, dst: Path, simulation: Namespace | None) -> str | None:
        """Rename ``src`` to ``dst`` without ever replacing what is there.

        Args:
            src: The path to move.
            dst: Where it goes.
            simulation: The dry run's namespace, or ``None`` for a real move.

        Returns:
            ``None`` on success, otherwise why nothing was moved.
        """
        if simulation is None:
            return rename_no_clobber(src, dst)
        if simulation.lexists(dst):
            return f"already exists at destination: {dst} (refused)"
        simulation.rename(src, dst)
        return None

    def _put_back(
        self,
        moves: list[tuple[Path, Path]],
        rewrites: list[tuple[Path, int, list[tuple[int, dict, dict]]]],
        created: list[Path],
        outcome: MoveResult,
        simulation: Namespace | None,
    ) -> None:
        """Undo a unit that failed part-way: renames first, then rewrites.

        A transcript whose file could not be moved back keeps its new header,
        so it stays consistent where it now is; every other rewrite is
        reversed. Whatever cannot be undone is reported in ``outcome`` as done
        — it is on disk — next to the reason it could not be put back, so the
        result never claims a cleaner state than the store is in.

        ``moves`` and ``rewrites`` are emptied as each step is undone (or
        found stuck), so a put-back cut short leaves in them exactly what is
        still done.

        Args:
            moves: The renames that happened, in order (emptied).
            rewrites: The rewrites that happened, with their undo records
                (emptied).
            created: Folders the unit created.
            outcome: This unit's result (mutated in place).
            simulation: The dry run's namespace, or ``None`` for a real move.
        """
        mark = len(outcome.moved)
        away: set[Path] = set()
        while moves:
            src, dst = moves[-1]
            error = self._rename(dst, src, simulation)
            if error is not None:
                away.add(src)
                outcome.moved.insert(mark, (src, dst))  # kept in the order they ran
                outcome.skipped[dst] = (
                    f"could not be put back after a failed move ({error}); "
                    "re-run the same move to finish it"
                )
            moves.pop()
        while rewrites:
            path, changed, undo = rewrites[-1]
            if path in away or path.parent in away:
                outcome.rewritten.append(path)  # consistent where it now is: kept
                outcome.fields_updated += changed
            elif simulation is None:
                error = self._undo_rewrite(path, undo)
                if error is not None:
                    outcome.rewritten.append(path)
                    outcome.fields_updated += changed
                    outcome.skipped[path] = (
                        f"was re-pointed but could not be restored ({error}); "
                        "re-run the same move to finish it"
                    )
            rewrites.pop()
        for directory in reversed(created):
            if simulation is not None:
                simulation.rmdir(directory)
                continue
            try:
                directory.rmdir()  # only when empty: something else may live there now
            except OSError:
                pass

    def _prune_empty_folders(
        self, plan: MovePlan, moved: list[tuple[Path, Path]] | None = None
    ) -> None:
        """Remove project folders a move left completely empty.

        Only a directly-held project folder is considered, and only when it has
        no entries at all — ``rmdir`` rather than a recursive remove, so anything
        the move did not relocate keeps the folder alive.

        Args:
            plan: The plan that was carried out.
            moved: The renames that actually ran; defaults to the plan's
                relocations.
        """
        root = self.sessions_root()
        pairs = plan.relocations if moved is None else moved
        for folder in sorted({src.parent for src, _ in pairs}):
            if folder.parent != root or folder.is_symlink() or not folder.is_dir():
                continue
            try:
                folder.rmdir()  # only when empty: anything left keeps it
            except OSError:
                pass
