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

import shutil
from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from sx.model import (
    Capability,
    DeleteResult,
    Message,
    MovePlan,
    MoveResult,
    Orphan,
    ProjectLeftovers,
    Session,
)
from sx.util import (
    dir_size,
    file_mtime,
    is_within,
    is_under,
    iter_jsonl,
    mount_unavailable,
    read_first_line,
    rewrite_jsonl,
)


def _same_path(a: Path, b: Path) -> bool:
    """Return True if two paths resolve to the same location."""
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return False


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
    """

    name: str = "base"
    display: str = "Base"
    capabilities: Capability = Capability.NONE

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
        """Carry out a :class:`MovePlan`. Default: do nothing."""
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
        """Permanently delete an orphan's files, with the same root guard."""
        return self._delete_paths(orphan.paths, dry_run=dry_run)

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
            if not path.exists():
                result.skipped[path] = "does not exist"
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
            size = dir_size(path)
            if dry_run:
                result.removed.append(path)
                result.freed_bytes += size
                continue
            try:
                if path.is_dir():
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
        transcript is left for :meth:`load`.
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
            is_orphan=(
                bool(project)
                and not Path(project).exists()
                and not mount_unavailable(project)
            ),
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
        """Apply :meth:`repoint_record` to every file in ``plan.rewrites``.

        Args:
            plan: The plan being executed.
            result: The result to record outcomes in (mutated in place).
            dry_run: If True, count without writing.
        """

        def transform(obj: dict) -> dict | None:
            return self.repoint_record(obj, plan.old, plan.new)

        for path in plan.rewrites:
            changed, error = rewrite_jsonl(path, transform, dry_run=dry_run)
            if error is not None:
                result.skipped[path] = error
                continue
            if changed:
                result.rewritten.append(path)
                result.fields_updated += changed
