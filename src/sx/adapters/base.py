"""Adapter contract and the reusable JSONL-folder base implementation.

Every harness is represented by a subclass of :class:`HarnessAdapter`. Most
harnesses store sessions as a folder of JSONL files, so the bulk of that work
lives in :class:`JsonlFolderAdapter`; a concrete harness usually overrides only
a few small hooks.

Harnesses that are not installed still ship as adapter subclasses; their
:meth:`HarnessAdapter.available` returns ``False`` and the UI greys them out
until their store appears on disk.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from sx.model import (
    Capability,
    DeleteResult,
    Message,
    Orphan,
    Session,
)
from sx.util import (
    dir_size,
    file_mtime,
    is_within,
    iter_jsonl,
    read_first_line,
)


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

    capabilities = Capability.BROWSE | Capability.ORPHANS | Capability.DELETE

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
            is_orphan=bool(project) and not Path(project).exists(),
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
