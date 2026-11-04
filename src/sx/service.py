"""Operation orchestration: previews, the active-session check, and the op-log.

The hard safety guards (refusing to touch anything outside a harness's store
roots) live in the adapters. These services layer the cross-cutting concerns on
top of them:

* a **dry-run preview** so the UI can show exactly what will happen;
* **active-session detection** — a session whose file changed within the last
  :data:`ACTIVE_WINDOW_SECONDS` is flagged as live, so the UI can warn and
  require an extra confirmation before touching it; and
* an **append-only op-log** recording every operation for accountability.

:class:`DeleteService` covers permanent removal, which has no undo.
:class:`MoveService` covers re-pointing a project at a new directory, which is
reversible by running the inverse move — the op-log records both endpoints so
that inverse is always recoverable.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

from sx.model import (
    Capability,
    DeleteResult,
    MovePlan,
    MoveResult,
    MoveUnit,
    Orphan,
    ProjectLeftovers,
    Session,
)
from sx.util import is_under

#: A session modified within this many seconds is considered "live".
ACTIVE_WINDOW_SECONDS = 90


def default_log_path() -> Path:
    """Return the op-log path: ``./sx-deletions.log``, or ``SX_LOG_FILE``.

    The log is written beside the working directory so it stays visible next to
    the work it describes. Note that it is therefore per-directory: running
    ``sx`` from several places produces several logs. Set ``SX_LOG_FILE`` to
    collect every deletion in one file instead.

    Returns:
        The op-log path.
    """
    override = os.environ.get("SX_LOG_FILE")
    if override:
        return Path(override).expanduser()
    return Path.cwd() / "sx-deletions.log"


def _no_adapter(harness: str, *, dry_run: bool) -> DeleteResult:
    """Return a refusing result for a harness with no registered adapter.

    Returned instead of raising ``KeyError``: these calls run inside Textual
    workers, where an unhandled exception tears down the whole app — possibly
    part-way through a batch of deletions.
    """
    return DeleteResult(
        dry_run=dry_run,
        skipped={Path(f"<{harness}>"): "no adapter registered (refused)"},
    )


def _append_log(log_path: Path, entry: dict) -> str | None:
    """Append one JSON record to the append-only op-log.

    Args:
        log_path: The log file to append to.
        entry: The record to serialize.

    Returns:
        ``None`` on success, otherwise a message describing the failure. Callers
        surface it rather than aborting: an audit-trail problem must never mask
        the result the user is acting on, but a silent failure would mean
        destructive work happening with no record at all.
    """
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        existed = log_path.exists()
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        if not existed:
            # The log records chat-derived titles and absolute project paths;
            # keep it owner-only rather than world-readable.
            try:
                log_path.chmod(0o600)
            except OSError:
                pass
        return None
    except OSError as exc:
        return f"could not write {log_path}: {exc}"


def session_is_active(adapter, session: Session, *, window: int = ACTIVE_WINDOW_SECONDS) -> bool:
    """Return True if ``session`` was written within ``window`` seconds.

    Liveness is asked of the owning adapter
    (:meth:`~sx.adapters.base.HarnessAdapter.last_activity`) and recomputed at
    call time, so the answer reflects the moment of the operation. File-backed
    harnesses re-``stat`` their transcript; database-backed ones consult the row
    that actually tracks conversation activity.

    When liveness cannot be determined the session is treated as **active**.
    That is the fail-safe direction: the cost of a wrong "live" is one extra
    typed confirmation, while a wrong "not live" removes the only guard standing
    between a keystroke and a conversation being written right now.

    Args:
        adapter: The adapter owning the session, or ``None``.
        session: The session to test.
        window: Recency window in seconds.

    Returns:
        True if the session appears to be in active use.
    """
    if adapter is None:
        return True
    try:
        last = adapter.last_activity(session)
    except Exception:  # noqa: BLE001 - a broken probe must not disable the guard
        return True
    if last is None:
        return True
    return (time.time() - last.timestamp()) <= window


class _OperationService:
    """Shared plumbing for the services: adapters, the op-log, and liveness.

    Args:
        adapters_by_name: Mapping of harness name to adapter instance.
        log_path: Where to append the op-log; defaults to
            :func:`default_log_path`.
    """

    def __init__(self, adapters_by_name: dict, log_path: Path | None = None) -> None:
        """Store adapters and resolve the op-log location."""
        self._adapters = adapters_by_name
        self._log_path = log_path or default_log_path()
        #: Set when the last op-log write failed, so the UI can surface it.
        self.last_log_error: str | None = None

    def is_active(self, session: Session, *, window: int = ACTIVE_WINDOW_SECONDS) -> bool:
        """Return True if the session appears to be in active use."""
        return session_is_active(
            self._adapters.get(session.harness), session, window=window
        )

    def _write_log(self, entry: dict) -> None:
        """Append ``entry`` to the op-log, recording any failure."""
        self.last_log_error = _append_log(self._log_path, entry)


class DeleteService(_OperationService):
    """Coordinates previews, deletions, the active check, and op-logging.

    Args:
        adapters_by_name: Mapping of harness name to adapter instance.
        log_path: Where to append the deletion op-log; defaults to
            :func:`default_log_path`.
    """

    # --- sessions --------------------------------------------------------

    def preview(self, session: Session) -> DeleteResult:
        """Return a dry-run :class:`DeleteResult` for a session.

        Args:
            session: The session to preview deleting.

        Returns:
            A dry-run result listing every path that would be removed.
        """
        adapter = self._adapters.get(session.harness)
        if adapter is None:
            return _no_adapter(session.harness, dry_run=True)
        return adapter.delete(session, dry_run=True)

    def delete(self, session: Session) -> DeleteResult:
        """Permanently delete a session and record it in the op-log.

        Args:
            session: The session to delete.

        Returns:
            The :class:`DeleteResult` describing what was removed.
        """
        adapter = self._adapters.get(session.harness)
        if adapter is None:
            return _no_adapter(session.harness, dry_run=False)
        result = adapter.delete(session, dry_run=False)
        self._log(
            action="delete_session",
            harness=session.harness,
            identifier=session.session_id,
            title=session.title,
            result=result,
        )
        return result

    # --- orphans ---------------------------------------------------------

    def preview_orphan(self, orphan: Orphan) -> DeleteResult:
        """Return a dry-run :class:`DeleteResult` for an orphan."""
        adapter = self._adapters.get(orphan.harness)
        if adapter is None:
            return _no_adapter(orphan.harness, dry_run=True)
        return adapter.delete_orphan(orphan, dry_run=True)

    def delete_orphan(
        self, orphan: Orphan, *, refused: dict[Path, str] | None = None
    ) -> DeleteResult:
        """Permanently delete an orphan and record it in the op-log.

        Args:
            orphan: The orphan to delete. The confirmation passes it narrowed
                to what it showed as removable, which may be nothing at all.
            refused: What the confirmation showed as refused, with why — paths
                ``orphan`` no longer holds. They are added to the result's
                refusals before the op-log is written, so the record and the
                result name every path that stayed; a refusal the delete makes
                itself keeps its own reason.

        Returns:
            The :class:`DeleteResult`, ``refused`` included.
        """
        adapter = self._adapters.get(orphan.harness)
        if adapter is None:
            return _no_adapter(orphan.harness, dry_run=False)
        result = adapter.delete_orphan(orphan, dry_run=False)
        for path, reason in (refused or {}).items():
            result.skipped.setdefault(path, reason)
        self._log(
            action="delete_orphan",
            harness=orphan.harness,
            identifier=orphan.kind.value,
            title=orphan.reason,
            result=result,
        )
        return result

    # --- memory ----------------------------------------------------------

    def preview_memory(self, memory) -> DeleteResult:
        """Return a dry-run result for deleting one memory document."""
        adapter = self._adapters.get("claude")
        if adapter is None:
            return _no_adapter("claude", dry_run=True)
        return adapter.delete_paths([memory.path], dry_run=True)

    def delete_memory(self, memory) -> DeleteResult:
        """Permanently delete one memory document and record it in the op-log.

        Memory is written to outlive its conversations, so its removal is logged
        under its own action with the project it belonged to — a session delete
        never produces one of these records.

        Args:
            memory: The :class:`~sx.memory.MemoryFile` to remove.

        Returns:
            A :class:`DeleteResult`.
        """
        adapter = self._adapters.get("claude")
        if adapter is None:
            return _no_adapter("claude", dry_run=False)
        result = adapter.delete_paths([memory.path], dry_run=False)
        self._log(
            action="delete_memory",
            harness="claude",
            identifier=memory.origin_session_id or memory.name,
            title=f"{memory.project_path or '(unknown)'} · {memory.name}",
            result=result,
        )
        return result

    # --- project-scoped state --------------------------------------------

    def remaining_sessions(
        self,
        session: Session,
        sessions_by_harness: dict[str, list[Session]] | None = None,
    ) -> int:
        """Count the sessions that would still point at this project afterwards.

        Counted across **every** harness, because the project-scoped state a
        deletion can offer to remove is itself cross-harness: Claude's memory and
        settings, Gemini's folder-trust entry. A Codex session still pointing at
        the directory means the project is not finished with.

        Args:
            session: The session about to be deleted.
            sessions_by_harness: An already-discovered pool to reuse.

        Returns:
            How many other sessions reference the same project.
        """
        if not session.project_path:
            return 0
        project = Path(session.project_path)
        total = 0
        for name, adapter in self._adapters.items():
            if Capability.BROWSE not in adapter.capabilities or not adapter.available():
                continue
            pool = sessions_by_harness.get(name) if sessions_by_harness else None
            try:
                found = adapter.sessions_for_project(project, pool)
            except Exception:  # noqa: BLE001, S112 - one harness must not break the count
                continue
            total += sum(
                1
                for other in found
                if not (
                    other.harness == session.harness
                    and other.session_id == session.session_id
                )
            )
        return total

    def project_leftovers(self, project: str) -> dict[str, ProjectLeftovers]:
        """Collect every harness's project-scoped state for ``project``.

        Args:
            project: The project directory.

        Returns:
            A mapping of harness name to :class:`ProjectLeftovers`, containing
            only the harnesses that hold something.
        """
        found: dict[str, ProjectLeftovers] = {}
        for name, adapter in self._adapters.items():
            if not adapter.available():
                continue
            try:
                leftovers = adapter.project_leftovers(project)
            except Exception:  # noqa: BLE001, S112
                continue
            if leftovers is not None and not leftovers.empty:
                found[name] = leftovers
        return found

    def delete_project_leftovers(
        self, leftovers: dict[str, ProjectLeftovers], *, dry_run: bool = False
    ) -> dict[str, DeleteResult]:
        """Remove each harness's project-scoped state, and record it.

        This is the only path in ``sx`` that destroys memory — knowledge written
        deliberately to outlive the conversations that produced it — so it is
        logged under its own action rather than folded into the session delete.

        Args:
            leftovers: The plans from :meth:`project_leftovers`.
            dry_run: If True, report what would happen and change nothing.

        Returns:
            A mapping of harness name to :class:`DeleteResult`.
        """
        results: dict[str, DeleteResult] = {}
        for name, plan in leftovers.items():
            adapter = self._adapters.get(name)
            if adapter is None:
                results[name] = _no_adapter(name, dry_run=dry_run)
                continue
            try:
                results[name] = adapter.delete_project_leftovers(plan, dry_run=dry_run)
            except Exception as exc:  # noqa: BLE001
                results[name] = DeleteResult(
                    dry_run=dry_run,
                    skipped={Path(f"<{name}>"): f"cleanup failed: {exc!r}"},
                )
        if not dry_run and results:
            project = next(iter(leftovers.values())).project_path
            memory_count = sum(len(p.memory_files) for p in leftovers.values())
            self._write_log(
                {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "action": "delete_project_state",
                    "project": project,
                    "memory_files": memory_count,
                    "harnesses": {
                        name: {
                            "removed": [str(p) for p in r.removed],
                            "freed_bytes": r.freed_bytes,
                            "skipped": {str(k): v for k, v in r.skipped.items()},
                            "note": r.note,
                        }
                        for name, r in results.items()
                    },
                }
            )
        return results

    # --- op-log ----------------------------------------------------------

    def _log(
        self,
        *,
        action: str,
        harness: str,
        identifier: str,
        title: str,
        result: DeleteResult,
    ) -> None:
        """Append one JSON record describing a deletion to the op-log.

        Logging failures are swallowed: an audit-trail problem must never abort
        or mask the deletion result the caller is acting on.

        Args:
            action: ``"delete_session"`` or ``"delete_orphan"``.
            harness: Harness name.
            identifier: Session id or orphan kind.
            title: Human-readable label (title or reason).
            result: The deletion result to record.
        """
        if result.dry_run:
            return
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "action": action,
            "harness": harness,
            "id": identifier,
            "title": title,
            "removed": [str(p) for p in result.removed],
            "freed_bytes": result.freed_bytes,
            "skipped": {str(k): v for k, v in result.skipped.items()},
        }
        if result.note:
            entry["note"] = result.note
        self._write_log(entry)


def _no_move_adapter(harness: str, *, dry_run: bool) -> MoveResult:
    """Return a refusing move result for a harness with no registered adapter."""
    return MoveResult(
        dry_run=dry_run,
        skipped={Path(f"<{harness}>"): "no adapter registered (refused)"},
    )


def _partial_move(harness: str, plan: MovePlan, progress: MoveResult, reason: str) -> MoveResult:
    """Return what a harness's move did before it failed, or was interrupted.

    Args:
        harness: The harness.
        plan: Its plan; what it refused up front is part of the result.
        progress: What it did, done or put back (see :func:`_progress_of`).
        reason: Why the move stopped, recorded under ``<harness>``.

    Returns:
        The partial :class:`MoveResult`.
    """
    result = MoveResult(dry_run=progress.dry_run)
    result.skipped.update(plan.blocked)
    result.absorb(progress)
    result.note = progress.note
    result.skipped[Path(f"<{harness}>")] = reason
    return result


def _progress_of(progress: MoveResult, exc: BaseException) -> MoveResult:
    """Return what a harness's move did before ``exc`` stopped it.

    ``progress`` is what each of its units reported as it ran. Work an adapter
    does around its units is known only to the adapter — Pi re-points the fork
    links of the units that finished on its way out — so ``move()`` may attach
    it to the exception it re-raises, as a :class:`~sx.model.MoveResult` named
    ``partial_result`` holding only that work: what no unit reported. It is
    absorbed into what the units reported, its note kept.

    Args:
        progress: What the harness's units reported.
        exc: What stopped the move.

    Returns:
        ``progress`` alone when ``exc`` carries no ``partial_result``,
        otherwise both together.
    """
    partial = getattr(exc, "partial_result", None)
    if not isinstance(partial, MoveResult):
        return progress
    merged = MoveResult(dry_run=progress.dry_run, note=partial.note or progress.note)
    merged.absorb(progress)
    merged.absorb(partial)
    return merged


def _plan_left_behind(adapter, old: Path, new: Path) -> MovePlan | None:
    """Plan what a harness with no session left to move still keeps for ``old``.

    Project state a harness relocates on its own — Oh My Pi's memory folders —
    can outlast its sessions under ``old``: an earlier run of the same move
    moved every session but not it (the memory was being written, say, or an
    older ``sx`` ran it last and was interrupted). No session selects the
    harness then, so only its own plan can find it. The plan is kept only
    when it would relocate something: a harness with no session here must not
    add refusals or notes to the move.

    Args:
        adapter: The harness's adapter.
        old: The project directory being moved away from.
        new: The project directory being moved to.

    Returns:
        The plan, or ``None`` when it would relocate nothing (or failed).
    """
    try:
        plan = adapter.plan_move([], old, new)
    except Exception:  # noqa: BLE001 - nothing was asked of this harness
        return None
    return plan if plan.relocations else None


class MoveService(_OperationService):
    """Coordinates re-pointing a project at a new directory across every harness.

    Two operations share this service:

    * **re-point** — the project directory was already moved by hand, and only
      the harness stores need to catch up;
    * **relocate** — ``sx`` moves the project directory itself first, then
      re-points the stores.

    Relocation runs *before* the stores are touched, and aborts everything if it
    fails. The other order would leave sessions pointing at a directory that was
    never created, which is exactly the broken state this feature exists to
    repair.

    Args:
        adapters_by_name: Mapping of harness name to adapter instance.
        log_path: Where to append the op-log; defaults to
            :func:`default_log_path`.
    """

    # --- planning --------------------------------------------------------

    def plan(
        self,
        old: Path,
        new: Path,
        *,
        sessions_by_harness: dict[str, list[Session]] | None = None,
        include_config: bool = False,
    ) -> dict[str, MovePlan]:
        """Ask every capable harness what re-pointing ``old`` to ``new`` involves.

        Sessions an interrupted move from ``old`` to ``new`` left half done —
        re-pointed, but still in the folder named for ``old`` — are included
        (see :meth:`~sx.adapters.base.HarnessAdapter.stranded_sessions`), so
        running the same move again finishes it. So is project state a
        harness relocates on its own, such as Oh My Pi's memory, when no
        session of that harness is left to move (:func:`_plan_left_behind`).

        Args:
            old: The project directory being moved away from.
            new: The project directory being moved to.
            sessions_by_harness: Already-discovered sessions to reuse; when
                omitted each adapter discovers its own.
            include_config: Also re-point project state a harness keeps outside
                its session store.

        Returns:
            A mapping of harness name to :class:`MovePlan`, containing only the
            harnesses with something to say.
        """
        plans: dict[str, MovePlan] = {}
        for name, adapter in self._adapters.items():
            if Capability.MOVE not in adapter.capabilities or not adapter.available():
                continue
            pool = sessions_by_harness.get(name) if sessions_by_harness else None
            try:
                if pool is None:
                    pool = list(adapter.discover())
                sessions = adapter.sessions_for_project(old, pool)
                # An interrupted move of the same two directories left these
                # re-pointed but not relocated; planning them again finishes it.
                stranded = adapter.stranded_sessions(old, new, pool)
            except Exception as exc:  # noqa: BLE001 - one harness must not break the rest
                plans[name] = MovePlan(
                    harness=name,
                    old=old,
                    new=new,
                    blocked={Path(f"<{name}>"): f"discovery failed: {exc!r}"},
                )
                continue
            sessions = sessions + [s for s in stranded if s not in sessions]
            if not sessions:
                plan = _plan_left_behind(adapter, old, new)
                if plan is None:
                    continue
            else:
                try:
                    plan = adapter.plan_move(sessions, old, new)
                except Exception as exc:  # noqa: BLE001
                    plan = MovePlan(
                        harness=name,
                        old=old,
                        new=new,
                        sessions=sessions,
                        blocked={Path(f"<{name}>"): f"planning failed: {exc!r}"},
                    )
            plan.include_config = include_config
            plan.live = [s for s in sessions if self.is_active(s)]
            if not plan.empty or plan.live:
                plans[name] = plan
        return plans

    # --- relocating the project directory itself -------------------------

    def check_relocation(self, old: Path, new: Path) -> str | None:
        """Return why the project directory cannot be moved, or ``None`` if it can.

        Moving the project directory reaches outside every harness store, so it
        gets its own guards rather than borrowing the store-root allowlist:

        * the source must be a real directory, and not the home directory, the
          filesystem root, or any directory containing a harness store — moving
          one of those would drag the session stores along with it;
        * the destination must not already hold anything, so nothing is merged
          into or overwritten; and
        * the destination must not be inside the source.

        Args:
            old: The project directory to move.
            new: Where it should end up.

        Returns:
            A refusal reason, or ``None`` when the move may proceed.
        """
        if old == new:
            return "source and destination are the same directory"
        if not old.is_dir():
            return f"not a directory: {old}"
        if is_under(str(new), old):
            return "destination is inside the directory being moved"

        try:
            resolved = old.resolve()
        except OSError:
            resolved = old
        if resolved == Path(resolved.anchor) or resolved == Path.home().resolve():
            return f"refusing to move {resolved}"
        for adapter in self._adapters.values():
            try:
                roots = adapter.store_roots()
            except Exception:  # noqa: BLE001, S112
                continue
            for root in roots:
                if is_under(str(root), old):
                    return f"refusing to move a directory containing a session store: {root}"

        if not new.parent.is_dir():
            return f"destination's parent does not exist: {new.parent}"
        if new.exists():
            if not new.is_dir():
                return f"destination already exists: {new}"
            try:
                next(new.iterdir())
            except StopIteration:
                pass  # an empty directory is fine to move into
            except OSError as exc:
                return f"cannot read destination: {exc}"
            else:
                return f"destination is not empty: {new}"
        return None

    def crosses_devices(self, old: Path, new: Path) -> bool:
        """Return True if the relocation would cross filesystems.

        A cross-device move is a copy followed by a delete: slower, and not
        atomic. The confirmation says so rather than letting a large project
        appear to hang.
        """
        try:
            return old.stat().st_dev != new.parent.stat().st_dev
        except OSError:
            return False

    def relocate_project(self, old: Path, new: Path) -> str | None:
        """Move the project directory itself, returning a reason on refusal.

        Args:
            old: The project directory to move.
            new: Where it should end up.

        Returns:
            ``None`` on success, otherwise why nothing was moved.
        """
        reason = self.check_relocation(old, new)
        if reason is not None:
            return reason
        try:
            if new.exists():
                # shutil.move would otherwise nest the source inside it.
                new.rmdir()
            shutil.move(str(old), str(new))
        except (OSError, shutil.Error) as exc:
            return f"could not move the project directory: {exc}"
        self._write_log(
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "action": "move_project",
                "old": str(old),
                "new": str(new),
            }
        )
        return None

    # --- executing -------------------------------------------------------

    def move(
        self, plans: dict[str, MovePlan], *, dry_run: bool = False
    ) -> dict[str, MoveResult]:
        """Execute every plan, recording it in the op-log as it happens.

        A real move writes a ``move_started`` record holding the whole plan
        before anything is touched, a ``move_unit`` record as each atomic unit
        finishes (or is refused), and the ``move_sessions`` record at the end —
        with ``"status": "interrupted"`` if the move was cut short, by Ctrl-C
        for instance. An interrupted move therefore always leaves a trail, and
        running the same move again finishes it.

        A harness whose move fails, or is interrupted, part-way keeps what its
        units did — done, or put back — in its result and in the final record,
        next to the failure, along with anything its adapter attached to the
        exception as ``partial_result`` (see :func:`_progress_of`).

        Args:
            plans: The plans produced by :meth:`plan`.
            dry_run: If True, report what would happen and change nothing.

        Returns:
            A mapping of harness name to :class:`MoveResult`.
        """
        results: dict[str, MoveResult] = {}
        if not dry_run and plans:
            self.last_log_error = None
            self._log_move_started(plans)
        status = "interrupted"
        try:
            for name, plan in plans.items():
                adapter = self._adapters.get(name)
                if adapter is None:
                    results[name] = _no_move_adapter(name, dry_run=dry_run)
                    continue
                progress = MoveResult(dry_run=dry_run)
                plan.on_unit = functools.partial(self._unit_ran, name, plan, progress)
                try:
                    results[name] = adapter.move(plan, dry_run=dry_run)
                except Exception as exc:  # noqa: BLE001 - one harness must not abort the rest
                    results[name] = _partial_move(
                        name, plan, _progress_of(progress, exc), f"move failed: {exc!r}"
                    )
                except BaseException as exc:
                    results[name] = _partial_move(
                        name,
                        plan,
                        _progress_of(progress, exc),
                        "interrupted before this harness's move finished; "
                        "re-run the same move to finish it",
                    )
                    raise
                finally:
                    plan.on_unit = None
            status = "finished"
        finally:
            if not dry_run and plans:
                self._log_move(plans, results, status=status)
        return results

    def _unit_ran(
        self,
        harness: str,
        plan: MovePlan,
        progress: MoveResult,
        unit: MoveUnit,
        outcome: MoveResult,
    ) -> None:
        """Keep one unit's outcome, and record it unless this is a dry run.

        Args:
            harness: The harness the unit belongs to.
            plan: The plan being executed.
            progress: What the harness's units have done so far (mutated).
            unit: The unit.
            outcome: What it did.
        """
        progress.absorb(outcome)
        if not progress.dry_run:
            self._log_move_unit(harness, plan, unit, outcome)

    def _log(self, entry: dict) -> None:
        """Append ``entry`` to the op-log, keeping the first failure of a move."""
        error = _append_log(self._log_path, entry)
        if error is not None and self.last_log_error is None:
            self.last_log_error = error

    def _log_move_started(self, plans: dict[str, MovePlan]) -> None:
        """Record the whole plan before a move touches anything.

        Args:
            plans: The plans about to be executed.
        """
        any_plan = next(iter(plans.values()))
        self._log(
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "action": "move_started",
                "old": str(any_plan.old),
                "new": str(any_plan.new),
                "harnesses": {
                    name: {
                        "sessions": len(plan.sessions),
                        "rewrites": [str(p) for p in plan.rewrites],
                        "relocations": [[str(a), str(b)] for a, b in plan.relocations],
                        "blocked": {str(k): v for k, v in plan.blocked.items()},
                        "note": plan.note,
                    }
                    for name, plan in plans.items()
                },
            }
        )

    def _log_move_unit(
        self, harness: str, plan: MovePlan, unit: MoveUnit, outcome: MoveResult
    ) -> None:
        """Record one unit of a move as soon as it has run.

        Args:
            harness: The harness the unit belongs to.
            plan: The plan being executed.
            unit: The unit.
            outcome: What it did.
        """
        if outcome.moved or outcome.rewritten:
            status = "partial" if outcome.refused else "done"
        else:
            status = "refused" if outcome.refused else "nothing to do"
        self._log(
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "action": "move_unit",
                "old": str(plan.old),
                "new": str(plan.new),
                "harness": harness,
                "unit": unit.kind,
                "planned": [[str(a), str(b)] for a, b in unit.relocations],
                "status": status,
                "rewritten": [str(p) for p in outcome.rewritten],
                "moved": [[str(a), str(b)] for a, b in outcome.moved],
                "relocated": {str(a): str(b) for a, b in outcome.relocated.items()},
                "fields_updated": outcome.fields_updated,
                "skipped": {str(k): v for k, v in outcome.skipped.items()},
            }
        )

    def _log_move(
        self,
        plans: dict[str, MovePlan],
        results: dict[str, MoveResult],
        *,
        status: str = "finished",
    ) -> None:
        """Append one record describing the whole move, once it has ended.

        Both endpoints are recorded, because that is what makes the operation
        reversible: running the inverse move restores the previous state, and
        this log is where the previous path can still be read afterwards.
        ``moved`` lists each rename as it actually ran and ``relocated`` where
        every transcript actually ended up, so the record matches the disk even
        when a folder had to be merged rather than renamed.

        Args:
            plans: The plans that were executed.
            results: What each harness did; a harness the move never reached
                is absent.
            status: ``"finished"``, or ``"interrupted"`` when the move was cut
                short.
        """
        any_plan = next(iter(plans.values()))
        self._log(
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "action": "move_sessions",
                "status": status,
                "old": str(any_plan.old),
                "new": str(any_plan.new),
                "harnesses": {
                    name: {
                        "rewritten": [str(p) for p in result.rewritten],
                        "moved": [[str(a), str(b)] for a, b in result.moved],
                        "relocated": {str(a): str(b) for a, b in result.relocated.items()},
                        "fields_updated": result.fields_updated,
                        "skipped": {str(k): v for k, v in result.skipped.items()},
                        "note": result.note,
                    }
                    for name, result in results.items()
                },
            }
        )


def move_summary(results: dict[str, MoveResult]) -> str:
    """Summarize a whole move in one line, per harness.

    Args:
        results: The per-harness results.

    Returns:
        A summary such as ``"claude: 12 file(s) re-pointed · 1 relocated"``, or a
        statement that nothing happened.
    """
    parts = [
        f"{name}: {result.summary()}"
        for name, result in results.items()
        if result.summary() != "nothing to move"
    ]
    return " · ".join(parts) if parts else "nothing to move"
