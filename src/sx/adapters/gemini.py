from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ..model import (
    DeleteResult,
    Message,
    MovePlan,
    MoveResult,
    Orphan,
    OrphanKind,
    ProjectLeftovers,
    Role,
    Session,
)
from ..util import (
    dir_size,
    home,
    is_within,
    is_under,
    iter_jsonl,
    parse_ts,
    read_first_line,
    repoint,
    write_text_atomic,
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

    # --- move ------------------------------------------------------------

    def _registry(self) -> Path:
        """Return ``~/.gemini/projects.json``, the path -> directory-name map."""
        return home() / ".gemini" / "projects.json"

    def plan_move(self, sessions: list[Session], old: Path, new: Path) -> MovePlan:
        """Plan a move as an update of Gemini's two path registries.

        Gemini never records the project directory inside a chat file, so no
        transcript is rewritten. The directory holding a session is named by an
        opaque key that Gemini itself allocates, so it is not renamed either.
        What has to change is the pair of places the path is actually written:
        the ``.project_root`` marker beside the chats, and the ``projects.json``
        map Gemini uses to find that directory again.

        Args:
            sessions: Sessions selected for the move.
            old: The project directory being moved away from.
            new: The project directory being moved to.

        Returns:
            A :class:`MovePlan` listing the registry files to update.
        """
        plan = MovePlan(harness=self.name, old=old, new=new, sessions=list(sessions))
        roots = self.store_roots()
        seen: set[Path] = set()
        for session in sessions:
            path = session.primary_path
            if path is None:
                continue
            marker = self._project_root_marker(path)
            if marker in seen:
                continue
            seen.add(marker)
            if not marker.is_file():
                plan.blocked[marker] = "does not exist"
                continue
            if not is_within(marker, roots):
                plan.blocked[marker] = "outside store root (refused)"
                continue
            plan.rewrites.append(marker)

        registry = self._registry()
        if plan.rewrites and registry.is_file() and is_within(registry, roots):
            plan.rewrites.append(registry)
        return plan

    def move(self, plan: MovePlan, *, dry_run: bool = False) -> MoveResult:
        """Update every ``.project_root`` marker, then the shared registry.

        Args:
            plan: The plan produced by :meth:`plan_move`.
            dry_run: If True, report what would change and write nothing.

        Returns:
            A :class:`MoveResult`.
        """
        result = MoveResult(dry_run=dry_run)
        result.skipped.update(plan.blocked)
        registry = self._registry()
        for path in plan.rewrites:
            if path == registry:
                self._repoint_registry(path, plan, result, dry_run=dry_run)
            else:
                self._repoint_marker(path, plan, result, dry_run=dry_run)
        return result

    @staticmethod
    def _repoint_marker(
        marker: Path, plan: MovePlan, result: MoveResult, *, dry_run: bool
    ) -> None:
        """Rewrite one ``.project_root`` marker, preserving its trailing bytes."""
        try:
            before = marker.stat()
            text = marker.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            result.skipped[marker] = f"cannot read: {exc}"
            return
        body = text.strip()
        updated = repoint(body, plan.old, plan.new)
        if updated is None:
            return
        if dry_run:
            result.rewritten.append(marker)
            result.fields_updated += 1
            return
        trailer = text[len(text.rstrip()):]
        error = write_text_atomic(marker, updated + trailer, expect=before)
        if error is not None:
            result.skipped[marker] = error
            return
        result.rewritten.append(marker)
        result.fields_updated += 1

    @staticmethod
    def _repoint_registry(
        registry: Path, plan: MovePlan, result: MoveResult, *, dry_run: bool
    ) -> None:
        """Re-key ``projects.json`` so Gemini finds the directory under its new path."""
        try:
            before = registry.stat()
            data = json.loads(registry.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError, UnicodeError) as exc:
            result.skipped[registry] = f"cannot read: {exc}"
            return
        projects = data.get("projects") if isinstance(data, dict) else None
        if not isinstance(projects, dict):
            return

        renames: dict[str, str] = {}
        for key in list(projects):
            target = repoint(key, plan.old, plan.new) if isinstance(key, str) else None
            if target is None:
                continue
            if target in projects:
                result.skipped[registry] = f"{target} is already registered (refused)"
                continue
            renames[key] = target
        if not renames:
            return
        if dry_run:
            result.rewritten.append(registry)
            result.fields_updated += len(renames)
            return

        for source_key, target_key in renames.items():
            projects[target_key] = projects.pop(source_key)
        error = write_text_atomic(registry, json.dumps(data, indent=2) + "\n", expect=before)
        if error is not None:
            result.skipped[registry] = error
            return
        result.rewritten.append(registry)
        result.fields_updated += len(renames)

    # --- project-scoped state --------------------------------------------

    def _trusted_folders(self) -> Path:
        """Return ``~/.gemini/trustedFolders.json``."""
        return home() / ".gemini" / "trustedFolders.json"

    def project_leftovers(self, project: str) -> ProjectLeftovers | None:
        """Describe Gemini's per-project state: its hash directory and registries.

        Gemini keys a project by an opaque directory name and records the mapping
        in two files. Once the project's chats are gone the directory holds only
        a ``.project_root`` marker, and both registry entries point at nothing.

        Args:
            project: The project directory.

        Returns:
            A :class:`ProjectLeftovers`, or ``None`` if Gemini knows nothing
            about this project.
        """
        leftovers = ProjectLeftovers(project_path=project)
        base = home() / ".gemini"
        for sub in ("tmp", "history"):
            root = base / sub
            if not root.is_dir():
                continue
            for hash_dir in sorted(root.iterdir()):
                marker = hash_dir / ".project_root"
                if not marker.is_file():
                    continue
                try:
                    target = marker.read_text(encoding="utf-8").strip()
                except (OSError, UnicodeError):
                    continue
                if target and is_under(target, project):
                    leftovers.paths.append(hash_dir)
                    leftovers.size_bytes += dir_size(hash_dir)

        if self._registry_keys(self._registry(), project):
            leftovers.config_notes.append("Gemini project registry entry")
        if self._registry_keys(self._trusted_folders(), project, fold_case=True):
            leftovers.config_notes.append("Gemini folder-trust entry")
        return leftovers if not leftovers.empty else None

    @staticmethod
    def _registry_keys(path: Path, project: str, *, fold_case: bool = False) -> list[str]:
        """Return the keys in a JSON path-registry that refer to ``project``.

        ``trustedFolders.json`` stores its paths lower-cased, so that file is
        matched case-insensitively; an exact comparison silently found nothing.

        Args:
            path: The JSON file to inspect.
            project: The project directory being removed.
            fold_case: Compare case-insensitively.

        Returns:
            The matching top-level keys.
        """
        if not path.is_file():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError, UnicodeError):
            return []
        mapping = data.get("projects") if isinstance(data.get("projects"), dict) else data
        if not isinstance(mapping, dict):
            return []
        wanted = project.lower() if fold_case else project
        return [
            key
            for key in mapping
            if isinstance(key, str) and (key.lower() if fold_case else key) == wanted
        ]

    def delete_project_leftovers(
        self, leftovers: ProjectLeftovers, *, dry_run: bool = False
    ) -> DeleteResult:
        """Remove Gemini's hash directory and both registry entries for a project."""
        result = self._delete_paths(leftovers.paths, dry_run=dry_run)
        details: list[str] = []
        for path, fold in ((self._registry(), False), (self._trusted_folders(), True)):
            error, removed = self._drop_registry_keys(
                path, leftovers.project_path, fold_case=fold, dry_run=dry_run
            )
            if error is not None:
                result.skipped[path] = error
            elif removed:
                verb = "would remove" if dry_run else "removed"
                details.append(f"{verb} {removed} entry(s) from {path.name}")
        if details:
            result.note = " · ".join(details)
        return result

    def _drop_registry_keys(
        self, path: Path, project: str, *, fold_case: bool, dry_run: bool
    ) -> tuple[str | None, int]:
        """Delete a project's keys from a JSON registry. Returns ``(error, count)``."""
        keys = self._registry_keys(path, project, fold_case=fold_case)
        if not keys:
            return (None, 0)
        if dry_run:
            return (None, len(keys))
        try:
            before = path.stat()
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError, UnicodeError) as exc:
            return (f"cannot read: {exc}", 0)
        mapping = data.get("projects") if isinstance(data.get("projects"), dict) else data
        for key in keys:
            mapping.pop(key, None)
        error = write_text_atomic(path, json.dumps(data, indent=2) + "\n", expect=before)
        return (error, 0 if error else len(keys))
