from __future__ import annotations

import json
import re
import shutil
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
    DROP,
    dir_size,
    home,
    human_size,
    is_under,
    is_within,
    iter_jsonl,
    mount_unavailable,
    parse_ts,
    repoint,
    rewrite_jsonl,
    scratchpad_root,
    write_text_atomic,
)
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

# The same shape, unanchored — some artifacts embed the id in the middle of a
# longer filename (``security_warnings_state_<uuid>.json``).
_UUID_ANYWHERE_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)

# Auxiliary ("cascade") directories under ``~/.claude`` searched by name prefix.
# Only ``file-history`` actually keys its entries by session id; the other three
# are kept because they cost one directory listing and a future version of the
# harness may start using them, but on the sampled machine they never match:
# shell snapshots are ``snapshot-zsh-<epoch>-<rand>``, ``sessions/`` is keyed by
# PID, and all 41 ``todos`` entries are ``<agent-id>-agent-<agent-id>.json`` —
# agent ids, which are not session ids. Do not assume these fire.
_CASCADE_DIRS = ("todos", "file-history", "shell-snapshots", "sessions")

# Directories under ``~/.claude`` holding one entry named *exactly* for a session
# id. These are built from the id rather than searched for, so there is no
# matching to get wrong.
_SESSION_DIRS = ("session-env", "tasks")

# Files under ``~/.claude`` whose name embeds the session id in the middle. A
# prefix search misses them entirely, which is why they were never cleaned up.
_SESSION_FILES = (
    ("security", "security_warnings_state_{sid}.json"),
    ("security", "security_warnings_state_{sid}.lock"),
)

# Directories that must be reachable by the delete guard because the cascade
# writes into them. ``~/.claude`` itself is deliberately NOT a store root: it
# holds ``plugins/`` (591 MB here), ``security/agent-sdk-venv`` (282 MB),
# ``settings.json`` and ``.credentials.json``. Naming the individual parents
# keeps the blast radius to directories this tool understands.
_GUARDED_SUBDIRS = _CASCADE_DIRS + _SESSION_DIRS + ("security",)

# Prompt history, tagged per session and per project. Named explicitly because
# it sits outside every store root; it is only ever rewritten in place, never
# removed.
_HISTORY_FILE = "history.jsonl"


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


def encode_project_dir(path: str | Path) -> str:
    """Encode an absolute project directory as Claude's project folder name.

    Claude replaces every character outside ``[A-Za-z0-9]`` with a hyphen, so
    ``/Users/a/.extra/git_clones`` becomes ``-Users-a--extra-git-clones``. The
    rule was confirmed against every project folder on the development machine
    (37 exact matches; the 3 that differed hold sessions whose directory was
    itself renamed, and 2 could not be read).

    Encoding is one-way, which is exactly what a move needs: the destination
    folder name is always computable, and nothing ever has to be decoded.

    Args:
        path: An absolute project directory.

    Returns:
        The folder name Claude would use for it.
    """
    return re.sub(r"[^A-Za-z0-9]", "-", str(path))


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

        The primary session store (``~/.claude/projects``) is always first. Each
        directory the cascade reaches into is named individually — never
        ``~/.claude`` as a whole, which also contains plugins, a 282 MB vendored
        virtualenv, the settings file and the credentials file. Claude's
        per-session scratchpad tree under ``/private/tmp`` is included for the
        same reason: the guard has to permit it, and nothing wider.

        Returns:
            A list of directories, primary store first.
        """
        base = home() / ".claude"
        roots = [base / "projects"]
        for sub in _GUARDED_SUBDIRS:
            d = base / sub
            if d.exists() and d not in roots:
                roots.append(d)
        scratch = scratchpad_root()
        if scratch.is_dir():
            roots.append(scratch)
        return roots

    def protected_paths(self) -> list[Path]:
        """Return the bulk inside ``~/.claude/security`` that is never deletable.

        ``security/`` has to be a store root so the cascade can reach
        ``security_warnings_state_<session-id>.*``, but it also holds a 282 MB
        vendored virtualenv and the harness's own logs. Only the state files are
        ever constructed as targets, so nothing reaches these in practice — this
        makes that a guarantee rather than a property of the calling code.

        Returns:
            Paths under ``~/.claude/security`` that must never be removed.
        """
        security = home() / ".claude" / "security"
        return [security / "agent-sdk-venv", security / "log.txt", security / "log.txt.1"]

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

        The session's own sidecar directory — ``<session-id>/`` sitting beside
        ``<session-id>.jsonl`` and holding its sub-agent transcripts and tool
        results — is included as well. It was previously missed entirely, so
        deleting a session left that directory behind as an orphan the cleanup
        screen could not attribute to anything.

        Returns:
            A list of matching auxiliary paths (commonly non-empty: on the
            sampled machine ``file-history/`` keys entries by session id).
        """
        sid = session.session_id
        matches: list[Path] = []
        sidecar = self.session_sidecar(session)
        if sidecar is not None:
            matches.append(sidecar)
        if not sid or not _UUID_RE.match(sid):
            return matches

        base = home() / ".claude"
        for sub in _CASCADE_DIRS:
            directory = base / sub
            if not directory.is_dir():
                continue
            for entry in sorted(directory.iterdir()):
                name = entry.name
                if name == sid or name.startswith(f"{sid}.") or name.startswith(f"{sid}-"):
                    matches.append(entry)

        # Constructed, not searched: the path is built from the session id, so
        # there is no pattern that could widen to a neighbour.
        for sub in _SESSION_DIRS:
            candidate = base / sub / sid
            if candidate.exists():
                matches.append(candidate)
        for sub, template in _SESSION_FILES:
            candidate = base / sub / template.format(sid=sid)
            if candidate.exists():
                matches.append(candidate)
        scratch = self.session_scratchpad(session)
        if scratch is not None:
            matches.append(scratch)

        # Belt and braces: every cascade target must carry the session id in its
        # own name. Nothing above can currently violate this, which is the point
        # — if a future rule does, it is refused here rather than in production.
        return [path for path in matches if sid in path.name]

    def session_scratchpad(self, session: Session) -> Path | None:
        """Return the session's scratchpad directory under ``/private/tmp``.

        Claude writes working files to
        ``/private/tmp/claude-<uid>/<encoded-project>/<session-id>/``, keyed by
        both the project and the session. It is outside every session store, so
        it survived every delete until now.

        Args:
            session: The session to locate.

        Returns:
            The scratchpad directory, or ``None`` when there is none.
        """
        if not session.project_path:
            return None
        candidate = (
            scratchpad_root()
            / encode_project_dir(session.project_path)
            / session.session_id
        )
        return candidate if candidate.is_dir() else None

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

        Artifacts keyed to sessions that no longer exist — scratchpads, task and
        environment directories, security state — are reported too, grouped by
        class so one stale class cannot bury the rest.

        Two deliberate abstentions keep real data out of the deletion list:
        a folder whose working directory cannot be read is never called dead
        (the lossy folder-name decode is not trusted for this decision), and a
        directory on an unmounted volume is treated as unavailable rather than
        gone.

        Returns:
            A list of discovered orphans.
        """
        projects = home() / ".claude" / "projects"
        orphans: list[Orphan] = self._stale_session_artifacts()
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

    # --- move ------------------------------------------------------------

    def _projects_root(self) -> Path:
        """Return the primary session store (``~/.claude/projects``)."""
        return home() / ".claude" / "projects"

    def session_sidecar(self, session: Session) -> Path | None:
        """Return the ``<session-id>/`` directory beside a transcript, if present.

        Claude keeps a session's sub-agent transcripts and tool results in a
        directory named for the session id, next to its ``.jsonl``. It has to
        travel with the transcript on a move, and go with it on a delete.

        Args:
            session: The session to inspect.

        Returns:
            The sidecar directory, or ``None`` if there is none.
        """
        path = session.primary_path
        if path is None:
            return None
        sidecar = path.parent / path.stem
        return sidecar if sidecar.is_dir() else None

    def repoint_record(self, obj: dict, old: Path, new: Path) -> dict | None:
        """Re-point a record's ``cwd`` field, leaving everything else alone.

        Claude stamps ``cwd`` on roughly three quarters of a transcript's lines,
        and a single session may carry several — sampling the development machine
        found files holding five or more distinct working directories, all
        subdirectories of one project. Re-pointing therefore covers the project
        directory and anything beneath it, and touches no other field: tool
        output and pasted text mentioning the old path are historical record.

        Args:
            obj: One parsed transcript record.
            old: The project directory being moved away from.
            new: The project directory being moved to.

        Returns:
            A replacement record, or ``None`` to leave the line untouched.
        """
        updated = repoint(obj.get("cwd", ""), old, new)
        if updated is None:
            return None
        replacement = dict(obj)
        replacement["cwd"] = updated
        return replacement

    @staticmethod
    def _folder_project(folder: Path, candidates: list[str]) -> str | None:
        """Return the project directory a store folder is named for.

        Folder names are a lossy encoding, so they are never decoded. Instead
        each candidate directory is *encoded* and compared with the folder name,
        which is exact.

        Args:
            folder: A folder directly under ``~/.claude/projects``.
            candidates: Directories that might have produced the folder name.

        Returns:
            The matching candidate, or ``None`` when the folder is named for some
            other directory (a project that was renamed after the folder was
            created, for instance).
        """
        for candidate in candidates:
            if candidate and encode_project_dir(candidate) == folder.name:
                return candidate
        return None

    def plan_move(self, sessions: list[Session], old: Path, new: Path) -> MovePlan:
        """Plan the rewrite *and* the relocation Claude's layout requires.

        The store encodes the project directory in the folder name, so a move is
        two operations: re-point every ``cwd`` inside the transcripts, then move
        the files to the folder the new directory encodes to.

        A folder is relocated whole whenever its name matches the directory it
        holds — that carries the project's ``memory/`` directory and every
        session's sidecar without enumerating them. Sessions that sit in a folder
        named for some other directory are relocated individually, transcript and
        sidecar together, so nothing is left stranded.

        Args:
            sessions: Sessions selected for the move.
            old: The project directory being moved away from.
            new: The project directory being moved to.

        Returns:
            A :class:`MovePlan` covering rewrites, relocations and refusals.
        """
        plan = super().plan_move(sessions, old, new)
        root = self._projects_root()
        roots = self.store_roots()

        by_folder: dict[Path, list[Session]] = {}
        for session in sessions:
            path = session.primary_path
            if path is not None:
                by_folder.setdefault(path.parent, []).append(session)

        strays = 0
        for folder, folder_sessions in sorted(by_folder.items()):
            if folder.parent != root:
                continue  # nested or unfamiliar layout — rewrite only
            candidates = [str(old)] + [
                s.project_path for s in folder_sessions if s.project_path
            ]
            implied = self._folder_project(folder, candidates)
            if implied is not None and is_under(implied, old):
                target = repoint(implied, old, new)
                self._plan_relocation(folder, root / encode_project_dir(target), plan, roots)
                continue
            for session in folder_sessions:
                strays += self._plan_session_relocation(session, folder, plan, roots)

        if strays:
            plan.note = (
                f"{strays} session(s) live in a folder named for another "
                "directory and move individually"
            )
        return plan

    def _plan_session_relocation(
        self,
        session: Session,
        folder: Path,
        plan: MovePlan,
        roots: list[Path],
    ) -> int:
        """Plan moving one transcript (and its sidecar) into the new folder.

        Args:
            session: The session to relocate.
            folder: The folder currently holding it.
            plan: The plan to record into.
            roots: The adapter's store roots, for the containment guard.

        Returns:
            ``1`` if a relocation was planned, ``0`` if the session is already in
            the right place.
        """
        path = session.primary_path
        target_dir = repoint(session.project_path or "", plan.old, plan.new)
        if path is None or target_dir is None:
            return 0
        destination = self._projects_root() / encode_project_dir(target_dir)
        if destination == folder:
            return 0
        self._plan_entry(path, destination / path.name, plan, roots)
        sidecar = self.session_sidecar(session)
        if sidecar is not None:
            self._plan_entry(sidecar, destination / sidecar.name, plan, roots)
        return 1

    def _plan_relocation(
        self, src: Path, dst: Path, plan: MovePlan, roots: list[Path]
    ) -> None:
        """Plan moving a whole project folder, merging when the target exists.

        A destination folder already exists whenever Claude has been run at the
        new path. Its entries are then merged one by one, and any name already
        taken is refused rather than overwritten — a move must never destroy a
        transcript that is already there.

        Args:
            src: The folder to move.
            dst: Where it should end up.
            plan: The plan to record into.
            roots: The adapter's store roots, for the containment guard.
        """
        if src == dst:
            return
        if not dst.exists():
            self._plan_entry(src, dst, plan, roots)
            return
        if not dst.is_dir():
            plan.blocked[dst] = "destination exists and is not a directory (refused)"
            return
        try:
            entries = sorted(src.iterdir())
        except OSError as exc:
            plan.blocked[src] = f"cannot read: {exc}"
            return
        for entry in entries:
            self._plan_entry(entry, dst / entry.name, plan, roots)

    @staticmethod
    def _plan_entry(src: Path, dst: Path, plan: MovePlan, roots: list[Path]) -> None:
        """Record one ``src -> dst`` relocation, or why it cannot happen."""
        if not is_within(src, roots) or not is_within(dst, roots):
            plan.blocked[src] = "outside store root (refused)"
            return
        if dst.exists():
            plan.blocked[src] = f"already exists at destination: {dst} (refused)"
            return
        plan.relocations.append((src, dst))

    def move(self, plan: MovePlan, *, dry_run: bool = False) -> MoveResult:
        """Re-point the transcripts, then relocate them to the new folder.

        Order matters: the rewrite runs first, while the planned paths still
        exist where the plan says they do. Relocating first would invalidate
        every one of them.

        Args:
            plan: The plan produced by :meth:`plan_move`.
            dry_run: If True, report what would happen and change nothing.

        Returns:
            A :class:`MoveResult`.
        """
        result = MoveResult(dry_run=dry_run)
        result.skipped.update(plan.blocked)
        if plan.note:
            result.note = plan.note
        self._rewrite_all(plan, result, dry_run=dry_run)
        for src, dst in plan.relocations:
            self._relocate(src, dst, result, dry_run=dry_run)
        if not dry_run:
            self._prune_empty_folders(plan)
        if plan.include_config:
            self._repoint_config(plan, result, dry_run=dry_run)
        return result

    def _relocate(self, src: Path, dst: Path, result: MoveResult, *, dry_run: bool) -> None:
        """Move one path, re-checking every guard at the moment of the move."""
        roots = self.store_roots()
        if not is_within(src, roots) or not is_within(dst, roots):
            result.skipped[src] = "outside store root (refused)"
            return
        if not src.exists():
            result.skipped[src] = "does not exist"
            return
        if dst.exists():
            result.skipped[src] = f"already exists at destination: {dst} (refused)"
            return
        if dry_run:
            result.moved.append((src, dst))
            return
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        except (OSError, shutil.Error) as exc:
            result.skipped[src] = f"move failed: {exc}"
            return
        result.moved.append((src, dst))

    def _prune_empty_folders(self, plan: MovePlan) -> None:
        """Remove project folders left completely empty by a merge.

        Only a directly-held project folder is considered, and only when it has
        no entries at all — ``rmdir`` rather than a recursive remove, so anything
        the move did not relocate keeps the folder alive.
        """
        root = self._projects_root()
        for src, _ in plan.relocations:
            folder = src.parent
            if folder.parent != root or not folder.is_dir():
                continue
            try:
                next(folder.iterdir())
            except StopIteration:
                try:
                    folder.rmdir()
                except OSError:
                    pass
            except OSError:
                continue

    # --- optional: Claude's own project state ----------------------------

    def _repoint_config(self, plan: MovePlan, result: MoveResult, *, dry_run: bool) -> None:
        """Re-point the project state Claude keeps outside its session store.

        Two files, both named explicitly rather than matched by pattern, because
        they sit outside :meth:`store_roots` and the containment guard therefore
        cannot vet them:

        * ``~/.claude.json`` — the ``projects`` map holding the trust decision,
          ``allowedTools`` and MCP servers for each directory. Without this a
          moved project is treated as brand new and its permissions are lost.
        * ``~/.claude/history.jsonl`` — the prompt history, tagged per project.

        Both belong to a harness that may be running, so each write re-checks
        that the file has not changed since it was read and abandons the update
        if it has.

        Args:
            plan: The plan being executed.
            result: The result to record outcomes in.
            dry_run: If True, count what would change and write nothing.
        """
        entries = self._repoint_settings(plan, result, dry_run=dry_run)
        entries += self._repoint_history(plan, result, dry_run=dry_run)
        if entries:
            verb = "would re-point" if dry_run else "re-pointed"
            detail = f"{verb} {entries} Claude config entry(s)"
            result.note = f"{result.note} · {detail}" if result.note else detail

    def _repoint_settings(self, plan: MovePlan, result: MoveResult, *, dry_run: bool) -> int:
        """Re-key the ``projects`` map in ``~/.claude.json``. Returns entries changed."""
        config = home() / ".claude.json"
        if not config.is_file():
            return 0
        try:
            before = config.stat()
            data = json.loads(config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError, UnicodeError) as exc:
            result.skipped[config] = f"cannot read: {exc}"
            return 0
        projects = data.get("projects") if isinstance(data, dict) else None
        if not isinstance(projects, dict):
            return 0

        renames: dict[str, str] = {}
        for key in list(projects):
            target = repoint(key, plan.old, plan.new) if isinstance(key, str) else None
            if target is None:
                continue
            if target in projects:
                result.skipped[config] = (
                    f"{target} already has Claude settings (refused)"
                )
                continue
            renames[key] = target
        if not renames:
            return 0
        if dry_run:
            return len(renames)

        for source_key, target_key in renames.items():
            projects[target_key] = projects.pop(source_key)
        error = write_text_atomic(config, json.dumps(data, indent=2), expect=before)
        if error is not None:
            result.skipped[config] = error
            return 0
        result.rewritten.append(config)
        result.fields_updated += len(renames)
        return len(renames)

    def _repoint_history(self, plan: MovePlan, result: MoveResult, *, dry_run: bool) -> int:
        """Re-point the ``project`` field in ``~/.claude/history.jsonl``."""
        history = home() / ".claude" / "history.jsonl"
        if not history.is_file():
            return 0

        def transform(obj: dict) -> dict | None:
            updated = repoint(obj.get("project", ""), plan.old, plan.new)
            if updated is None:
                return None
            replacement = dict(obj)
            replacement["project"] = updated
            return replacement

        changed, error = rewrite_jsonl(history, transform, dry_run=dry_run)
        if error is not None:
            result.skipped[history] = error
            return 0
        if changed and not dry_run:
            result.rewritten.append(history)
            result.fields_updated += changed
        return changed

    # --- prompt history --------------------------------------------------

    def _history_path(self) -> Path:
        """Return ``~/.claude/history.jsonl``."""
        return home() / ".claude" / _HISTORY_FILE

    def _drop_history_rows(
        self,
        *,
        session_id: str | None = None,
        project: str | None = None,
        dry_run: bool,
    ) -> tuple[int, str | None]:
        """Remove prompt-history rows belonging to a session or a project.

        The file is named explicitly rather than matched, because it lives
        outside every store root and the containment guard therefore cannot vet
        it. It is only ever rewritten in place — never removed — through the
        atomic rewriter, which abandons the write if Claude appends while it is
        in progress.

        Args:
            session_id: Drop rows whose ``sessionId`` matches.
            project: Drop rows whose ``project`` matches.
            dry_run: If True, count the rows and write nothing.

        Returns:
            A ``(rows_removed, error)`` pair; ``error`` is ``None`` on success.
        """
        history = self._history_path()
        if not history.is_file() or (session_id is None and project is None):
            return (0, None)

        def transform(obj: dict):
            if session_id is not None and obj.get("sessionId") == session_id:
                return DROP
            if project is not None and obj.get("project") == project:
                return DROP
            return None

        return rewrite_jsonl(history, transform, dry_run=dry_run)

    def delete(self, session: Session, *, dry_run: bool = False) -> DeleteResult:
        """Delete a session's files, then its rows in the prompt history.

        Files go first: a rewritten history with the transcript still on disk is
        a cosmetic inconsistency, whereas the reverse would claim the history was
        cleaned when it was not.

        Args:
            session: The session to remove.
            dry_run: If True, report what would happen and change nothing.

        Returns:
            A :class:`DeleteResult` whose ``note`` records the row count.
        """
        result = super().delete(session, dry_run=dry_run)
        rows, error = self._drop_history_rows(
            session_id=session.session_id, dry_run=dry_run
        )
        if error is not None:
            result.skipped[self._history_path()] = error
        elif rows:
            verb = "would remove" if dry_run else "removed"
            detail = f"{verb} {rows} prompt-history row(s)"
            result.note = f"{result.note} · {detail}" if result.note else detail
        return result

    # --- project-scoped leftovers ----------------------------------------

    def project_leftovers(self, project: str) -> ProjectLeftovers:
        """Describe the project-scoped state that outlives a project's sessions.

        Memory is the reason this exists. It belongs to the directory rather than
        to any conversation, so it is listed on every delete (to say it is being
        kept) and only offered for removal when the last session goes.

        Args:
            project: The project directory to describe.

        Returns:
            A :class:`ProjectLeftovers` naming the memory, paths and config
            entries involved.
        """
        leftovers = ProjectLeftovers(project_path=project)
        folder = self._projects_root() / encode_project_dir(project)

        memory_dir = folder / "memory"
        if memory_dir.is_dir():
            leftovers.memory_files.extend(
                sorted(f for f in memory_dir.rglob("*.md") if f.is_file())
            )
        index = folder / "MEMORY.md"
        if index.is_file():
            leftovers.memory_files.append(index)

        if folder.is_dir():
            leftovers.paths.append(folder)
            leftovers.size_bytes += dir_size(folder)
        scratch = scratchpad_root() / encode_project_dir(project)
        if scratch.is_dir():
            leftovers.paths.append(scratch)
            leftovers.size_bytes += dir_size(scratch)

        if self._settings_entry(project) is not None:
            leftovers.config_notes.append("Claude settings entry (trust, allowedTools)")
        rows, _ = self._drop_history_rows(project=project, dry_run=True)
        if rows:
            leftovers.config_notes.append(f"{rows} prompt-history row(s)")
        return leftovers

    def _settings_entry(self, project: str) -> dict | None:
        """Return the ``~/.claude.json`` settings for a project, if any."""
        config = home() / ".claude.json"
        if not config.is_file():
            return None
        try:
            data = json.loads(config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError, UnicodeError):
            return None
        projects = data.get("projects") if isinstance(data, dict) else None
        if isinstance(projects, dict):
            entry = projects.get(project)
            return entry if isinstance(entry, dict) else None
        return None

    def delete_project_leftovers(
        self, leftovers: ProjectLeftovers, *, dry_run: bool = False
    ) -> DeleteResult:
        """Remove a project's memory, store folder and settings.

        Only ever reached through an explicit, unticked confirmation: this is the
        one path in ``sx`` that destroys knowledge deliberately written to
        outlive its conversations.

        Args:
            leftovers: The plan produced by :meth:`project_leftovers`.
            dry_run: If True, report what would happen and change nothing.

        Returns:
            A :class:`DeleteResult` describing the removal.
        """
        result = self._delete_paths(leftovers.paths, dry_run=dry_run)
        details: list[str] = []

        if self._settings_entry(leftovers.project_path) is not None:
            error = self._drop_settings_entry(leftovers.project_path, dry_run=dry_run)
            if error is not None:
                result.skipped[home() / ".claude.json"] = error
            else:
                details.append(
                    "would remove Claude settings entry"
                    if dry_run
                    else "removed Claude settings entry"
                )

        rows, error = self._drop_history_rows(
            project=leftovers.project_path, dry_run=dry_run
        )
        if error is not None:
            result.skipped[self._history_path()] = error
        elif rows:
            verb = "would remove" if dry_run else "removed"
            details.append(f"{verb} {rows} prompt-history row(s)")

        if details:
            result.note = " · ".join(details)
        return result

    def _drop_settings_entry(self, project: str, *, dry_run: bool) -> str | None:
        """Remove a project's key from ``~/.claude.json``. Returns an error or None."""
        if dry_run:
            return None
        config = home() / ".claude.json"
        try:
            before = config.stat()
            data = json.loads(config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError, UnicodeError) as exc:
            return f"cannot read: {exc}"
        projects = data.get("projects") if isinstance(data, dict) else None
        if not isinstance(projects, dict) or project not in projects:
            return None
        projects.pop(project)
        return write_text_atomic(config, json.dumps(data, indent=2), expect=before)

    # --- stale artifacts left by sessions that are already gone ----------

    def _stale_session_artifacts(self) -> list[Orphan]:
        """Find artifacts keyed to session ids that no longer exist.

        Reported as one grouped orphan per class rather than one per entry: the
        sampled machine has 350 stale ``session-env`` directories alone, which
        would bury every other finding in the cleanup screen.

        Returns:
            A list of :attr:`OrphanKind.STALE_SESSION` orphans.
        """
        live = {path.stem for path in self.session_files()}
        base = home() / ".claude"
        groups: list[tuple[str, list[Path]]] = []

        for sub in ("file-history",) + _SESSION_DIRS:
            directory = base / sub
            if not directory.is_dir():
                continue
            stale = [
                entry
                for entry in sorted(directory.iterdir())
                if _UUID_RE.match(entry.name) and entry.name not in live
            ]
            if stale:
                groups.append((f"~/.claude/{sub}", stale))

        security = base / "security"
        if security.is_dir():
            stale = []
            for entry in sorted(security.glob("security_warnings_state_*")):
                found = _UUID_ANYWHERE_RE.search(entry.name)
                if found and found.group(0) not in live:
                    stale.append(entry)
            if stale:
                groups.append(("~/.claude/security", stale))

        scratch_root = scratchpad_root()
        if scratch_root.is_dir():
            stale = []
            try:
                project_dirs = sorted(d for d in scratch_root.iterdir() if d.is_dir())
            except OSError:
                project_dirs = []
            for project_dir in project_dirs:
                try:
                    entries = sorted(project_dir.iterdir())
                except OSError:
                    continue
                stale.extend(
                    e for e in entries if _UUID_RE.match(e.name) and e.name not in live
                )
            if stale:
                groups.append((str(scratch_root), stale))

        orphans: list[Orphan] = []
        for label, paths in groups:
            size = sum(dir_size(path) for path in paths)
            orphans.append(
                Orphan(
                    harness=self.name,
                    kind=OrphanKind.STALE_SESSION,
                    paths=paths,
                    reason=(
                        f"{len(paths)} entry(s) in {label} belong to sessions "
                        "that no longer exist"
                    ),
                    size_bytes=size,
                )
            )
        return orphans
