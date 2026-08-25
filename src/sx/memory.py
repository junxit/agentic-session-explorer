"""Project memory: the documents a harness keeps *between* sessions.

Claude Code writes durable notes to ``~/.claude/projects/<encoded-cwd>/memory/``
— facts about the user, corrections they have given, ongoing project state —
loaded back into every later session for that directory. They belong to the
project, not to any one conversation, which is why deleting a session never
removes them and why they need somewhere of their own to be managed.

Each document opens with a small frontmatter block::

    ---
    name: harness-session-formats
    description: "On-disk session formats for Claude/Codex/Gemini"
    metadata:
      type: reference
      originSessionId: f40d65c1-ce10-4c3a-9797-2c56e1caab0f
    ---

``originSessionId`` records which conversation wrote the memory — present on
about 70 % of them — so this module can say whether that session still exists.
The block is parsed with the standard library rather than a YAML dependency: the
fields needed are flat ``key: value`` lines, and a memory whose frontmatter does
not parse still lists, with its filename as the name.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from sx.util import file_mtime, home, sanitize_text

#: Frontmatter delimiter at the very start of a memory document.
_FENCE = "---"

#: One ``key: value`` line inside the frontmatter, at any indent level.
_FIELD_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$")

#: Only the first part of a document is frontmatter; never scan a whole file.
_MAX_FRONTMATTER_LINES = 40


@dataclass(slots=True)
class MemoryFile:
    """One memory document belonging to a project.

    Attributes:
        path: The document on disk.
        project_path: The project directory it belongs to, if resolvable.
        name: The ``name`` field, falling back to the filename stem.
        description: The ``description`` field; may be empty.
        kind: The declared ``type`` (``user``, ``feedback``, ``project``,
            ``reference``), or ``"index"`` for a ``MEMORY.md`` table of contents.
        origin_session_id: The session that wrote it, when recorded.
        origin_exists: Whether that session's transcript is still on disk.
        size_bytes: File size.
        modified: Last-modified time.
    """

    path: Path
    project_path: str | None = None
    name: str = ""
    description: str = ""
    kind: str = ""
    origin_session_id: str | None = None
    origin_exists: bool = False
    size_bytes: int = 0
    modified: datetime | None = None
    #: Extra frontmatter fields, kept out of the shared surface.
    extra: dict = field(default_factory=dict)

    @property
    def project_name(self) -> str:
        """Short, display-friendly name of the owning project."""
        if self.project_path:
            return Path(self.project_path).name or self.project_path
        return "(unknown project)"

    @property
    def origin_label(self) -> str:
        """How the originating session should be shown in a list."""
        if self.origin_session_id is None:
            return "—"
        short = self.origin_session_id[:8]
        return short if self.origin_exists else f"{short} (deleted)"


def parse_frontmatter(text: str) -> dict[str, str]:
    """Extract the flat ``key: value`` pairs from a document's frontmatter.

    Nesting is ignored — ``metadata:`` and the keys under it are flattened into
    one mapping — because every field this module needs is unique by name. Quotes
    around a value are stripped.

    Args:
        text: The start of a memory document.

    Returns:
        The parsed fields, empty if the document has no frontmatter block.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FENCE:
        return {}
    fields: dict[str, str] = {}
    for line in lines[1 : _MAX_FRONTMATTER_LINES + 1]:
        if line.strip() == _FENCE:
            break
        match = _FIELD_RE.match(line)
        if match is None:
            continue
        key, value = match.group(1), match.group(2)
        if value[:1] == value[-1:] and value[:1] in ('"', "'"):
            value = value[1:-1]
        if value:
            fields[key] = value
    return fields


def _memory_from_file(
    path: Path, project_path: str | None, live_sessions: set[str]
) -> MemoryFile:
    """Build a :class:`MemoryFile` from one document on disk."""
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:4096]
    except OSError:
        head = ""
    fields = parse_frontmatter(head)
    origin = fields.get("originSessionId")
    is_index = path.name == "MEMORY.md"

    size = 0
    try:
        size = path.stat().st_size
    except OSError:
        pass

    return MemoryFile(
        path=path,
        project_path=project_path,
        name=sanitize_text(fields.get("name") or path.stem),
        description=sanitize_text(fields.get("description", "")),
        kind=fields.get("type") or ("index" if is_index else "—"),
        origin_session_id=origin,
        origin_exists=bool(origin) and origin in live_sessions,
        size_bytes=size,
        modified=file_mtime(path),
        extra=fields,
    )


def _project_for_folder(folder: Path) -> str | None:
    """Return the project directory a Claude store folder belongs to.

    Read from the ``cwd`` recorded inside one of the folder's transcripts, which
    is exact. The folder name encodes the same path but lossily, so it is used
    only when no transcript can be read — a folder whose sessions were all
    deleted still deserves a best-effort label.

    Args:
        folder: A directory under ``~/.claude/projects``.

    Returns:
        The project directory, or ``None``.
    """
    from sx.adapters.claude import ClaudeAdapter, _decode_folder

    adapter = ClaudeAdapter()
    for transcript in sorted(folder.glob("*.jsonl")):
        cwd = adapter._cwd_from_file(transcript)
        if cwd:
            return cwd
    decoded = _decode_folder(folder.name)
    return decoded if Path(decoded).exists() else None


def discover_memories(root: Path | None = None) -> list[MemoryFile]:
    """Find every memory document across all Claude projects.

    Args:
        root: The projects store to scan; defaults to ``~/.claude/projects``.

    Returns:
        Memories sorted by project, then name.
    """
    projects = root or (home() / ".claude" / "projects")
    if not projects.is_dir():
        return []

    live_sessions = {path.stem for path in projects.glob("*/*.jsonl")}
    found: list[MemoryFile] = []
    for folder in sorted(projects.iterdir()):
        if not folder.is_dir():
            continue
        documents = sorted((folder / "memory").rglob("*.md")) if (folder / "memory").is_dir() else []
        index = folder / "MEMORY.md"
        if index.is_file():
            documents.append(index)
        if not documents:
            continue
        project_path = _project_for_folder(folder)
        for document in documents:
            if document.is_file():
                found.append(_memory_from_file(document, project_path, live_sessions))

    found.sort(key=lambda m: ((m.project_path or "~"), m.name.lower()))
    return found


def memories_for_project(project: str, root: Path | None = None) -> list[MemoryFile]:
    """Return only the memories belonging to one project directory."""
    return [m for m in discover_memories(root) if m.project_path == project]


def export_memory(memory: MemoryFile, dest_dir: Path | None = None) -> Path:
    """Copy a memory document out to a directory, never overwriting.

    Args:
        memory: The memory to export.
        dest_dir: Destination; defaults to ``./session-exports``.

    Returns:
        The path written.
    """
    from sx.export import _unique_path, default_export_dir

    dest = dest_dir or default_export_dir()
    dest.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", memory.name.lower()).strip("-") or "memory"
    project = re.sub(r"[^a-z0-9]+", "-", memory.project_name.lower()).strip("-")
    out_path = _unique_path(dest / f"memory-{project}-{slug}.md")
    shutil.copy2(memory.path, out_path)
    return out_path
