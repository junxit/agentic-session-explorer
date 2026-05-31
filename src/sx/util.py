"""Shared utilities: safe JSONL reading, path guards, sizing, timestamps."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path


def home() -> Path:
    """Return the user's home directory."""
    return Path.home()


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield parsed JSON objects from a JSONL file, skipping bad lines.

    Tolerant by design: malformed or partially written lines (common in
    live-appended session logs) are silently skipped rather than aborting the
    whole read.

    Args:
        path: Path to a ``.jsonl`` file.

    Yields:
        Each successfully parsed line as a ``dict``. Non-dict JSON values are
        skipped.
    """
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(obj, dict):
                    yield obj
    except (OSError, UnicodeError):
        return


def read_first_line(path: Path) -> dict | None:
    """Return the first parseable JSON object in a JSONL file, or ``None``."""
    for obj in iter_jsonl(path):
        return obj
    return None


def parse_ts(value: object) -> datetime | None:
    """Parse an ISO-8601 timestamp into a naive/aware datetime, tolerantly.

    Args:
        value: Typically an ISO string like ``"2026-05-30T00:35:37.638Z"``.

    Returns:
        A ``datetime`` on success, else ``None``.
    """
    if not isinstance(value, str) or not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def file_mtime(path: Path) -> datetime | None:
    """Return a path's modification time as a datetime, or ``None``."""
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return None


def dir_size(path: Path) -> int:
    """Return total size in bytes of a file or directory tree (best effort)."""
    try:
        if path.is_file():
            return path.stat().st_size
        total = 0
        for child in path.rglob("*"):
            try:
                if child.is_file():
                    total += child.stat().st_size
            except OSError:
                continue
        return total
    except OSError:
        return 0


def is_within(path: Path, roots: list[Path]) -> bool:
    """Return True if ``path`` is inside any of ``roots`` (after resolving).

    Used as a hard safety guard: delete operations refuse to touch anything that
    is not contained within a known harness store root.

    Args:
        path: Candidate path to check.
        roots: Allowed root directories.

    Returns:
        True if ``path`` resolves to a location within one of ``roots``.
    """
    try:
        rp = path.resolve()
    except OSError:
        return False
    for root in roots:
        try:
            rp.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def human_size(n: int) -> str:
    """Format a byte count as a short human-readable string (e.g. ``1.2 MB``)."""
    step = 1024.0
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < step:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= step
    return f"{size:.1f} PB"
