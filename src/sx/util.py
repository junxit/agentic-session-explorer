"""Shared utilities: safe JSONL reading, path guards, sizing, timestamps."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

# C0 control characters and DEL, excluding tab (\x09) and newline (\x0a) which
# are legitimate transcript content. Carriage return is included: it enables
# line-overwriting tricks.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _control_marker(match: re.Match[str]) -> str:
    """Map a control character to its visible Unicode Control Pictures glyph."""
    code = ord(match.group())
    if code == 0x7F:
        return "␡"  # ␡
    return chr(0x2400 + code)  # ␀ … ␟ — e.g. ESC → ␛, BEL → ␇


def sanitize_text(text: str) -> str:
    """Neutralize terminal control sequences in untrusted transcript text.

    Chat logs can contain arbitrary bytes — including content an agent copied
    from a web page. Rich strips only BEL/BS/VT/FF/CR, leaving ESC intact, so an
    unsanitized transcript can drive the user's terminal when merely *viewed*:
    set the window title, rewrite the clipboard via OSC 52, or clear the screen.

    Control characters are replaced with visible, inert glyphs rather than
    dropped, so the transcript still shows that something was there.

    Args:
        text: Untrusted text from a session transcript.

    Returns:
        The text with control characters replaced by Control Pictures glyphs.
    """
    if not text:
        return text
    return _CONTROL_RE.sub(_control_marker, text)


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


#: Directories under which removable/!network volumes are conventionally mounted.
_MOUNT_PARENTS = ("Volumes", "mnt", "media")


def mount_unavailable(path: str | Path) -> bool:
    """Return True if ``path`` lives on a mount point that is not present.

    A project directory on an unplugged external drive is *unavailable*, not
    deleted. Without this distinction a simple ``exists()`` check would classify
    every session on that drive as an orphan and offer to permanently delete
    real transcripts because a disk was unmounted.

    Args:
        path: An absolute path to test.

    Returns:
        True if the path's mount root (e.g. ``/Volumes/Archive``) is absent.
    """
    try:
        parts = Path(path).parts
    except (TypeError, ValueError):
        return False
    if len(parts) >= 3 and parts[1] in _MOUNT_PARENTS:
        try:
            return not Path(*parts[:3]).exists()
        except OSError:
            return False
    return False


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
