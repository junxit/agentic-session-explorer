"""Shared utilities: safe JSONL reading and rewriting, path guards, sizing, timestamps."""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import unicodedata
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

# C0 control characters and DEL, excluding tab (\x09) and newline (\x0a) which
# are legitimate transcript content. Carriage return is included: it enables
# line-overwriting tricks.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

# Claude Code's own config-home override. When set it replaces ``~/.claude``
# wholesale, and ``.claude.json`` moves inside it.
CLAUDE_CONFIG_ENV = "CLAUDE_CONFIG_DIR"


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


def short_path(path: Path) -> str:
    """Render a path for display, collapsing a home-directory prefix to ``~``.

    Args:
        path: The path to render.

    Returns:
        ``~/.claude/todos`` for a path under the home directory, the absolute
        path otherwise.
    """
    try:
        return f"~/{path.relative_to(Path.home())}"
    except ValueError:
        return str(path)


def claude_config_override() -> Path | None:
    """Return the ``CLAUDE_CONFIG_DIR`` override, when it is usable.

    Claude Code reads this variable raw: it applies no tilde expansion and
    refuses to start when the value is not absolute. ``sx`` mirrors both, so a
    relative value is ignored rather than resolved against the working
    directory — that would let a store root, and with it the delete allowlist,
    follow ``cwd``.

    The home directory and the filesystem root are refused for the same reason:
    :func:`sx.adapters.claude.ClaudeAdapter.store_roots` would promote
    ``~/projects`` and its siblings to store roots, putting ordinary user
    directories on the delete allowlist.

    Returns:
        The override directory, or ``None`` when it is unset, blank, relative,
        the home directory or a filesystem root.
    """
    raw = os.environ.get(CLAUDE_CONFIG_ENV, "").strip()
    if not raw:
        return None
    # NFC to match the harness, so a non-ASCII override names the same
    # directory Claude Code created rather than a decomposed twin.
    candidate = Path(unicodedata.normalize("NFC", raw))
    if not candidate.is_absolute():
        return None
    if candidate == Path(candidate.anchor) or candidate == Path.home():
        return None
    return candidate


def claude_config_warning() -> str | None:
    """Return a warning when ``CLAUDE_CONFIG_DIR`` is set but was ignored.

    Silence would be the wrong default: the user has said where Claude's store
    lives, and ``sx`` is about to read — and offer to delete from — somewhere
    else.

    Returns:
        The warning text, or ``None`` when the variable is unset or honored.
    """
    raw = os.environ.get(CLAUDE_CONFIG_ENV, "").strip()
    if not raw or claude_config_override() is not None:
        return None
    candidate = Path(unicodedata.normalize("NFC", raw))
    if not candidate.is_absolute():
        reason = "not an absolute path (Claude Code requires one too)"
    elif candidate == Path(candidate.anchor):
        reason = "a filesystem root"
    else:
        reason = "the home directory"
    return f"{CLAUDE_CONFIG_ENV} is {reason}; ignoring it and reading ~/.claude."


def claude_config_dir(base: Path) -> Path:
    """Return Claude Code's config home.

    Args:
        base: Home directory to fall back to. Passed in rather than resolved
            here: every caller imports ``home`` into its own namespace, and
            resolving it here instead would slip out from under the monkeypatch
            a test applies to that namespace.

    Returns:
        The ``CLAUDE_CONFIG_DIR`` override when usable, else ``base/.claude``.
    """
    return claude_config_override() or (base / ".claude")


def claude_config_file(base: Path) -> Path:
    """Return the path of Claude Code's ``.claude.json``.

    Claude Code joins this file onto ``CLAUDE_CONFIG_DIR`` when that is set and
    onto the home directory otherwise — so under an override it lives *inside*
    the config directory, at ``<override>/.claude.json``, not at
    ``~/.claude.json``.

    Args:
        base: Home directory to fall back to, as for :func:`claude_config_dir`.

    Returns:
        The config file path, whether or not it currently exists.
    """
    override = claude_config_override()
    return (override / ".claude.json") if override else (base / ".claude.json")


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


def _lexical(path: str | Path) -> str:
    """Return a path as a string with redundant trailing separators removed.

    Comparison is deliberately lexical rather than resolved: the old project
    directory usually no longer exists, so ``resolve()`` could not confirm it,
    and resolving would also rewrite symlinked parents into paths the harness
    never recorded.

    Args:
        path: The path to normalize.

    Returns:
        The normalized string form.
    """
    text = str(path)
    while len(text) > 1 and text.endswith("/"):
        text = text[:-1]
    return text


def repoint(value: str, old: str | Path, new: str | Path) -> str | None:
    """Re-root ``value`` from ``old`` to ``new``, or return ``None`` if unrelated.

    Matching is on path boundaries, so moving ``/a/foo`` re-points ``/a/foo`` and
    ``/a/foo/bar`` but leaves ``/a/foobar`` alone. A plain string replace would
    corrupt the second case, and transcripts really do record subdirectories of
    the project — sampling this machine found several sessions carrying five or
    more distinct working directories under one project.

    Comparison is case-sensitive even on case-insensitive filesystems: the old
    path comes from the harness's own records, so it already matches byte for
    byte, and a looser test could re-point a genuinely different directory.

    Args:
        value: A recorded path, typically a session's ``cwd``.
        old: The project directory being moved away from.
        new: The project directory being moved to.

    Returns:
        The re-pointed path, or ``None`` when ``value`` is neither ``old`` nor
        inside it (in which case the caller must leave it untouched).
    """
    if not isinstance(value, str) or not value:
        return None
    current = _lexical(value)
    source = _lexical(old)
    target = _lexical(new)
    if current == source:
        return target
    if current.startswith(source + "/"):
        return target + current[len(source):]
    return None


def is_under(value: str, root: str | Path) -> bool:
    """Return True if ``value`` is ``root`` or lives inside it.

    Shares :func:`repoint`'s boundary-aware comparison, so ``/a/foobar`` is not
    treated as being inside ``/a/foo``.

    Args:
        value: A recorded path.
        root: The directory to test against.

    Returns:
        True if ``value`` is within ``root``.
    """
    return repoint(value, root, root) is not None


class _Drop:
    """Type of the :data:`DROP` sentinel."""

    __slots__ = ()

    def __repr__(self) -> str:
        """Render as ``DROP`` so failed assertions read clearly."""
        return "DROP"


#: Returned by a :func:`rewrite_jsonl` transform to delete that record entirely.
#: Distinct from ``None`` (leave the line exactly as it is) and from a mapping
#: (replace it), so a transform can re-point, drop, and keep in one pass.
DROP = _Drop()


def rewrite_jsonl(path: Path, transform, *, dry_run: bool = False) -> tuple[int, str | None]:
    """Rewrite a JSONL file record by record, atomically.

    The new content is written to a temporary file in the same directory and
    swapped in with :func:`os.replace`, so an interrupted rewrite leaves the
    original file untouched rather than half-written.

    Two details matter for correctness:

    * **Untransformed lines are copied byte for byte.** :func:`iter_jsonl`
      silently drops lines it cannot parse, which is right for reading and fatal
      for rewriting — a rewrite built on it would quietly delete every
      partially-written record in a live session log.
    * **The source is re-stat'ed before the swap.** If its size or modification
      time changed while the rewrite was in progress, the harness appended to it
      and the new copy is already stale, so it is discarded instead of clobbering
      those turns.

    Args:
        path: The JSONL file to rewrite.
        transform: Callable taking one parsed record and returning a replacement
            record, :data:`DROP` to delete the record, or ``None`` to leave that
            line exactly as it was.
        dry_run: If True, count the records that would change and write nothing.

    Returns:
        A ``(records_changed, error)`` pair. ``error`` is ``None`` on success;
        otherwise it explains why the rewrite was abandoned, and nothing was
        changed.
    """
    try:
        before = path.stat()
    except OSError as exc:
        return (0, f"cannot read: {exc}")

    if dry_run:
        changed = 0
        try:
            with path.open("rb") as src:
                for raw in src:
                    obj = _parse_record(raw)
                    if obj is not None and transform(obj) is not None:
                        changed += 1  # a replacement or a DROP; both are changes
        except OSError as exc:
            return (0, f"cannot read: {exc}")
        return (changed, None)

    tmp_path: Path | None = None
    changed = 0
    try:
        handle, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".sx-move-")
        tmp_path = Path(tmp_name)
        with os.fdopen(handle, "wb") as out, path.open("rb") as src:
            for raw in src:
                replacement: bytes | None = None
                obj = _parse_record(raw)
                dropped = False
                if obj is not None:
                    updated = transform(obj)
                    if updated is DROP:
                        dropped = True
                        changed += 1
                    elif updated is not None:
                        body = raw.rstrip(b"\r\n")
                        ending = raw[len(body):] or b"\n"
                        replacement = (
                            json.dumps(updated, ensure_ascii=False).encode("utf-8") + ending
                        )
                        changed += 1
                if dropped:
                    continue
                out.write(replacement if replacement is not None else raw)

        if changed == 0:
            tmp_path.unlink(missing_ok=True)
            return (0, None)

        after = path.stat()
        if after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns:
            tmp_path.unlink(missing_ok=True)
            return (0, "changed while being rewritten (refused)")

        os.chmod(tmp_path, stat.S_IMODE(before.st_mode))
        os.replace(tmp_path, path)
        return (changed, None)
    except OSError as exc:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        return (0, f"write failed: {exc}")


def _parse_record(raw: bytes) -> dict | None:
    """Decode one raw JSONL line into a dict, or ``None`` if it is not one.

    Args:
        raw: The line as read, including its terminator.

    Returns:
        The parsed mapping, or ``None`` for blank, malformed, or non-object
        lines — all of which a rewrite must pass through untouched.
    """
    body = raw.rstrip(b"\r\n")
    if not body:
        return None
    try:
        obj = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def write_text_atomic(path: Path, text: str, *, expect: os.stat_result | None = None) -> str | None:
    """Replace a text file's contents atomically, preserving its mode.

    Used for the small path registries a move updates (Gemini's
    ``.project_root`` marker and ``projects.json``, Claude's ``.claude.json``).
    These belong to a harness that may be running, so they are never truncated in
    place.

    Args:
        path: The file to replace.
        text: The new contents.
        expect: A stat taken before the file was read. When given, the write is
            abandoned if the file changed in the meantime, so a concurrently
            running harness cannot have its own update silently overwritten.

    Returns:
        ``None`` on success, otherwise a reason the write was abandoned.
    """
    tmp_path: Path | None = None
    try:
        mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
        handle, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".sx-move-")
        tmp_path = Path(tmp_name)
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(text)
        if expect is not None:
            after = path.stat()
            if after.st_size != expect.st_size or after.st_mtime_ns != expect.st_mtime_ns:
                tmp_path.unlink(missing_ok=True)
                return "changed while being rewritten (refused)"
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
        return None
    except OSError as exc:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        return f"write failed: {exc}"


def scratchpad_root() -> Path:
    """Return the directory holding Claude's per-session scratchpads.

    Claude Code writes working files to
    ``/private/tmp/claude-<uid>/<encoded-project>/<session-id>/``. The uid is the
    current user's, so the path is derived rather than guessed, and on macOS
    ``/tmp`` is a symlink to ``/private/tmp`` — the real location is preferred so
    the delete guard compares resolved paths consistently.

    Returns:
        The scratchpad root, whether or not it currently exists.
    """
    uid = os.getuid() if hasattr(os, "getuid") else 0
    private = Path("/private/tmp")
    base = private if private.is_dir() else Path("/tmp")
    return base / f"claude-{uid}"


def _ui_state_file() -> Path:
    """Return the UI state path (honors ``XDG_CONFIG_HOME``)."""
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "sx" / "ui.json"


def load_left_pane_width() -> int | None:
    """Return the saved tree width, or ``None`` if absent or unusable."""
    try:
        width = json.loads(_ui_state_file().read_text(encoding="utf-8")).get("left_width")
    except (OSError, ValueError, AttributeError):
        return None
    return width if isinstance(width, int) and not isinstance(width, bool) and width > 0 else None


def save_left_pane_width(width: int) -> None:
    """Persist the tree width; failure to write is ignored (cosmetic state)."""
    path = _ui_state_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"left_width": width}), encoding="utf-8")
    except OSError:
        pass
