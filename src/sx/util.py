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
# line-overwriting tricks. C1 controls (U+0080-U+009F) are included too: some
# terminals act on them, and U+009B on its own is a control sequence
# introducer, the same as ESC [.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")

# Claude Code's own config-home override. When set it replaces ``~/.claude``
# wholesale, and ``.claude.json`` moves inside it.
CLAUDE_CONFIG_ENV = "CLAUDE_CONFIG_DIR"


def _control_marker(match: re.Match[str]) -> str:
    """Map a control character to a visible, inert stand-in.

    A C0 control becomes its Unicode Control Pictures glyph (ESC → ␛,
    BEL → ␇) and DEL becomes ␡. A C1 control has no glyph of its own, so it is
    shown as the 7-bit sequence it stands for, led by the inert ␛ glyph:
    U+009B becomes ``␛[``.
    """
    code = ord(match.group())
    if code == 0x7F:
        return "␡"  # ␡
    if code >= 0x80:
        return "␛" + chr(code - 0x40)  # C1: ESC Fe, e.g. CSI → ␛[
    return chr(0x2400 + code)  # ␀ … ␟ — e.g. ESC → ␛, BEL → ␇


def sanitize_text(text: str) -> str:
    """Neutralize terminal control sequences in untrusted transcript text.

    Chat logs can contain arbitrary bytes — including content an agent copied
    from a web page. Rich strips only BEL/BS/VT/FF/CR, leaving ESC intact, so an
    unsanitized transcript can drive the user's terminal when merely *viewed*:
    set the window title, rewrite the clipboard via OSC 52, or clear the screen.

    Control characters, C0 and C1 alike, are replaced with visible, inert
    glyphs rather than dropped, so the transcript still shows that something
    was there.

    Args:
        text: Untrusted text from a session transcript.

    Returns:
        The text with every control character but tab and newline replaced by
        an inert stand-in.
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
                except (json.JSONDecodeError, ValueError, RecursionError):
                    # RecursionError is not a ValueError: a pathologically
                    # nested line would otherwise abort the whole read.
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

#: A UTF-16 surrogate code point. ``json.loads`` pairs up escaped surrogate
#: pairs, so any surrogate left in a decoded string is a lone one.
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def dumps_record(obj: object) -> bytes:
    """Serialize one JSONL record the way the harnesses themselves write it.

    Every harness ``sx`` supports writes compact JSON — ``JSON.stringify`` in
    the Node-based ones, ``serde_json`` in Codex — with no space after ``,`` or
    ``:``. Python's defaults add both, so a rewritten line would come back
    longer than the harness wrote it and an inverse move could not restore the
    original bytes.

    Lone surrogates (a truncated emoji in tool output, say) are escaped as
    ``\\udxxx``, which is also what ``JSON.stringify`` emits. Left raw they
    cannot be encoded as UTF-8 at all, and the rewrite would fail.

    Args:
        obj: The record to serialize.

    Returns:
        The UTF-8 encoded line, without a line terminator.
    """
    return _dumps_text(obj).encode("utf-8")


def _dumps_text(obj: object) -> str:
    """Serialize a JSON value as :func:`dumps_record` does, as text.

    Args:
        obj: The value to serialize.

    Returns:
        The compact JSON text, with lone surrogates escaped.
    """
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return _SURROGATE_RE.sub(lambda m: f"\\u{ord(m.group()):04x}", text)


#: JSON's insignificant whitespace: the four characters :mod:`json` skips.
_WS_RE = re.compile(r"[ \t\n\r]*")

#: The C scanner behind :func:`json.loads`. Given a string and an index, it
#: parses the one value that starts there and returns it with the index just
#: past it, so the splicer can never disagree with the parser about where a
#: string, number or nested value ends.
_scan_value = json.JSONDecoder().scan_once


class _ObjectEdit:
    """The key-level changes a transform made to one JSON object.

    Attributes:
        replace: Keys whose value changed, mapped to the new value.
        delete: Keys the transform removed.
        add: Keys the transform added, with their values, in insertion order.
    """

    __slots__ = ("add", "delete", "replace")

    def __init__(self) -> None:
        """Start with no changes."""
        self.replace: dict[str, object] = {}
        self.delete: set[str] = set()
        self.add: list[tuple[str, object]] = []


def _same(a: object, b: object) -> bool:
    """Return True if two decoded JSON values would serialize identically.

    Stricter than ``==``, which holds ``1 == 1.0 == True`` and ``0.0 == -0.0``:
    each of those pairs is written differently, so a value changed that way
    must count as changed.

    Args:
        a: A value decoded from the original line.
        b: The value the transform put in its place.

    Returns:
        True if ``b`` can keep ``a``'s original bytes.
    """
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return len(a) == len(b) and all(
            key_a == key_b and _same(value_a, value_b)
            for (key_a, value_a), (key_b, value_b) in zip(a.items(), b.items())
        )
    if isinstance(a, list):
        return len(a) == len(b) and all(map(_same, a, b))
    if isinstance(a, float):
        return repr(a) == repr(b)  # tells -0.0 from 0.0; NaN matches itself
    return a == b


def _diff_object(
    old: dict, new: dict, path: tuple[str, ...], edits: dict[tuple[str, ...], _ObjectEdit]
) -> bool:
    """Record the key-level edits that turn ``old`` into ``new``.

    Objects are compared key by key and recursed into; any other value,
    arrays included, is compared whole. Edits are filed under the path of the
    object they apply to, ``()`` being the record itself.

    Args:
        old: An object as decoded from the original line.
        new: The object the transform returned in its place.
        path: The keys leading from the record to ``old``.
        edits: Collects the edits, keyed by object path (mutated in place).

    Returns:
        False if key edits cannot express the change because ``new`` reorders
        ``old``'s keys or adds a key ahead of one it kept. The caller then
        replaces the object whole; nothing is recorded for it.
    """
    added = [key for key in new if key not in old]
    if [key for key in old if key in new] + added != list(new):
        return False
    edit = _ObjectEdit()
    for key, before in old.items():
        if key not in new:
            edit.delete.add(key)
            continue
        after = new[key]
        if before is after:
            continue
        if isinstance(before, dict) and isinstance(after, dict):
            if not _diff_object(before, after, (*path, key), edits):
                edit.replace[key] = after
        elif not _same(before, after):
            edit.replace[key] = after
    edit.add = [(key, new[key]) for key in added]
    if edit.replace or edit.delete or edit.add:
        edits[path] = edit
    return True


def _queue_edits(
    members: list[tuple[str, int, int, int]],
    close: int,
    edit: _ObjectEdit,
    splices: list[tuple[int, int, str]],
) -> bool:
    """Turn one object's edits into splices of the original text.

    A replaced value is overwritten in place. A deleted key goes with its value
    and exactly one comma: the one before it, or the one after it when it leads
    the object, or none when nothing else is left. An added key goes at the end
    of the object, which is where JavaScript's insertion order puts it.

    Args:
        members: The object's members in text order, each as ``(key, key
            start, value start, value end)``.
        close: The index of the object's closing brace.
        edit: The changes to make.
        splices: Collects ``(start, end, text)`` splices (mutated in place).

    Returns:
        False if an edit names a key the text does not hold.
    """
    where = {member[0]: index for index, member in enumerate(members)}
    if any(key not in where for key in (*edit.replace, *edit.delete)):
        return False
    for key, value in edit.replace.items():
        _, _, value_start, value_end = members[where[key]]
        splices.append((value_start, value_end, _dumps_text(value)))
    gone = {where[key] for key in edit.delete}
    kept = [index for index in range(len(members)) if index not in gone]
    if gone and not kept:
        splices.append((members[0][1], members[-1][3], ""))
    elif gone:
        first = kept[0]
        if first:
            # The leading run goes up to the first survivor, taking the comma
            # after each deleted member.
            splices.append((members[0][1], members[first][1], ""))
        for index in sorted(gone):
            if index > first:
                # From the end of the previous value: the comma before it.
                splices.append((members[index - 1][3], members[index][3], ""))
    if edit.add:
        pairs = ",".join(f"{_dumps_text(key)}:{_dumps_text(value)}" for key, value in edit.add)
        if kept:
            end = members[kept[-1]][3]
            splices.append((end, end, f",{pairs}"))
        else:
            splices.append((close, close, pairs))
    return True


def _splice_record(text: str, old: dict, new: object) -> str | None:
    """Rewrite only the values that differ between ``old`` and ``new``.

    ``old`` is the record decoded from ``text``. Every byte outside the values
    the transform changed is kept, so numbers keep the form their writer gave
    them (``1.5e-7`` rather than Python's ``1.5e-07``), escapes stay escaped
    and spacing stays as it was.

    Args:
        text: The original line, decoded, without its terminator.
        old: The record decoded from ``text``.
        new: The record the transform returned.

    Returns:
        The spliced line, or ``None`` if the edits cannot be placed: ``new`` is
        not an object or reorders the record's own keys (a nested object that
        is reordered is replaced whole instead), an object on an edit's path
        holds a duplicate key, or the nesting is deeper than this recursion
        allows.
    """
    if not isinstance(new, dict):
        return None
    edits: dict[tuple[str, ...], _ObjectEdit] = {}
    descend: set[tuple[str, ...]] = set()
    splices: list[tuple[int, int, str]] = []
    placed: set[tuple[str, ...]] = set()

    def walk(start: int, path: tuple[str, ...]) -> int | None:
        """Map the object at ``start``, queue its edits, return the index past it."""
        members: list[tuple[str, int, int, int]] = []
        seen: set[str] = set()
        pos = _WS_RE.match(text, start + 1).end()
        close = pos if text.startswith("}", pos) else None
        while close is None:
            if not text.startswith('"', pos):
                return None
            key, after_key = _scan_value(text, pos)
            if key in seen:
                return None  # which duplicate counts is the parser's call, not ours
            seen.add(key)
            colon = _WS_RE.match(text, after_key).end()
            if not text.startswith(":", colon):
                return None
            value_start = _WS_RE.match(text, colon + 1).end()
            child = (*path, key)
            if child in descend and text.startswith("{", value_start):
                value_end = walk(value_start, child)
                if value_end is None:
                    return None
            else:
                _, value_end = _scan_value(text, value_start)
            members.append((key, pos, value_start, value_end))
            pos = _WS_RE.match(text, value_end).end()
            if text.startswith(",", pos):
                pos = _WS_RE.match(text, pos + 1).end()
            elif text.startswith("}", pos):
                close = pos
            else:
                return None
        edit = edits.get(path)
        if edit is not None:
            if not _queue_edits(members, close, edit, splices):
                return None
            placed.add(path)
        return close + 1

    try:
        if not _diff_object(old, new, (), edits):
            return None
        descend.update(path[:depth] for path in edits for depth in range(1, len(path) + 1))
        start = _WS_RE.match(text).end()
        end = walk(start, ()) if text.startswith("{", start) else None
    except (ValueError, TypeError, StopIteration, RecursionError):
        return None
    if end is None or placed != edits.keys() or _WS_RE.match(text, end).end() != len(text):
        return None

    parts: list[str] = []
    pos = 0
    for splice_start, splice_end, replacement in sorted(splices, key=lambda s: s[:2]):
        if splice_start < pos:
            return None
        parts.append(text[pos:splice_start])
        parts.append(replacement)
        pos = splice_end
    parts.append(text[pos:])
    return "".join(parts)


def _rewrite_line(body: bytes, old: dict, new: object) -> bytes:
    """Return the bytes a line becomes when its record is replaced.

    The splice is checked before it is used: the spliced line must decode to
    exactly the record the transform returned, down to key order and number
    types. Anything else — a line the splicer cannot map, or a transform that
    edited the record it was given in place, which hides its changes from the
    diff — falls back to serializing the whole record with
    :func:`dumps_record`, as every rewrite did before.

    Args:
        body: The original line, without its terminator. It decoded as UTF-8
            JSON to produce ``old``.
        old: The record decoded from ``body`` and handed to the transform.
        new: The record the transform returned.

    Returns:
        The new line, without a terminator.
    """
    expected = dumps_record(new)
    spliced = _splice_record(body.decode("utf-8"), old, new)
    if spliced is not None:
        try:
            if dumps_record(json.loads(spliced)) == expected:
                return spliced.encode("utf-8")
        except (ValueError, RecursionError):
            pass
    return expected


def rewrite_jsonl(path: Path, transform, *, dry_run: bool = False) -> tuple[int, str | None]:
    """Rewrite a JSONL file record by record, atomically.

    The new content is written to a temporary file in the same directory and
    swapped in with :func:`os.replace`, so an interrupted rewrite leaves the
    original file untouched rather than half-written. The temporary file is
    removed on every path that does not swap it in, including an exception.

    Three details matter for correctness:

    * **Untransformed lines are copied byte for byte.** :func:`iter_jsonl`
      silently drops lines it cannot parse, which is right for reading and fatal
      for rewriting — a rewrite built on it would quietly delete every
      partially-written record in a live session log.
    * **Transformed lines change only where the record did.** Each value the
      transform changed is spliced into the original line, so everything else
      keeps the bytes its writer chose: ``JSON.stringify`` prints ``1.5e-7``
      where Python prints ``1.5e-07``, and re-serializing the whole line would
      stop an inverse move from restoring it. A replaced value is overwritten
      in place, a deleted key goes with one comma, and an added key goes last
      in its object. The value itself is written as :func:`dumps_record`
      writes it, and an object whose keys the transform reorders is written
      whole the same way. A line the splicer cannot map — a duplicate key,
      say — or a transform that edits its argument in place instead of
      returning a new record is re-serialized whole, as before.
    * **The source is re-stat'ed before the swap.** If its size or modification
      time changed while the rewrite was in progress, the harness wrote to it
      and the new copy is already stale, so it is discarded instead of clobbering
      those turns. Size catches an append under a coarse clock; mtime catches a
      same-size rewrite in place, such as Oh My Pi's fixed-width title slot.

    Args:
        path: The JSONL file to rewrite.
        transform: Callable taking one parsed record and returning a replacement
            record, :data:`DROP` to delete the record, or ``None`` to leave that
            line exactly as it was. It should return a new mapping rather than
            edit the one it is given.
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
                        replacement = _rewrite_line(body, obj, updated) + ending
                        changed += 1
                if dropped:
                    continue
                out.write(replacement if replacement is not None else raw)

        if changed == 0:
            return (0, None)

        after = path.stat()
        if after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns:
            return (0, "changed while being rewritten (refused)")

        os.chmod(tmp_path, stat.S_IMODE(before.st_mode))
        os.replace(tmp_path, path)
        tmp_path = None  # swapped in: it is the file now, not a leftover
        return (changed, None)
    except OSError as exc:
        return (0, f"write failed: {exc}")
    finally:
        # Unless it was swapped in, the temp file is a full copy of the
        # transcript; left behind, it would outlive a later delete of the session.
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


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
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
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
