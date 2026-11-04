"""Oh My Pi sessions: Pi's format, with a title slot, leases and its own store.

Oh My Pi (CLI ``omp``, ``can1357/oh-my-pi``) is a fork of the Pi coding agent.
It kept Pi's session format — one compact JSON line per tree entry — and
changed nearly everything around it, so :class:`OmpAdapter` builds on
:class:`~sx.adapters.pi.PiFormatAdapter` and overrides what differs. Every rule
below was read from omp 18.15.0's own code (the strings of its Bun bundle) and
its embedded documentation; none was sampled from a real file.

* **Store.** ``~/.omp/agent/sessions/<bucket>/<ts>_<id>.jsonl``, where
  ``PI_CONFIG_DIR`` renames ``.omp``, a profile (``OMP_PROFILE``, else
  ``PI_PROFILE``) selects ``~/.omp/profiles/<p>/agent``, and
  ``PI_CODING_AGENT_DIR`` replaces the agent directory for the default profile
  only. ``$XDG_DATA_HOME/omp`` (or ``…/omp/profiles/<p>``) takes the sessions
  over when it already exists, and ``$XDG_STATE_HOME/omp`` the memories and the
  leases. At startup omp fills variables its environment leaves unset or empty
  from ``<cwd>/.env``, ``<agent dir>/.env``, ``<config root>/.env`` and
  ``~/.env``, the first file to set one winning; ``sx`` reads the last three
  the same way (the first differs from one omp process to the next), as the
  Bun runtime omp runs on parses them. A variable ``sx`` follows that Node's
  parser would read otherwise makes every lease unknown. An agent
  directory named by ``PI_CODING_AGENT_DIR`` is read only when it shows Oh My
  Pi's marks (:func:`~sx.adapters.pi.omp_markers`); without them it is Pi's.
* **Buckets.** Named for the project's real path: ``-<rel>`` under the home
  directory, ``-tmp-<rel>`` under the temp directory (checked first), and
  ``--<abs>--`` elsewhere. Older names still count as the project's. omp
  migrates them: every Pi-style home bucket (``--<home>-…--``) in its sessions
  directory the first time it looks up a bucket for any working directory, and
  a project's other old names (its ``--<abs>--``, hashed and shadowed names)
  when it looks up that project's bucket.
* **Files.** Line 1 is a fixed 256-byte title slot omp rewrites in place; the
  session header is line 2 (legacy files start with the header). Beside
  ``S.jsonl`` sit its artifacts directory ``S/`` (tool output and nested
  sub-agent transcripts), backups ``S.jsonl.<id>.bak`` (omp renames the newest
  back when ``S.jsonl`` is missing), rewrite temp files and lock files. A file
  that begins with a tree entry — no slot, no header — is the debris omp's
  writer leaves when it reopens a transcript deleted or moved under it.
  ``omp gc`` archives cold sessions out of the store, to
  ``…/archive/sessions/<bucket>/S.jsonl.gz`` beside the sessions directory,
  moving ``S/`` along; omp neither lists nor restores them, and sx never
  deletes anything that belongs to one.
* **Liveness.** omp holds an ownership lease — an exclusive ``flock`` on
  ``<run>/session-owners/<id>.lock`` — only while it writes a session; a
  session it has open but has not written to holds nothing. ``sx`` asks the
  kernel whether anyone holds the lease (``F_GETLK``) and never takes it: omp
  tries once, and a session whose lease it cannot get is forked to a new file.
  Every place omp may keep its leases is asked; anything ``sx`` cannot read —
  a lease directory it cannot find next to sessions, a ``.env`` file it cannot
  read the way omp does or whose reading Bun's and Node's parsers disagree on
  — counts as open. Since what places the leases can live in omp's own
  environment alone, the running omp processes are looked at too (``ps``,
  then ``lsof``): a lease file one has open is held wherever it is, and one
  whose open files cannot be listed makes every lease that would read free
  unknown.

Delete, move and cleanup follow omp's own ``deleteSessionWithArtifacts`` and
``moveTo``, with deliberate differences: ``sx`` refuses a session whose lease
is held or cannot be checked (omp's delete checks none); it refuses a file
whose format it does not know rather than guess; it removes a session's
backups first and stops if one cannot go; and a move carries the backups along
— omp's ``moveTo`` leaves them, and its next listing of the old bucket would
restore the session there. A file that does not look the way omp writes it is
still listed, but never deleted, moved or offered for cleanup.
"""

from __future__ import annotations

import bisect
import contextvars
import errno
import hashlib
import json
import os
import re
import stat
import struct
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import IO

try:
    import fcntl
except ImportError:  # pragma: no cover - not POSIX: every lease is unknown
    fcntl = None

from ..model import (
    DeleteResult,
    Message,
    MovePlan,
    MoveResult,
    MoveUnit,
    Orphan,
    OrphanKind,
    ProjectLeftovers,
    Role,
    Session,
    Transcript,
)
from ..util import dir_size, file_mtime, home, is_under, parse_ts, repoint, short_path
from .pi import (
    _CONTROL_RE,
    _TITLE_LIMIT,
    AGENT_DIR_ENV,
    SESSION_DIR_ENV,
    PiFormatAdapter,
    TranscriptKind,
    _folder_transcripts,
    _header_cwd,
    _js_trim,
    _js_truthy,
    _node_join,
    _node_normpath,
    _one_line,
    _parent_of,
    _parse_line,
    _realpath,
    _same_place,
    _truncate,
    _unknown_version,
    _unrecognized_refusal,
    encode_session_folder,
    omp_markers,
)
from .pi import _omp_backup_owner as _backup_owner
from .relocation import Namespace, is_transcript, same_entry

#: Renames the config root, ``~/.omp``. A directory *name*: omp joins it onto
#: the home directory, so even an absolute value lands inside home.
CONFIG_DIR_ENV = "PI_CONFIG_DIR"

#: Profile selectors, in precedence order. ``OMP_PROFILE`` wins whenever it is
#: set, even to an empty value.
PROFILE_ENVS = ("OMP_PROFILE", "PI_PROFILE")

#: Base directories omp moves its data and state under when ``<dir>/omp``
#: already exists.
XDG_DATA_ENV = "XDG_DATA_HOME"
XDG_STATE_ENV = "XDG_STATE_HOME"

#: Replaces the lease directory whenever it is set. Named for omp's tests, but
#: honored in every process.
LEASE_DIR_ENV = "PI_TEST_SESSION_OWNERS_DIR"

#: omp's test-runtime switch (its ``Dn``): set to ``1``, and with no lease
#: directory of its own, omp keeps its leases in
#: ``<tmpdir>/omp-test-session-owners-<uid>`` (its ``ys``).
TEST_RUNTIME_ENV = "PI_TEST_RUNTIME"

#: Results of :meth:`OmpAdapter.lease_state`.
LEASE_FREE = "free"
LEASE_HELD = "held"
LEASE_UNKNOWN = "unknown"

#: What :func:`_probe_lease` and :func:`_flock_state` report for nothing there.
_ABSENT = "absent"

#: How long the TUI's live badge may reuse a scan of the running Oh My Pi
#: processes (:func:`_scan_processes`), counted from when the scan finished —
#: and the store and ``.env`` files it resolved. A delete, a move, and an
#: orphan or leftover delete always scan and resolve afresh.
_SCAN_REUSE_SECONDS = 5.0

#: How long ``ps`` or ``lsof`` may take before the scan gives up, which counts
#: as not knowing.
_SCAN_TIMEOUT_SECONDS = 10.0

#: The same limit for the live badge, which the TUI asks on its own thread: a
#: scan that runs longer gives up and is blind — every lease that would read
#: free reads unknown — until the reuse window passes.
_BADGE_TIMEOUT_SECONDS = 1.0

#: The limit the scan under way gives ``ps`` and ``lsof``; unset means
#: :data:`_SCAN_TIMEOUT_SECONDS`.
_SCAN_LIMIT: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "sx_omp_scan_limit", default=None
)

#: What ``lsof`` escapes in a name it prints: ``\n``, ``\t``, ``\r``, ``\b``,
#: ``\f``, ``\\`` for a backslash, and ``\xNN`` for a byte it does not take for
#: a printable character in its locale. (It prints other control characters
#: as ``^`` and a letter, which cannot be told from those two characters.)
_LSOF_ESCAPE_RE = re.compile(rb"\\(?:x([0-9a-fA-F]{2})|([ntrbf\\]))")
_LSOF_ESCAPES = {b"n": b"\n", b"t": b"\t", b"r": b"\r", b"b": b"\b", b"f": b"\f", b"\\": b"\\"}

#: The locale ``lsof`` runs in, so it prints a UTF-8 name as it is.
_LSOF_LOCALE = "en_US.UTF-8"

#: macOS's process and open-file listers, by absolute path: never looked up on
#: ``PATH``.
_PS = "/bin/ps"
_LSOF = "/usr/sbin/lsof"

#: A command line that may be omp's (see :func:`_runs_omp`).
_OMP_COMMAND_RE = re.compile(r"(?:^|[\s/])omp(?:$|[\s._-])|oh-my-pi", re.IGNORECASE)

#: The title slot's fixed size, its newline included.
SLOT_BYTES = 256

#: The entry types omp 18.15 writes (its ``SessionEntry``). A file that begins
#: with one of them, with no title slot and no header, is what omp's writer
#: leaves when it reopens a transcript that was deleted or moved under it.
OMP_ENTRY_TYPES = frozenset(
    {
        "message",
        "model_usage",
        "thinking_level_change",
        "model_change",
        "service_tier_change",
        "compaction",
        "branch_summary",
        "reset_boundary",
        "custom",
        "custom_message",
        "label",
        "title_change",
        "ttsr_injection",
        "credential_pin",
        "session_init",
        "mode_change",
    }
)

#: Another memory store shares the memories directory; it is never deleted.
_MEMORY_BACKENDS = ("sharpshooter", "mnemopi")

#: A lock file written this recently means omp is writing the session.
_LOCK_WINDOW_SECONDS = 60

#: How old a rewrite temp file must be before it is debris — omp's own ``gc``
#: grace period. A live rewrite renames its temp file within milliseconds.
_TEMP_GRACE_SECONDS = 300

#: How long an artifacts directory without a transcript must sit untouched
#: before it is offered as stale — the day omp's ``gc`` waits on markers whose
#: session file is missing. A new session writes artifacts before its first
#: transcript line, so younger directories may belong to a live session.
_ARTIFACT_IDLE_SECONDS = 24 * 3600

#: Memory written this recently is neither deleted nor moved: omp's memory
#: pipeline runs in the background after a session ends.
_MEMORY_IDLE_SECONDS = 90

#: An artifacts tree with more entries than this is never called idle: the
#: scan stops, and not knowing counts as in use.
_IDLE_SCAN_LIMIT = 20_000

#: omp reads at most this much of a file to find its session header; ``sx``
#: reads no more of a header-less file's first line to call it debris.
_LINE_LIMIT = 1024 * 1024

#: How much of a transcript omp's session list reads for a title
#: (``readTextSlices(path, 4096, …)``).
_TITLE_HEAD_BYTES = 4096

#: What omp's session list calls a session without a first message.
_NO_MESSAGES = "(no messages)"

#: The most of a ``.env`` file read; omp itself reads any size.
_DOTENV_LIMIT = 1024 * 1024

#: The variables omp fills from a ``.env`` file that move what ``sx`` reads.
#: ``PI_CODING_AGENT_DIR`` is not among them: omp reads it before the files.
_DOTENV_KEYS = (
    CONFIG_DIR_ENV,
    *PROFILE_ENVS,
    XDG_DATA_ENV,
    XDG_STATE_ENV,
    LEASE_DIR_ENV,
    TEST_RUNTIME_ENV,
    SESSION_DIR_ENV,
)

#: The names in a ``.env`` file that set a variable of :data:`_DOTENV_KEYS`:
#: each of them, and the ``OMP_`` name omp copies to each ``PI_`` one.
_DOTENV_NAMES = tuple(
    dict.fromkeys(
        [*_DOTENV_KEYS, *(f"OMP_{key[3:]}" for key in _DOTENV_KEYS if key.startswith("PI_"))]
    )
)

#: The entry id that stands for omp's empty leaf (``activeLeafId: null``).
#: Never a real id: real ones are short hex strings or UUIDs.
_ROOT_LEAF = "\x00sx:oh-my-pi-root"

#: The empty-branch view's stand-in entry; :meth:`entry_message` shows nothing.
_ROOT_ENTRY_TYPE = "sx:root"

#: omp's draft marker: ``S/`` holds an unsent prompt, not debris.
_DRAFT_MARKER = ".draft-only-session"

#: Where ``omp gc`` archives cold sessions, beside the sessions directory:
#: ``<its parent>/archive/sessions/<bucket>/S.jsonl.gz``, with ``S/`` moved next
#: to it. omp neither lists nor restores what it archives.
_ARCHIVE_PARTS = ("archive", "sessions")

#: An archived transcript's suffix.
_ARCHIVE_SUFFIX = ".jsonl.gz"

#: ``omp gc --compress`` stages compressed copies in such a directory beside
#: the files it compresses, and removes leftovers on its next run.
_COMPRESS_SCRATCH = ".omp-gc-compress-"

#: Entry types omp writes that hold nothing worth showing as a turn.
_SKIPPED_ENTRIES = frozenset(
    {
        "model_usage",
        "service_tier_change",
        "reset_boundary",
        "title_change",
        "ttsr_injection",
        "credential_pin",
        "session_init",
        "mode_change",
    }
)

_PROFILE_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_RESERVED_RE = re.compile(r"(?:CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])(?:\..*)?", re.IGNORECASE)
_LEASE_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,128}")
_SESSION_NAME_RE = re.compile(
    r"([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2})-([0-9]{2})-([0-9]{2})-([0-9]{3})Z_(.+)\.jsonl"
)
_TEMP_RE = re.compile(r"\.(.+)\.[0-9a-f]{16}\.tmp")
_SEPARATOR_RE = re.compile(r"[/\\:]")
_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_DOTENV_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

#: What Node's dotenv parser trims from keys, values and lines.
_DOTENV_BLANKS = " \t\n"
_DOTENV_NOT_BLANK_RE = re.compile(r"[^ \t\n]")
_DOTENV_KEY_END_RE = re.compile(r"[=\n]")

#: What Bun's dotenv parser skips between tokens and trims from values: tab,
#: vertical tab, form feed, space, newline and carriage return.
_BUN_BLANKS = "\t\x0b\x0c \n\r"
_BUN_NOT_BLANK_RE = re.compile(r"[^\t\x0b\x0c \n\r]")
_BUN_KEY_RE = re.compile(r"[A-Za-z0-9_.-]+")
_BUN_LINE_END_RE = re.compile(r"[\n\r]")
_BUN_VALUE_END_RE = re.compile(r"[#\n\r]")
_BUN_QUOTES = "\"'`"
#: Where a scan for a value's closing quote stops: at the quote, or at a
#: backslash, which escapes the character after it.
_BUN_QUOTE_STOPS = {quote: re.compile(f"[\\\\{quote}]") for quote in _BUN_QUOTES}
_BUN_DOUBLE_RE = re.compile(r"\\([\s\S])|\r\n?")
_BUN_CR_RE = re.compile(r"\r\n?")

#: Whether omp keeps its leases in files sx can probe. On Linux omp holds an
#: abstract socket instead, which creates no file: a probe there would always
#: find the lease free, so every lease is unknown.
_LEASES_ARE_FILES = not sys.platform.startswith("linux")

#: ``struct flock`` as ``F_GETLK`` reads and writes it, on the platforms where
#: its layout was checked: macOS's ``<sys/fcntl.h>`` declares ``off_t l_start``,
#: ``off_t l_len``, ``pid_t l_pid``, ``short l_type``, ``short l_whence`` — 24
#: bytes, verified on Darwin 27 (arm64) with ``offsetof`` and against a live
#: lock. Elsewhere the layout differs (Linux and FreeBSD put other fields
#: first, or add one), so ``None``: every lease is unknown.
_FLOCK_LAYOUT = "@qqihh" if sys.platform == "darwin" else None

#: File layouts :func:`_layout` reports.
_SLOT = "slot"
_LEGACY = "legacy"
_SLOT_SIZE = "slot of the wrong size"
_UNKNOWN = "unknown"

_OPEN = "Oh My Pi has this session open (refused)"
_UNSURE = "cannot tell whether Oh My Pi has this session open (refused)"
_WRITING = "Oh My Pi is writing this session (refused)"

#: Why a transcript whose layout ``sx`` does not know is never destroyed.
_LAYOUT_REFUSAL = (
    "does not begin the way Oh My Pi writes a session (a title slot, then the "
    "header); the format may have changed (refused)"
)
_SLOT_REFUSAL = (
    f"its title slot is not {SLOT_BYTES} bytes, which sx does not know, so it is "
    "neither deleted nor rewritten (refused)"
)


# --- small helpers -------------------------------------------------------------


def _node_resolve(path: str) -> str:
    """Resolve ``path`` lexically, the way Node's ``path.resolve`` does.

    Args:
        path: A path, absolute or relative to the working directory.

    Returns:
        The absolute, normalized path. Symlinks are not resolved.
    """
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)
    return _node_normpath(path)


def _node_tmpdir() -> str:
    """Return the temp directory Node's ``os.tmpdir()`` reports on POSIX.

    Returns:
        ``TMPDIR``, else ``TMP``, else ``TEMP``, else ``/tmp``, with one
        trailing ``/`` removed.
    """
    path = os.environ.get("TMPDIR") or os.environ.get("TMP") or os.environ.get("TEMP") or "/tmp"
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    return path


def _node_relative(start: str, target: str) -> str:
    """Return ``target`` relative to ``start``, as Node's ``path.relative`` does.

    Args:
        start: An absolute, normalized directory.
        target: An absolute, normalized path.

    Returns:
        The relative path; an empty string — not ``.`` — when they are equal.
    """
    if start == target:
        return ""
    relative = os.path.relpath(target, start)
    return "" if relative == "." else relative


def _inside(relative: str) -> bool:
    """Return True if a relative path stays inside its base, by omp's test.

    omp checks the text: a first segment such as ``..foo`` counts as leaving
    the base, exactly as ``..`` does.

    Args:
        relative: A result of :func:`_node_relative`.

    Returns:
        True for the base itself or anything beneath it.
    """
    return relative == "" or (not relative.startswith("..") and not os.path.isabs(relative))


def _relative_name(prefix: str, relative: str) -> str:
    r"""Build a home- or temp-relative bucket name (omp's ``di``).

    Args:
        prefix: ``-`` or ``-tmp``.
        relative: The project's path relative to that directory.

    Returns:
        The prefix alone for the directory itself, else the prefix and the
        relative path with ``/``, ``\`` and ``:`` turned into ``-``.
    """
    encoded = _SEPARATOR_RE.sub("-", relative)
    if not encoded:
        return prefix
    return f"{prefix}{encoded}" if prefix.endswith("-") else f"{prefix}-{encoded}"


def _hashed_name(real: str, kind: str) -> str:
    """Build the hashed bucket name omp used briefly in 17.2.5–17.2.8.

    Args:
        real: The project's real path.
        kind: ``home``, ``tmp`` or ``abs``.

    Returns:
        ``<kind>-<slug>-<sha256 of the path>``.
    """
    digest = hashlib.sha256(real.replace("\\", "/").encode("utf-8", "surrogateescape"))
    slug = _SLUG_RE.sub("-", os.path.basename(real)).strip("-")[-80:]
    return f"{kind}-{slug or 'project'}-{digest.hexdigest()}"


@dataclass(frozen=True, slots=True)
class _Buckets:
    """Every folder name omp has used for one project directory.

    Attributes:
        encoded: Today's name.
        hashed: The short-lived hashed name.
        shadowed_home: The home-relative name, when the temp directory lies
            inside home and the project inside both.
        shadowed_home_hashed: Its hashed form.
        resolved: The project path, resolved lexically.
        real: The project path with symlinks resolved.
    """

    encoded: str
    hashed: str
    shadowed_home: str | None
    shadowed_home_hashed: str | None
    resolved: str
    real: str


def _buckets(project: str, base: Path) -> _Buckets:
    """Compute omp's bucket names for ``project`` (its ``pi()``).

    The temp directory is checked before home, so a project in a temp
    directory inside home is named for the temp directory.

    Args:
        project: The project directory.
        base: The home directory.

    Returns:
        The names.
    """
    resolved = _node_resolve(project)
    real = _realpath(resolved)
    home_real = _realpath(str(base))
    tmp_real = _realpath(_node_resolve(_node_tmpdir()))
    from_home = _node_relative(home_real, real)
    from_tmp = _node_relative(tmp_real, real)
    in_home = _inside(from_home)
    shadowed = shadowed_hashed = None
    if _inside(from_tmp):
        encoded = _relative_name("-tmp", from_tmp)
        if in_home:
            shadowed = _relative_name("-", from_home)
            shadowed_hashed = _hashed_name(real, "home")
        kind = "tmp"
    elif in_home:
        encoded = _relative_name("-", from_home)
        kind = "home"
    else:
        encoded = encode_session_folder(real)
        kind = "abs"
    return _Buckets(encoded, _hashed_name(real, kind), shadowed, shadowed_hashed, resolved, real)


@dataclass(frozen=True, slots=True)
class _MemoryMove:
    """A memory folder a move relocates, shaped as the core orders its own moves.

    Attributes:
        folder: The memory folder.
        destination: Where it goes.
        kind: :attr:`~sx.model.MoveUnit.FOLDER`: a whole folder, vacated
            before anything else moves into its name.
    """

    folder: Path
    destination: Path
    kind: str = MoveUnit.FOLDER


def memory_folder_name(cwd: str) -> str:
    r"""Return the folder omp keeps a working directory's memory in (its ``HEe``).

    Applied to the recorded directory as is — no normalization, no symlink
    resolution — exactly as omp does.

    Args:
        cwd: A session's recorded working directory.

    Returns:
        ``--<cwd, leading separator dropped, / \ : turned into ->--``.
    """
    text = cwd[1:] if cwd[:1] in ("/", "\\") else cwd
    return f"--{_SEPARATOR_RE.sub('-', text)}--"


def _name_id(name: str) -> str | None:
    """Return the session id in an omp transcript name, if it has one.

    Args:
        name: A file name such as ``2026-05-01T10-00-00-000Z_<id>.jsonl``.

    Returns:
        The id, or ``None`` for a name omp did not generate.
    """
    match = _SESSION_NAME_RE.fullmatch(name)
    return match.group(5) if match else None


def _entry_session_id(name: str) -> str | None:
    """Return the session id an entry beside the transcripts is named for.

    Args:
        name: An artifacts directory (``<ts>_<id>``) or a file named for a
            transcript, such as a backup (``<ts>_<id>.jsonl.<x>.bak``).

    Returns:
        The id, or ``None`` when the name holds none omp would have written.
    """
    marker = name.find(".jsonl.")
    if marker > 0:
        return _name_id(name[: marker + len(".jsonl")])
    return _name_id(f"{name}.jsonl")


def _owns_backup(transcript: str, name: str) -> bool:
    """Return True if ``name`` is one of transcript ``transcript``'s backups.

    That is every ``<transcript>.<anything>.bak`` — what omp's own delete
    removes with the session — not only the ones omp would restore.

    Args:
        transcript: A transcript's file name, ``S.jsonl``.
        name: Another file name in the same bucket.

    Returns:
        Whether deleting or moving the session takes it too.
    """
    return name.startswith(f"{transcript}.") and name.endswith(".bak")


def _temp_owner(name: str) -> str | None:
    """Return the transcript a rewrite temp file belongs to.

    Args:
        name: A file name such as ``.S.jsonl.0123456789abcdef.tmp``.

    Returns:
        The transcript's file name, or ``None`` if ``name`` is no temp file.
    """
    match = _TEMP_RE.fullmatch(name)
    return match.group(1) if match else None


def _is_lock_name(name: str) -> bool:
    """Return True for a lock file omp keeps beside its transcripts."""
    return name.endswith((".lock", ".lock.os"))


def _modified_within(path: Path, seconds: float, now: float) -> bool:
    """Return True if ``path`` changed within ``seconds`` of ``now``.

    Args:
        path: Any path; symlinks are not followed.
        seconds: The window.
        now: The current time, as from :func:`time.time`.

    Returns:
        False when it does not exist; True when it is recent or cannot be
        examined — not knowing counts as in use.
    """
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return now - info.st_mtime < seconds


def _idle(directory: Path, seconds: float) -> bool:
    """Return True if nothing in a directory tree changed within ``seconds``.

    Args:
        directory: The tree to examine; symlinks are not followed.
        seconds: How long it must have been untouched.

    Returns:
        True only when every entry was examined and all are older.
    """
    now = time.time()
    if _modified_within(directory, seconds, now):
        return False
    seen = 0
    for top, dirs, files in os.walk(directory, onerror=_raise):
        for name in (*dirs, *files):
            seen += 1
            if seen > _IDLE_SCAN_LIMIT or _modified_within(Path(top, name), seconds, now):
                return False
    return True


def _safe_idle(directory: Path, seconds: float) -> bool:
    """Return :func:`_idle`, or False when the tree cannot be read in full."""
    try:
        return _idle(directory, seconds)
    except OSError:
        return False


def _raise(error: OSError) -> None:
    """Re-raise an :func:`os.walk` error, so a partial scan is never trusted."""
    raise error


def _first_line(text: str) -> str | None:
    """Return the first line of a title, as omp's session list cleans it (its ``kt``).

    Args:
        text: A recorded title or a first message.

    Returns:
        The first line without control characters, trimmed, or ``None`` when
        nothing is left — also when the first line is blank and a later one
        is not, as in omp.
    """
    line = text.split("\n", 1)[0]
    cleaned = _js_trim(_CONTROL_RE.sub("", line))
    return cleaned or None


def _next_value(fh: IO[bytes]) -> object:
    """Return the next parsed JSON value in an open JSONL file.

    Blank and malformed lines are passed over, and so are values JavaScript
    treats as false, as Pi's header reader does.

    Args:
        fh: The file, opened in binary mode.

    Returns:
        The value, or ``None`` at the end of the file.
    """
    for raw in fh:
        value = _parse_line(raw)
        if _js_truthy(value):
            return value
    return None


def _is_header(value: object) -> bool:
    """Return True for a session header: ``type`` ``session`` and a string id."""
    return (
        isinstance(value, dict)
        and value.get("type") == "session"
        and isinstance(value.get("id"), str)
    )


def _slot_and_header(path: Path) -> tuple[dict | None, dict | None]:
    """Read a transcript's title slot and session header.

    Lenient, like omp's session list: a first record of type ``title`` is
    taken as the slot whatever its shape, and the header must follow it.
    :func:`_layout` is the strict check destructive operations rely on.

    Args:
        path: A ``.jsonl`` file.

    Returns:
        ``(slot, header)``; either is ``None`` when absent.
    """
    try:
        with path.open("rb") as fh:
            first = _next_value(fh)
            if isinstance(first, dict) and first.get("type") == "title":
                second = _next_value(fh)
                return first, second if _is_header(second) else None
            return None, first if _is_header(first) else None
    except OSError:
        return None, None


def _valid_slot(value: dict) -> bool:
    """Return True for a title slot omp 18.15 loads (its ``tl`` check).

    Args:
        value: The first record of a transcript, of type ``title``.

    Returns:
        Whether it has version 1, string title, update time and padding, and
        either no source or ``auto``/``user``.
    """
    version = value.get("v")
    if isinstance(version, bool) or not isinstance(version, (int, float)) or version != 1:
        return False
    if not all(isinstance(value.get(key), str) for key in ("title", "updatedAt", "pad")):
        return False
    return "source" not in value or value["source"] in ("auto", "user")


def _layout(path: Path) -> str:
    """Classify how a transcript begins, byte for byte.

    Args:
        path: A transcript.

    Returns:
        :data:`_SLOT` — a valid 256-byte slot, then the header; :data:`_LEGACY`
        — the header first; :data:`_SLOT_SIZE` — a valid slot of another size;
        :data:`_UNKNOWN` — anything else, an unreadable file included.
    """
    try:
        with path.open("rb") as fh:
            first = fh.readline(_LINE_LIMIT + 1)
            second = fh.readline(_LINE_LIMIT + 1) if first else b""
    except OSError:
        return _UNKNOWN
    head = _parse_line(first)
    if _is_header(head):
        return _LEGACY
    if not isinstance(head, dict) or head.get("type") != "title":
        return _UNKNOWN
    if not _valid_slot(head) or not _is_header(_parse_line(second)):
        return _UNKNOWN
    if len(first) != SLOT_BYTES or not first.endswith(b"\n"):
        return _SLOT_SIZE
    return _SLOT


def _leaf_target(entry: dict) -> object:
    """Return the leaf an entry leaves the session on (omp's ``Xj``).

    Args:
        entry: A tree entry.

    Returns:
        Its ``activeLeafId`` when the key is present — ``None`` for the empty
        leaf — otherwise its own id.
    """
    return entry["activeLeafId"] if "activeLeafId" in entry else entry.get("id")


def _resume_leaf(entries: list[dict]) -> str | None:
    """Return the leaf omp resumes from (its ``YT`` and ``Pbe``).

    Args:
        entries: The tree entries, in file order.

    Returns:
        The leaf id, or ``None`` for the empty leaf — nothing on the current
        branch.
    """
    if not entries:
        return None
    last = entries[-1]
    target = _leaf_target(last)
    if target is None or target == last.get("id") or any(e.get("id") == target for e in entries):
        return target if isinstance(target, str) else None
    for entry in reversed(entries):
        if _leaf_target(entry) == entry.get("id"):
            found = entry.get("id")
            return found if isinstance(found, str) else None
    return None


def _python_message(message: dict, timestamp: datetime | None) -> Message:
    """Render a standalone Python run, the way a ``!`` bash run is shown.

    Args:
        message: A ``pythonExecution`` message.
        timestamp: The entry's timestamp.

    Returns:
        A :attr:`Role.TOOL` message.
    """
    code = message.get("code")
    summary = (
        "python: " + _truncate(_one_line(code), 120)
        if isinstance(code, str) and code.strip()
        else "python"
    )
    output = message.get("output")
    text = _truncate(output, 200) if isinstance(output, str) else ""
    exit_code = message.get("exitCode")
    if isinstance(exit_code, int) and not isinstance(exit_code, bool):
        text = f"{text}\n(exit code {exit_code})" if text else f"(exit code {exit_code})"
    return Message(Role.TOOL, text=text, timestamp=timestamp, tool_summary=summary)


def _file_mention(message: dict, timestamp: datetime | None) -> Message | None:
    """Render inlined ``@file`` mentions as the files they name.

    Contents are left out — they can be whole files — and images show as
    ``image``; nothing is read from omp's blob store.

    Args:
        message: A ``fileMention`` message.
        timestamp: The entry's timestamp.

    Returns:
        An :attr:`Role.OTHER` message, or ``None`` when no file is named.
    """
    files = message.get("files")
    lines: list[str] = []
    for item in files if isinstance(files, list) else []:
        path = item.get("path") if isinstance(item, dict) else None
        if not isinstance(path, str) or not path:
            continue
        skipped = item.get("skippedReason")
        count = item.get("lineCount")
        if isinstance(skipped, str) and skipped:
            detail = f"skipped: {skipped}"
        elif _js_truthy(item.get("image")):
            detail = "image"
        elif isinstance(count, int) and not isinstance(count, bool):
            detail = f"{count} line" if count == 1 else f"{count} lines"
        else:
            detail = ""
        lines.append(f"Read {path} ({detail})" if detail else f"Read {path}")
    if not lines:
        return None
    return Message(Role.OTHER, text="\n".join(lines), timestamp=timestamp)


# --- titles ------------------------------------------------------------------------


def _head_records(path: Path) -> tuple[str, list[dict]]:
    """Read what omp's session list reads for a title: a file's first 4096 bytes.

    Args:
        path: A transcript.

    Returns:
        ``(text, records)``: those bytes decoded as UTF-8 (a character cut in
        two becomes U+FFFD), and every line in them that parses as a JSON
        object, in order — a line cut off at the end does not. Empty when the
        file cannot be read.
    """
    try:
        with path.open("rb") as fh:
            head = fh.read(_TITLE_HEAD_BYTES)
    except OSError:
        return "", []
    text = head.decode("utf-8", errors="replace")
    records: list[dict] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except (ValueError, RecursionError):
            continue
        if isinstance(value, dict):
            records.append(value)
    return text, records


def _text_blocks(content: object) -> str:
    """Join a message's text the way omp's session list does (its ``mu(content, " ")``).

    Args:
        content: A message's ``content``.

    Returns:
        A string content as it is, or every text block's text joined with a
        space — an empty one included — and ``""`` for anything else.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return " ".join(
        block["text"] if isinstance(block.get("text"), str) else ""
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _unescape(raw: str) -> str:
    """Decode a JSON string body the way omp's text scan does (its ``po``).

    Args:
        raw: The characters between the quotes, possibly cut short.

    Returns:
        The decoded string; a body JSON cannot decode has its common escapes
        replaced by hand.
    """
    body = raw.removesuffix("\\")
    try:
        value = json.loads(f'"{body}"')
    except (ValueError, RecursionError):
        value = None
    if isinstance(value, str):
        return value
    for escape, char in (("\\n", "\n"), ("\\r", "\r"), ("\\t", "\t"), ('\\"', '"'), ("\\\\", "\\")):
        body = body.replace(escape, char)
    return body


def _scan_value(text: str, key: str, start: int = 0) -> str | None:
    """Find the string after the next ``"key":`` in raw text (omp's ``Be``).

    Args:
        text: The text to scan.
        key: The key, without quotes.
        start: Where to start looking.

    Returns:
        The value when the key is followed by a string — what of it there is,
        when the text ends inside it — else ``None``.
    """
    found = text.find(f'"{key}"', start)
    if found < 0:
        return None
    colon = text.find(":", found + len(key) + 2)
    if colon < 0:
        return None
    position = colon + 1
    while position < len(text) and text[position] in " \t\n\r":
        position += 1
    if position >= len(text) or text[position] != '"':
        return None
    begin = position + 1
    escaped = False
    for index in range(begin, len(text)):
        char = text[index]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return _unescape(text[begin:index])
    return _unescape(text[begin:])


def _scanned_message(text: str) -> str | None:
    """Find a first message in raw text, as omp's session list falls back to (its ``Cc``).

    Used when no user message parsed in the first 4096 bytes — typically
    because the first one is too long to end in them.

    Args:
        text: The first 4096 bytes of a transcript.

    Returns:
        The content (or first text block) after the first ``"role":"user"``;
        failing that, after the first developer or assistant role; else
        ``None``.
    """
    fallback: str | None = None
    position = text.find('"role"')
    while position >= 0:
        role = _scan_value(text, "role", position)
        value = _scan_value(text, "content", position)
        if value is None:
            value = _scan_value(text, "text", position)
        if value:
            if role == "user":
                return value
            if fallback is None and role in ("developer", "assistant"):
                fallback = value
        position = text.find('"role"', position + 6)
    return fallback


def _twelve_hour_clock() -> bool:
    """Return True if omp's time-of-day format would use a 12-hour clock here.

    omp formats with ``toLocaleTimeString`` in its runtime's default locale,
    which ``sx`` cannot query. It approximates that locale with the POSIX
    variables for time formats — ``LC_ALL``, else ``LC_TIME``, else ``LANG`` —
    and treats US English (``en_US``, or ``en`` alone) as the 12-hour locale;
    everything else, an unset or ``C`` locale included, gets 24 hours.

    Returns:
        Whether to show ``09:05 AM`` rather than ``09:05``.
    """
    for variable in ("LC_ALL", "LC_TIME", "LANG"):
        value = os.environ.get(variable)
        if value:
            break
    else:
        return False
    language = value.split(".", 1)[0].split("@", 1)[0].replace("-", "_")
    return language in ("en", "en_US")


def _clock(moment: datetime) -> str:
    """Format a time of day as omp's untitled placeholder does, for this locale.

    Args:
        moment: A local time.

    Returns:
        ``09:05 AM`` / ``09:05 PM`` for a 12-hour locale (see
        :func:`_twelve_hour_clock`), else ``09:05``. omp's own text may set a
        narrow no-break space before the period; ``sx`` uses a plain one.
    """
    if _twelve_hour_clock():
        hour = moment.hour % 12 or 12
        return f"{hour:02d}:{moment.minute:02d} {'AM' if moment.hour < 12 else 'PM'}"
    return f"{moment:%H:%M}"


# --- leases and lock files -----------------------------------------------------------


def _flock_state(path: Path) -> str:
    """Ask the kernel whether anyone holds a lock on ``path``, taking none.

    The file is opened read-only, without following a link or creating
    anything, and queried with ``F_GETLK`` for an exclusive lock over the whole
    file: any lock another holder has — omp's ``flock``, a ``lockf``, shared or
    exclusive — conflicts with that and is reported. Nothing is acquired, so a
    holder trying to take the lock at the same moment always gets it.

    Args:
        path: A lock file.

    Returns:
        :data:`LEASE_HELD`, :data:`LEASE_FREE`, :data:`_ABSENT` when nothing is
        there, or :data:`LEASE_UNKNOWN` — for anything but a regular file, a
        file replaced while it was opened, an error, an answer ``sx`` cannot
        read, and a platform whose ``struct flock`` ``sx`` does not know.
    """
    if fcntl is None or _FLOCK_LAYOUT is None:
        return LEASE_UNKNOWN
    try:
        before = os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return _ABSENT
    except (OSError, ValueError):
        return LEASE_UNKNOWN
    if not stat.S_ISREG(before.st_mode):
        return LEASE_UNKNOWN
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    try:
        fd = os.open(path, flags)
    except (FileNotFoundError, NotADirectoryError):
        return _ABSENT
    except OSError:
        return LEASE_UNKNOWN
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            return LEASE_UNKNOWN
        query = struct.pack(_FLOCK_LAYOUT, 0, 0, 0, fcntl.F_WRLCK, os.SEEK_SET)
        try:
            answer = fcntl.fcntl(fd, fcntl.F_GETLK, query)
        except OSError:
            return LEASE_UNKNOWN
    finally:
        os.close(fd)
    kind = struct.unpack(_FLOCK_LAYOUT, answer)[3]
    if kind == fcntl.F_UNLCK:
        return LEASE_FREE
    if kind in (fcntl.F_RDLCK, fcntl.F_WRLCK):
        return LEASE_HELD
    return LEASE_UNKNOWN


def _probe_lease(directory: Path, session_id: str) -> str:
    """Probe one session's ownership lease in one lease directory, without running omp.

    On macOS and other non-Linux Unix, omp holds the lease as an exclusive
    ``flock`` on ``<directory>/<id>.lock``, inside a private directory it
    checks the same way; :func:`_flock_state` asks about it without taking it.

    Args:
        directory: A lease directory.
        session_id: An id omp uses as is.

    Returns:
        :data:`LEASE_FREE`, :data:`LEASE_HELD` or :data:`LEASE_UNKNOWN`; a
        missing lease file in a private directory is free, as in omp's own
        check, a directory that is not private is unknown — omp runs without a
        lease there — and :data:`_ABSENT` means there is no such directory.
    """
    try:
        info = os.lstat(directory)
    except (FileNotFoundError, NotADirectoryError):
        return _ABSENT
    except (OSError, ValueError):
        return LEASE_UNKNOWN
    owner = os.getuid() if hasattr(os, "getuid") else info.st_uid
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != owner or info.st_mode & 0o077:
        return LEASE_UNKNOWN  # omp refuses such a directory too
    state = _flock_state(directory / f"{session_id}.lock")
    return LEASE_FREE if state == _ABSENT else state


def _lock_reason(path: Path) -> str | None:
    """Return why omp may be writing ``path`` right now, judged by its locks.

    The publish lock ``.S.jsonl.lock`` exists only while omp publishes (it is
    removed straight after), so any one counts. The OS lock files
    ``.S.jsonl.lock.os`` and ``S.jsonl.lock`` outlive their use, so they count
    while someone holds them (asked as for a lease, without taking them), when
    they cannot be checked, or when they were written recently.

    Args:
        path: A transcript.

    Returns:
        A refusal reason, or ``None``.
    """
    folder, name = path.parent, path.name
    if os.path.lexists(folder / f".{name}.lock"):
        return _WRITING
    now = time.time()
    for lock in (folder / f".{name}.lock.os", folder / f"{name}.lock"):
        if _flock_state(lock) not in (LEASE_FREE, _ABSENT):
            return _WRITING
        if _modified_within(lock, _LOCK_WINDOW_SECONDS, now):
            return _WRITING
    return None


# --- the lease files running omp processes hold -------------------------------------


@dataclass(frozen=True, slots=True)
class _ProcessScan:
    """What the running omp processes have open, as one scan found it.

    The variables that place omp's leases may be set in omp's own environment
    alone — a shell's ``XDG_STATE_HOME``, ``PI_TEST_RUNTIME``, a ``.env`` in
    the directory omp was started from — where ``sx`` cannot see them. The
    lease files omp holds open can be seen, wherever they are.

    Attributes:
        held: Session ids whose lease file (``<id>.lock``) an omp process has
            open: omp holds a lease only while the file is open.
        dirs: The private directories those lease files are in — lease
            directories in use, asked about for every other session too.
        blind: True when an omp process may be running whose open files could
            not be listed, or when the processes could not be listed at all.
        taken: When the scan finished, as from :func:`time.monotonic`.
    """

    held: frozenset[str] = frozenset()
    dirs: tuple[Path, ...] = ()
    blind: bool = False
    taken: float = 0.0


def _runs_omp(command: str) -> bool:
    """Return True if a process's command line may be Oh My Pi's.

    omp ships as a Bun binary named ``omp``; installed from npm it runs as
    ``bun …/omp`` (its ``bin`` shim) or from a path under ``@oh-my-pi/``.
    ``ps`` joins arguments with spaces, so the line is matched rather than
    parsed: ``omp`` as a command, or as the last part of a path, anywhere in
    it, or ``oh-my-pi`` anywhere — under ``bun``, ``node`` or anything else. A
    false match only costs a look at that process's open files.

    Args:
        command: The command line, as ``ps`` shows it.

    Returns:
        Whether to look at the process's open files.
    """
    return _OMP_COMMAND_RE.search(command) is not None


def _parse_ps(output: bytes) -> list[tuple[int, int, str]]:
    """Parse ``ps -o pid=,uid=,args=`` output.

    Args:
        output: What ``ps`` printed.

    Returns:
        ``(pid, uid, command line)`` for every line that names a process.
    """
    found: list[tuple[int, int, str]] = []
    for line in output.decode("utf-8", errors="replace").splitlines():
        parts = line.split(None, 2)
        if len(parts) < 2:
            continue
        try:
            pid, uid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        found.append((pid, uid, parts[2] if len(parts) > 2 else ""))
    return found


def _scan_timeout() -> float:
    """Return how long ``ps`` or ``lsof`` may take in the scan under way (:data:`_SCAN_LIMIT`)."""
    limit = _SCAN_LIMIT.get()
    return _SCAN_TIMEOUT_SECONDS if limit is None else limit


def _list_processes() -> list[tuple[int, int, str]] | None:
    """List every process with ``ps``, never running anything else.

    Returns:
        ``(pid, uid, command line)`` per process; ``None`` when ``ps`` cannot
        run, fails or does not answer in time (:func:`_scan_timeout`).
    """
    try:
        done = subprocess.run(
            [_PS, "-axwwo", "pid=,uid=,args="],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=_scan_timeout(),
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return _parse_ps(done.stdout) if done.returncode == 0 else None


def _lsof_unescape(match: re.Match[bytes]) -> bytes:
    """Return the byte one of ``lsof``'s escapes (:data:`_LSOF_ESCAPE_RE`) stands for."""
    code, letter = match.groups()
    return bytes([int(code, 16)]) if code is not None else _LSOF_ESCAPES[letter]


def _parse_lsof(output: bytes) -> dict[int, list[str]]:
    """Parse ``lsof -F n0`` output: NUL-terminated fields, one set per line.

    ``lsof`` escapes some characters in the names it prints
    (:data:`_LSOF_ESCAPE_RE`); each name is unescaped, so a lease directory
    whose path holds a newline, a tab, a backslash or a character ``lsof``
    printed as ``\\xNN`` is the directory on disk.

    Args:
        output: What ``lsof`` printed.

    Returns:
        Each process ``lsof`` listed -> the names of its open files.
    """
    files: dict[int, list[str]] = {}
    current: list[str] | None = None
    for field in output.split(b"\0"):
        if field.startswith(b"\n"):
            field = field[1:]
        if not field:
            continue
        tag, value = field[:1], field[1:]
        if tag == b"p":
            try:
                current = files.setdefault(int(value), [])
            except ValueError:
                current = None
        elif tag == b"n" and current is not None:
            current.append(os.fsdecode(_LSOF_ESCAPE_RE.sub(_lsof_unescape, value)))
    return files


def _list_open_files(pids: list[int]) -> dict[int, list[str]] | None:
    """List what each process has open, with ``lsof``.

    ``lsof`` runs with ``-b``, so it never waits on a file system that does not
    answer (a hung network mount), and in a UTF-8 locale, so it prints a
    non-ASCII name as it is rather than as escapes.

    Args:
        pids: The processes.

    Returns:
        Each process ``lsof`` could list -> the names of its open files; a
        process missing from it could not be listed, or has exited. ``None``
        when ``lsof`` cannot run or does not answer in time
        (:func:`_scan_timeout`).
    """
    if not pids:
        return {}
    try:
        done = subprocess.run(
            [_LSOF, "-nP", "-w", "-b", "-F", "n0", "-p", ",".join(str(pid) for pid in pids)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=_scan_timeout(),
            check=False,
            env={**os.environ, "LC_ALL": _LSOF_LOCALE},
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return _parse_lsof(done.stdout)  # exit status 1 only says some pid was not found


def _alive(pid: int) -> bool:
    """Return True if a process may still exist (asked with signal 0, which sends nothing)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # it exists; it is just not ours to signal
    return True


def _lease_name(name: str) -> str | None:
    """Return the session id a lease file is named for, if ``name`` is a lease file.

    Args:
        name: An open file's path, as ``lsof`` names it.

    Returns:
        The id in ``…/<id>.lock`` when it is one omp could use, else ``None``.
    """
    if not name.startswith("/") or not name.endswith(".lock"):
        return None
    stem = name.rsplit("/", 1)[1][: -len(".lock")]
    return stem if _LEASE_ID_RE.fullmatch(stem) else None


def _private_dir(directory: Path) -> bool:
    """Return True for a directory only this user can use, as omp's lease directories are."""
    try:
        info = os.lstat(directory)
    except (OSError, ValueError):
        return False
    owner = os.getuid() if hasattr(os, "getuid") else info.st_uid
    return stat.S_ISDIR(info.st_mode) and info.st_uid == owner and not info.st_mode & 0o077


def _scan_processes() -> _ProcessScan:
    """Find the lease files running omp processes have open, without running omp.

    This user's processes whose command line may be omp's (:func:`_runs_omp`)
    are listed with ``ps``, and what they have open with ``lsof``. A lease
    file found open is held, and the private directory it is in is a lease
    directory in use. A process that is still running but whose open files
    ``lsof`` could not list leaves the scan blind, and so does a process list
    that cannot be read.

    Returns:
        The scan, stamped when it finished: a slow scan is as young as any
        when it is reused.
    """
    processes = _list_processes()
    if processes is None:
        return _ProcessScan(blind=True, taken=time.monotonic())
    me = os.getpid()
    uid = os.getuid() if hasattr(os, "getuid") else None
    pids = sorted(
        {
            pid
            for pid, owner, command in processes
            if pid != me and (uid is None or owner == uid) and _runs_omp(command)
        }
    )
    if not pids:
        return _ProcessScan(taken=time.monotonic())
    files = _list_open_files(pids)
    held: set[str] = set()
    dirs: list[Path] = []
    blind = False
    for pid in pids:
        names = files.get(pid) if files is not None else None
        if names is None:
            blind = blind or _alive(pid)
            continue
        for name in names:
            session_id = _lease_name(name)
            if session_id is None:
                continue
            held.add(session_id)
            directory = Path(name).parent
            if directory not in dirs and _private_dir(directory):
                dirs.append(directory)
    return _ProcessScan(frozenset(held), tuple(dirs), blind, time.monotonic())


def _merge(into: DeleteResult, other: DeleteResult) -> None:
    """Fold one delete result into another (mutated in place)."""
    into.removed.extend(other.removed)
    into.freed_bytes += other.freed_bytes
    into.skipped.update(other.skipped)


# --- .env files ------------------------------------------------------------------------


def _env_trim(text: str) -> str:
    """Trim what Node's dotenv parser trims: spaces, tabs and newlines, at both ends."""
    return text.strip(_DOTENV_BLANKS)


def _node_parse_env(text: str) -> dict[str, str]:
    """Parse ``.env`` text exactly as Node's ``util.parseEnv`` does.

    A line-for-line port of ``Dotenv::ParseContent`` in Node 26's
    ``src/node_dotenv.cc``, quirks included: every ``\\r`` is dropped first; a
    key is trimmed of spaces, tabs and newlines and loses one ``export ``
    prefix (with a space, not a tab); an empty value is stored before that
    prefix is removed; a value in double quotes turns ``\\n`` into a newline
    and may span lines, as single and back quotes may; an unclosed quote takes
    the rest of its line, quote included; any other value ends at a ``#`` and
    is trimmed. Nothing is expanded, and the last value of a key wins.

    omp runs on Bun, whose ``util.parseEnv`` is another parser
    (:func:`_bun_parse_env`); this one only says where the two may differ.

    The text is walked by position rather than sliced, and each quote's
    closing match is looked up rather than searched for, so a large file costs
    time in proportion to its size.

    Args:
        text: The file's contents.

    Returns:
        Every key as the parser stores it, valid names or not.
    """
    text = _env_trim(text.replace("\r", ""))
    end = len(text)
    quotes = {quote: [m.start() for m in re.finditer(quote, text)] for quote in "'\"`"}

    def skip_blanks(at: int) -> int:
        """Return the first position at or after ``at`` that is not blank."""
        found = _DOTENV_NOT_BLANK_RE.search(text, at)
        return found.start() if found else end

    def next_line(at: int) -> int:
        """Return where the line after the one holding ``at`` starts."""
        newline = text.find("\n", at)
        return newline + 1 if newline >= 0 else end

    def closing(quote: str, at: int) -> int:
        """Return the first ``quote`` at or after ``at``, or -1."""
        spots = quotes[quote]
        index = bisect.bisect_left(spots, at)
        return spots[index] if index < len(spots) else -1

    store: dict[str, str] = {}
    start = 0
    while start < end:
        if text[start] in "\n#":
            start = next_line(start)  # a blank line or a comment
            continue
        found = _DOTENV_KEY_END_RE.search(text, start)
        if found is None:
            break
        stop = found.start()
        if text[stop] == "\n":
            start = skip_blanks(stop + 1)  # no "=" on the line
            continue
        key = _env_trim(text[start:stop])
        start = stop + 1
        if start >= end or text[start] == "\n":
            store[key] = ""  # before "export " is taken off the key
            continue
        start = skip_blanks(start)
        if not key:
            continue  # the value is parsed as if it began a line
        if key.startswith("export "):
            key = _env_trim(key[len("export ") :])
        if start >= end:
            store[key] = ""
            break
        quote = text[start]
        if quote in quotes:
            close = closing(quote, start + 1)
            if close >= 0:
                value = text[start + 1 : close]
                store[key] = value.replace("\\n", "\n") if quote == '"' else value
                start = next_line(close + 1)  # the rest of the line is dropped
                continue
            newline = text.find("\n", start)
            if newline < 0:
                store[key] = text[start:]
                break
            store[key] = text[start:newline]  # unclosed: the line, quote and all
            start = skip_blanks(newline + 1)
            continue
        newline = text.find("\n", start)
        line_end = newline if newline >= 0 else end
        hashed = text.find("#", start, line_end)
        store[key] = _env_trim(text[start : hashed if hashed >= 0 else line_end])
        start = skip_blanks(newline + 1) if newline >= 0 else end
    return store


def _bun_unescape(match: re.Match[str]) -> str:
    """Return what Bun makes of one escape, or carriage return, in double quotes."""
    escaped = match.group(1)
    if escaped is None:
        return "\n"  # a carriage return, alone or before a newline
    if escaped == "n":
        return "\n"
    if escaped == "r":
        return "\r"
    return match.group(0)  # any other escape stays as written: \" is \"


def _bun_parse_env(text: str) -> dict[str, str]:
    """Parse ``.env`` text exactly as Bun's ``util.parseEnv`` does — the parser omp runs.

    A port of ``Parser`` in Bun's ``src/dotenv/env_loader.rs``, as
    ``parse_env`` in ``node_util_binding.rs`` drives it
    (``Loader::load_from_string::<true, false>``: the last value of a key wins,
    nothing is expanded), quirks included:

    * a byte-order mark at the start is dropped;
    * a line is skipped unless it starts — after blanks, which may span lines —
      with a key: letters, digits, ``_``, ``-`` and ``.``, then ``=``, or ``:``
      and one blank, with blanks, newlines included, allowed before either
      (``KEY\\n=value`` sets ``KEY``); ``export`` and at least one blank (a tab
      too) may come first;
    * a carriage return ends a line as a newline does;
    * a value starting — after blanks, which may span lines — with a quote ends
      at the first closing quote a backslash does not escape, and the rest of
      that line is dropped; in double quotes ``\\n`` and ``\\r`` become a
      newline and a carriage return and every other escape stays as written,
      and in every quote a carriage return, alone or before a newline, becomes
      a newline. A quote that is never closed is no quote;
    * any other value ends at ``#`` or at the end of its line, and is trimmed of
      tabs, vertical tabs, form feeds, spaces, newlines and carriage returns.

    Bun parses UTF-8 bytes, and every character it treats specially is ASCII,
    so walking the decoded text gives the same result. Each step is a search
    from a position, so a large file costs time in proportion to its size.

    Args:
        text: The file's contents.

    Returns:
        Every key as the parser stores it, valid variable names or not.
    """
    text = text.removeprefix("\ufeff")  # a byte-order mark
    end = len(text)

    def skip_blanks(at: int) -> int:
        """Return the first position at or after ``at`` that is not blank."""
        found = _BUN_NOT_BLANK_RE.search(text, at)
        return found.start() if found else end

    def skip_line(at: int) -> int:
        """Return where the line after the one holding ``at`` starts."""
        found = _BUN_LINE_END_RE.search(text, at)
        return found.end() if found else end

    def parse_key(at: int, check_export: bool) -> tuple[str | None, int]:
        """Return the key at ``at`` and where its value starts, or ``(None, start)``."""
        if check_export:
            at = skip_blanks(at)
        found = _BUN_KEY_RE.match(text, at)
        stop = found.end() if found else at
        if at < stop < end:
            after = skip_blanks(stop)
            if after < end:
                if check_export and stop < after and text[at:stop] == "export":
                    key, value_at = parse_key(after, False)
                    if key is not None:
                        return key, value_at
                if text[after] == "=":
                    return text[at:stop], after + 1
                if text[after] == ":" and after + 1 < end and text[after + 1] in _BUN_BLANKS:
                    return text[at:stop], after + 2
        return None, at

    def parse_quoted(at: int) -> tuple[str, int] | None:
        """Return the quoted value at ``at`` and where the next line starts, or ``None``."""
        quote = text[at]
        stops = _BUN_QUOTE_STOPS[quote]
        close = at + 1
        while True:
            found = stops.search(text, close)
            if found is None:
                return None
            close = found.start()
            if text[close] == quote:
                break
            close += 2  # a backslash escapes the next character, whatever it is
        inner = text[at + 1 : close]
        if quote == '"':
            value = _BUN_DOUBLE_RE.sub(_bun_unescape, inner)
        else:
            value = _BUN_CR_RE.sub("\n", inner)
        return value, skip_line(close + 1)

    store: dict[str, str] = {}
    position = 0
    while position < end:
        key, position = parse_key(position, True)
        if key is None:
            position = skip_line(position)
            continue
        first = skip_blanks(position)
        if first >= end:
            store[key] = ""
            break
        if text[first] in _BUN_QUOTES:
            quoted = parse_quoted(first)
            if quoted is not None:
                store[key], position = quoted
                continue
        found = _BUN_VALUE_END_RE.search(text, position)
        stop = found.start() if found else end
        store[key] = text[position:stop].strip(_BUN_BLANKS)
        position = stop
    return store


def _omp_values(parsed: Mapping[str, str]) -> dict[str, str]:
    """Return what omp's ``JI`` keeps of one parsed ``.env`` file.

    Args:
        parsed: Every key a parser found, with its value.

    Returns:
        The keys that are valid variable names and whose value holds no NUL.
    """
    return {
        key: value
        for key, value in parsed.items()
        if _DOTENV_KEY_RE.fullmatch(key) and "\x00" not in value
    }


@dataclass(frozen=True, slots=True)
class _DotEnv:
    """One ``.env`` file, as omp takes it.

    Attributes:
        values: Its variables, as Bun's parser — the one omp runs — reads them.
        unsure: The names in it that set a variable ``sx`` follows
            (:data:`_DOTENV_NAMES`) and that Node's parser reads otherwise —
            set or not, or set to another value. What omp gets rests on how
            its own Bun parses that line, so ``sx`` trusts neither reading.
    """

    values: Mapping[str, str]
    unsure: tuple[str, ...] = ()


#: A ``.env`` file that sets nothing, as when omp cannot read one.
_NO_DOTENV = _DotEnv({})


def _parse_dotenv(text: str) -> _DotEnv:
    """Parse ``.env`` text into the variables omp takes from it.

    A key starting ``OMP_`` also sets the ``PI_`` key it names, as omp does
    for these files (and only for these): ``OMP_CONFIG_DIR`` in a ``.env`` is
    ``PI_CONFIG_DIR``.

    Args:
        text: The file's contents.

    Returns:
        :func:`_bun_parse_env`'s reading, kept as omp keeps it
        (:func:`_omp_values`), and the names ``sx`` follows that
        :func:`_node_parse_env` reads otherwise.
    """
    values = _omp_values(_bun_parse_env(text))
    other = _omp_values(_node_parse_env(text))
    unsure = tuple(name for name in _DOTENV_NAMES if values.get(name) != other.get(name))
    for key in [key for key in values if key.startswith("OMP_")]:
        values[f"PI_{key[4:]}"] = values[key]
    return _DotEnv(values, unsure)


def _read_dotenv(path: Path) -> _DotEnv | None:
    """Read one ``.env`` file omp loads, as omp reads it, never raising.

    omp reads the whole file (``readFileSync(path, "utf-8")``) and gets
    nothing from one it cannot read. ``sx`` reads a regular file of up to 1
    MiB the same way, and finds nothing where omp's read fails too: no file,
    a directory, a path that cannot exist. Anything else — a larger file,
    whose tail ``sx`` would not see; a pipe or a device, which would block;
    a file ``sx`` may not open, which omp may — is unknown.

    Args:
        path: The file.

    Returns:
        Its variables (:func:`_parse_dotenv`); :data:`_NO_DOTENV` when omp
        would find none; ``None`` when ``sx`` cannot tell which variables it
        sets.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK)
    except (FileNotFoundError, NotADirectoryError, ValueError):
        return _NO_DOTENV
    except OSError as exc:
        return _NO_DOTENV if exc.errno in (errno.ELOOP, errno.ENAMETOOLONG) else None
    chunks: list[bytes] = []
    try:
        info = os.fstat(fd)
        if stat.S_ISDIR(info.st_mode):
            return _NO_DOTENV  # omp's read of a directory fails: it finds nothing
        if not stat.S_ISREG(info.st_mode) or info.st_size > _DOTENV_LIMIT:
            return None  # a pipe or a device would block; a tail would go unread
        size = 0
        while size <= _DOTENV_LIMIT:
            chunk = os.read(fd, _DOTENV_LIMIT + 1 - size)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
    except OSError:
        return None
    finally:
        os.close(fd)
    data = b"".join(chunks)
    if len(data) > _DOTENV_LIMIT:
        return None  # it grew while it was read
    return _parse_dotenv(data.decode("utf-8", errors="replace"))


# --- the store -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Store:
    """Where omp keeps everything ``sx`` reads, as resolved right now.

    Attributes:
        root: The config root (``~/.omp``).
        agent: The agent directory.
        sessions: The sessions directory.
        state: The state directory, which holds ``memories/``.
        leases: The lease directory an omp started from ``sx``'s environment
            uses, or ``None`` when it cannot be known.
        home: The home directory everything was resolved from.
        profile: The profile in effect, if any.
        profile_variable: The variable that selected it.
        overridden: True when ``PI_CODING_AGENT_DIR`` replaced the agent
            directory, which is then omp's only if it shows omp's marks.
        lease_dirs: Every lease directory an omp process may be using.
        lease_unknowable: True when one of those cannot be placed — a relative
            path, or a process that runs with no lease at all.
        lease_watch: Where an omp whose own environment differs may keep its
            leases — the test runtime's directory — asked only whether a lease
            is held there: a guess never vouches that one is free.
        notes: Informational lines for ``sx harnesses``.
        warnings: Problems the user should act on.
    """

    root: Path
    agent: Path
    sessions: Path
    state: Path
    leases: Path | None
    home: Path = Path("/dev/null/sx-omp")
    profile: str | None = None
    profile_variable: str | None = None
    overridden: bool = False
    lease_dirs: tuple[Path, ...] = ()
    lease_unknowable: bool = False
    lease_watch: tuple[Path, ...] = ()
    notes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


#: Returned when the home directory cannot be determined: under ``/dev/null``,
#: so nothing there can ever exist, be listed or be deleted.
_NOWHERE = _Store(
    root=Path("/dev/null/sx-omp"),
    agent=Path("/dev/null/sx-omp/agent"),
    sessions=Path("/dev/null/sx-omp/agent/sessions"),
    state=Path("/dev/null/sx-omp/agent"),
    leases=None,
    lease_unknowable=True,
    warnings=("could not determine the home directory; Oh My Pi sessions are not listed.",),
)


def _profile_name(raw: str | None) -> tuple[str | None, bool]:
    """Validate a profile value the way omp does (its ``fq``).

    Args:
        raw: The variable's value, or ``None`` when unset.

    Returns:
        ``(profile, valid)``: the profile, or ``None`` for the default one;
        ``valid`` is False when omp would refuse to start.
    """
    if raw is None:
        return None, True
    value = _js_trim(raw)
    if not value or value == "default":
        return None, True
    if value.endswith(".") or not _PROFILE_RE.fullmatch(value) or _RESERVED_RE.fullmatch(value):
        return None, False
    return value, True


def _selected_profile(env: Mapping[str, str]) -> tuple[str | None, str | None, bool]:
    """Return the profile an environment selects (omp's ``m4e``).

    Args:
        env: The variables omp sees.

    Returns:
        ``(profile, variable, valid)``: ``OMP_PROFILE`` when it is set at all,
        else ``PI_PROFILE``.
    """
    variable = next((name for name in PROFILE_ENVS if name in env), None)
    profile, valid = _profile_name(env.get(variable) if variable else None)
    return profile, variable, valid


def _agent_override(raw: str, base: Path) -> tuple[str | None, str | None]:
    """Vet a ``PI_CODING_AGENT_DIR`` value.

    omp resolves it against its own working directory, with no ``~``
    expansion. ``sx`` follows an absolute value only — a relative one would
    make a store root, the delete allowlist, follow ``sx``'s working directory
    — and refuses the home directory and a filesystem root, whose
    ``sessions`` could be an ordinary user directory.

    Args:
        raw: The value, as set.
        base: The home directory.

    Returns:
        ``(agent dir, None)`` when usable, else ``(None, reason)``.
    """
    if not os.path.isabs(raw):
        return None, "a relative path, which Oh My Pi resolves against wherever it is started"
    candidate = _node_normpath(raw)
    if candidate == "/":
        return None, "a filesystem root"
    if _same_place(Path(candidate), base):
        return None, "the home directory"
    return candidate, None


def _agent_dir(
    base: Path, root: str, profile: str | None, env: Mapping[str, str]
) -> tuple[str, str, str | None, str | None]:
    """Work out omp's agent directory and how ``PI_CODING_AGENT_DIR`` was taken.

    Args:
        base: The home directory.
        root: The config root in effect, ``~/<PI_CONFIG_DIR|.omp>``.
        profile: The profile in effect, if any.
        env: The real environment: omp reads ``PI_CODING_AGENT_DIR`` — and
            decides whether it names ``PI_PROFILE``'s agent directory, which
            counts as unset — before it loads any ``.env`` file.

    Returns:
        ``(agent, default, outcome, detail)``: the agent directory; the
        profile's default one; and ``"profile"``, ``"legacy"``, ``"ignored"``,
        ``"followed"`` or ``None``, with the reason or override for it.
    """
    default = _node_join(root, "profiles", profile, "agent") if profile else _node_join(root, "agent")
    raw = env.get(AGENT_DIR_ENV)
    if not raw:
        return default, default, None, None
    if profile:
        return default, default, "profile", None
    legacy, _ = _profile_name(env.get("PI_PROFILE"))
    early_root = _node_join(str(base), env.get(CONFIG_DIR_ENV) or ".omp")
    if legacy and raw == _node_join(early_root, "profiles", legacy, "agent"):
        return default, default, "legacy", legacy
    override, reason = _agent_override(raw, base)
    if override is None:
        return default, default, "ignored", reason
    return override, default, "followed", override


def _dotenv_files(base: Path, env: Mapping[str, str]) -> list[Path]:
    """Return the ``.env`` files omp loads at startup, in the order that decides a variable.

    omp finds them from its directories as its own environment sets them,
    before any ``.env`` is read: ``<agent dir>/.env``, ``<config root>/.env``
    (``~/.omp``, or the profile's ``~/.omp/profiles/<p>``), ``~/.env``. Its
    ``<cwd>/.env``, read before them all, is not included.

    Args:
        base: The home directory.
        env: The real environment.

    Returns:
        The three paths.
    """
    root = _node_join(str(base), env.get(CONFIG_DIR_ENV) or ".omp")
    profile, _, _ = _selected_profile(env)
    config_root = _node_join(root, "profiles", profile) if profile else root
    agent, _, _, _ = _agent_dir(base, root, profile, env)
    return [Path(agent, ".env"), Path(config_root, ".env"), base / ".env"]


def _effective_env(
    real: Mapping[str, str], loaded: list[tuple[Path, Mapping[str, str]]]
) -> tuple[dict[str, str], dict[str, Path]]:
    """Fill the variables ``sx`` follows from ``.env`` files, as omp does at startup.

    Args:
        real: ``sx``'s own environment.
        loaded: Each ``.env`` file with its variables, the first winning.

    Returns:
        ``(env, origins)``: the variables of :data:`_DOTENV_KEYS` as omp would
        see them, and for each one a ``.env`` supplied, that file. A file
        supplies a variable only where the environment leaves it unset or
        empty.
    """
    env = {key: real[key] for key in _DOTENV_KEYS if key in real}
    origins: dict[str, Path] = {}
    for path, values in loaded:
        for key in _DOTENV_KEYS:
            if key in values and not env.get(key):
                env[key] = values[key]
                if values[key]:
                    origins[key] = path
    return env, origins


def _xdg_dir(env: Mapping[str, str], variable: str, profile: str | None) -> tuple[str | None, bool]:
    """Return the ``$XDG_*/omp`` directory omp would move into, if it exists.

    Args:
        env: The variables omp sees.
        variable: ``XDG_DATA_HOME`` or ``XDG_STATE_HOME``.
        profile: The profile in effect, if any; its directory is
            ``<dir>/omp/profiles/<profile>``.

    Returns:
        ``(directory, relative)``: the directory when it exists, and whether
        the variable was set to a relative path, which ``sx`` cannot place.
    """
    value = env.get(variable)
    if not value:
        return None, False
    if not os.path.isabs(value):
        return None, True
    target = _node_join(value, "omp", "profiles", profile) if profile else _node_join(value, "omp")
    return (target if os.path.exists(target) else None), False


def _lease_dirs_from(
    base: Path, sources: Iterable[Mapping[str, str]]
) -> tuple[list[Path], list[str]]:
    """Return the lease directories a set of variables could send omp to.

    Every value counts on its own, since one omp process can see another's
    variables but not this one's: a ``PI_TEST_SESSION_OWNERS_DIR``; an
    ``$XDG_STATE_HOME/omp/run/session-owners`` whose ``…/omp`` exists; a
    ``~/<PI_CONFIG_DIR>/run/session-owners``; and the test runtime's lease
    directory (:func:`_test_lease_dir`) for ``PI_TEST_RUNTIME=1``.

    Args:
        base: The home directory.
        sources: Mappings of variables — an environment, a ``.env`` file.

    Returns:
        ``(dirs, relative)``: the directories, and the variables whose value
        is a relative path, which omp resolves against its own working
        directory and ``sx`` therefore cannot place.
    """
    dirs: list[Path] = []
    relative: list[str] = []
    for source in sources:
        owners = source.get(LEASE_DIR_ENV)
        if owners:
            if os.path.isabs(owners):
                dirs.append(Path(_node_normpath(owners)))
            else:
                relative.append(LEASE_DIR_ENV)
        state = source.get(XDG_STATE_ENV)
        if state:
            if not os.path.isabs(state):
                relative.append(XDG_STATE_ENV)
            elif os.path.exists(_node_join(state, "omp")):
                dirs.append(Path(_node_join(state, "omp", "run", "session-owners")))
        config = source.get(CONFIG_DIR_ENV)
        if config:
            dirs.append(Path(_node_join(str(base), config, "run", "session-owners")))
        if source.get(TEST_RUNTIME_ENV) == "1":
            dirs.append(_test_lease_dir())
    return dirs, relative


def _test_lease_dir() -> Path:
    """Return where omp keeps its leases in its test runtime, as its ``ys`` names it.

    That is ``<os.tmpdir()>/omp-test-session-owners-<uid>``, for an omp run
    with ``PI_TEST_RUNTIME=1`` and no lease directory of its own. The temp
    directory is ``sx``'s own: the one a user's processes share.

    Returns:
        The directory.
    """
    owner = os.getuid() if hasattr(os, "getuid") else os.environ.get("USERNAME", "")
    return Path(_node_join(_node_tmpdir(), f"omp-test-session-owners-{owner}"))


def _dedupe(paths: Iterable[Path]) -> tuple[Path, ...]:
    """Return ``paths`` without repeats, in their first order."""
    return tuple(dict.fromkeys(paths))


def _resolve(base: Path) -> _Store:
    """Resolve omp's store from the environment and its ``.env`` files, as omp 18.15 does.

    Args:
        base: The home directory.

    Returns:
        The store.
    """
    real = os.environ
    found = [(path, _read_dotenv(path)) for path in _dotenv_files(base, real)]
    unreadable = [path for path, dotenv in found if dotenv is None]
    unsure = [
        (path, dotenv.unsure) for path, dotenv in found if dotenv is not None and dotenv.unsure
    ]
    loaded = [(path, dotenv.values if dotenv is not None else {}) for path, dotenv in found]
    env, origins = _effective_env(real, loaded)
    notes = [
        f"{key} is set in {short_path(origins[key])}, which Oh My Pi loads at startup; "
        "sx reads it from there too"
        for key in _DOTENV_KEYS
        if key in origins
    ]
    warnings: list[str] = []
    config_name = env.get(CONFIG_DIR_ENV) or ".omp"
    root = _node_join(str(base), config_name)
    if config_name != ".omp":
        notes.append(f"{CONFIG_DIR_ENV} is set; Oh My Pi's config root is {short_path(Path(root))}")

    profile, variable, valid = _selected_profile(env)
    if not valid:
        warnings.append(
            f"{variable} is {env[variable]!r}, which is not a valid Oh My Pi profile name (omp "
            "refuses to start with it); reading the default profile's sessions."
        )
    agent, default_agent, outcome, detail = _agent_dir(base, root, profile, real)
    if outcome == "profile":
        warnings.append(
            f"{AGENT_DIR_ENV} is set, but Oh My Pi ignores it under profile {profile!r} "
            f"({variable}); reading {short_path(Path(agent))}."
        )
    elif outcome == "legacy":
        notes.append(
            f"{AGENT_DIR_ENV} names profile {detail!r}'s agent directory, which Oh My Pi "
            "treats as unset here; reading the default profile's sessions."
        )
    elif outcome == "ignored":
        warnings.append(
            f"{AGENT_DIR_ENV} is {detail}; ignoring it and reading {short_path(Path(agent))}."
        )
    elif outcome == "followed":
        notes.append(f"{AGENT_DIR_ENV} is set; Oh My Pi's agent directory is {detail}")

    data = state = None
    if agent == default_agent:
        data, data_relative = _xdg_dir(env, XDG_DATA_ENV, profile)
        state, _ = _xdg_dir(env, XDG_STATE_ENV, profile)
        if data_relative:
            warnings.append(
                f"{XDG_DATA_ENV} is a relative path, which Oh My Pi resolves against wherever "
                "it is started; sx ignores it."
            )
    sessions = _node_join(data or agent, "sessions")
    if data:
        notes.append(f"$XDG_DATA_HOME/omp exists; reading Oh My Pi sessions from {data}/sessions")
    if state:
        notes.append(f"$XDG_STATE_HOME/omp exists; Oh My Pi's memories are in {state}/memories")

    owners = env.get(LEASE_DIR_ENV)
    lease_base, _ = _xdg_dir(env, XDG_STATE_ENV, None)
    dirs, relative = _lease_dirs_from(base, [real, *(values for _, values in loaded)])
    state_value = env.get(XDG_STATE_ENV)
    test_runtime = env.get(TEST_RUNTIME_ENV) == "1"
    if owners:
        leases = _node_normpath(owners) if os.path.isabs(owners) else None
    elif test_runtime:
        leases = str(_test_lease_dir())  # omp's ys(): even when the variable is set but empty
    elif owners == "":
        leases = None  # set but empty: omp runs with no lease at all
    elif state_value and not os.path.isabs(state_value):
        leases = None
    else:
        leases = _node_join(lease_base or root, "run", "session-owners")
    default_leases = Path(_node_join(str(base), ".omp", "run", "session-owners"))
    conventional, _ = _lease_dirs_from(base, [{XDG_STATE_ENV: str(base / ".local" / "state")}])
    lease_dirs = _dedupe([*([Path(leases)] if leases else []), default_leases, *dirs, *conventional])
    # An omp whose own environment sets PI_TEST_RUNTIME keeps its leases here;
    # nothing sx can read says one does, so it is only watched for a held lease.
    watch = tuple(path for path in (_test_lease_dir(),) if path not in lease_dirs)
    for path in unreadable:
        warnings.append(
            f"{short_path(path)} cannot be read the way Oh My Pi reads it at startup (it is "
            "larger than 1 MiB, not a regular file, or sx may not open it), so sx cannot tell "
            "where it sends Oh My Pi's session leases: every Oh My Pi session counts as open, "
            "and none can be deleted or moved."
        )
    for path, keys in unsure:
        warnings.append(
            f"{short_path(path)} sets {', '.join(keys)} in a form that Bun's .env parser, which "
            "Oh My Pi runs on, and Node's read differently, so sx cannot be sure what Oh My Pi "
            "makes of it; sx follows Bun's reading, but every Oh My Pi session counts as open, "
            "and none can be deleted or moved."
        )
    if owners == "" and not test_runtime:
        warnings.append(
            f"{LEASE_DIR_ENV} is set but empty, so Oh My Pi runs with no session leases: "
            "every Oh My Pi session counts as open, and none can be deleted or moved."
        )
    for name in sorted(set(relative)):
        warnings.append(
            f"{name} is a relative path (in the environment, or in a .env file Oh My Pi "
            "loads), which Oh My Pi resolves against wherever it is started, so sx cannot find "
            "its session leases: every Oh My Pi session counts as open, and none can be "
            "deleted or moved."
        )
    if not _LEASES_ARE_FILES:
        warnings.append(
            "On Linux, Oh My Pi's session leases are abstract sockets sx cannot probe: "
            "every Oh My Pi session counts as open, and none can be deleted or moved."
        )
    elif _FLOCK_LAYOUT is None:
        warnings.append(
            "sx cannot probe Oh My Pi's session leases on this platform: every Oh My Pi "
            "session counts as open, and none can be deleted or moved."
        )
    custom = env.get(SESSION_DIR_ENV, "")
    if custom:
        warnings.append(
            f"{SESSION_DIR_ENV} is set ({custom}): Oh My Pi writes new sessions there, and sx "
            "does not read custom session directories yet, so they are not listed."
        )
    return _Store(
        root=Path(root),
        agent=Path(agent),
        sessions=Path(sessions),
        state=Path(state or agent),
        leases=Path(leases) if leases else None,
        home=base,
        profile=profile,
        profile_variable=variable,
        overridden=agent != default_agent,
        lease_dirs=lease_dirs,
        lease_unknowable=bool(relative) or leases is None or bool(unreadable) or bool(unsure),
        lease_watch=watch,
        notes=tuple(notes),
        warnings=tuple(warnings),
    )


# --- the adapter ---------------------------------------------------------------


class OmpAdapter(PiFormatAdapter):
    """Adapter for Oh My Pi sessions.

    Store roots are the sessions directory and the memories directory; omp's
    shared stores beside them — ``history.db``, ``stats.db``, ``agent.db``,
    ``session-pins.json``, ``blobs/``, ``terminal-sessions/``,
    ``custom-session-files/`` — are outside both and never touched, as omp's own
    delete leaves them. Lease files live outside both too, and are only
    asked about. Lock files sit beside the transcripts but are never removed or
    moved: a session goes without them, and a bucket holding one is emptied,
    or relocated, around it.

    Attributes:
        name: Machine identifier (``"omp"``).
        display: Human-readable harness name.
        open_session_caveat: omp's lease marks a session open only while omp
            writes it, so the confirmations say what deleting or moving an
            open one does.
        fragment_entry_types: :data:`OMP_ENTRY_TYPES`.
        fragment_line_limit: 1 MiB, the most omp reads of a line to find a
            header.
        folder_idle_seconds: ``0``. omp creates a project's bucket when it
            starts there, before it writes a session, as Pi does — but every
            omp write path creates the bucket again when it is gone
            (``ensureDirSync``, ``mkdirSync(…, {recursive: true})``), so
            removing an empty one cannot break a running omp the way it breaks
            Pi. The store-wide 90-second quiet window is enough.
    """

    name = "omp"
    display = "Oh My Pi"
    open_session_caveat = (
        "Oh My Pi marks a session open only while it writes to it, so close a session in "
        "Oh My Pi before deleting or moving it: otherwise its next write can bring a deleted "
        "session back whole, leave a moved session in two places, or leave a header-less "
        "file that Oh My Pi will not resume."
    )
    fragment_entry_types = OMP_ENTRY_TYPES
    fragment_line_limit = _LINE_LIMIT
    folder_idle_seconds = 0

    # --- store layout ----------------------------------------------------

    def _store(self) -> _Store:
        """Resolve the store now; ``home`` is looked up in this module's namespace.

        The live badge (:meth:`_reused_scan`) may reuse a store resolved for the
        same home directory in the last :data:`_SCAN_REUSE_SECONDS`, rather
        than read every ``.env`` file again for each session it shows.
        Everything else resolves afresh.

        Returns:
            The store, or a store that can hold nothing when the home
            directory cannot be determined.
        """
        try:
            base = home()
        except Exception:  # noqa: BLE001 - sessions_dir() must never raise
            return _NOWHERE
        reuse = self._reuse_window()
        cached = self.__dict__.get("_sx_store")
        if reuse and cached is not None:
            store, taken = cached
            if store.home == base and time.monotonic() - taken < reuse:
                return store
        try:
            store = _resolve(base)
        except Exception:  # noqa: BLE001 - sessions_dir() must never raise
            return _NOWHERE
        self.__dict__["_sx_store"] = (store, time.monotonic())
        return store

    def _reuse_window(self) -> float:
        """Return how old a scan, store or ``.env`` reading the check under way may reuse.

        Returns:
            :data:`_SCAN_REUSE_SECONDS` inside :meth:`_reused_scan` (the live
            badge); ``0`` — nothing is reused — anywhere else, and always
            inside :meth:`_fresh_scan`.
        """
        state = self._thread_state()
        if getattr(state, "pinned", False):
            return 0.0
        return getattr(state, "reuse", 0.0)

    def _dotenv(self, path: Path) -> _DotEnv | None:
        """Read a project's ``.env`` file (:func:`_read_dotenv`), reused as the store is.

        Args:
            path: ``<project>/.env``.

        Returns:
            What :func:`_read_dotenv` returns.
        """
        reuse = self._reuse_window()
        cache = self.__dict__.setdefault("_sx_dotenvs", {})
        cached = cache.get(path)
        if reuse and cached is not None and time.monotonic() - cached[1] < reuse:
            return cached[0]
        dotenv = _read_dotenv(path)
        cache[path] = (dotenv, time.monotonic())
        return dotenv

    def sessions_dir(self) -> Path:
        """Return the directory holding omp's buckets. Never raises."""
        return self._store().sessions

    def agent_dir(self) -> Path:
        """Return omp's agent directory, for the active profile."""
        return self._store().agent

    def memories_dir(self) -> Path:
        """Return the directory holding omp's per-project memory folders."""
        return self._store().state / "memories"

    def lease_dir(self) -> Path | None:
        """Return the lease directory an omp started from ``sx``'s environment uses.

        Returns:
            ``PI_TEST_SESSION_OWNERS_DIR`` when set, else
            ``$XDG_STATE_HOME/omp/run/session-owners`` when ``…/omp`` exists,
            else ``~/<PI_CONFIG_DIR|.omp>/run/session-owners`` — each read as
            omp reads it, ``.env`` files included; ``None`` when it cannot be
            known. :meth:`lease_state` asks :meth:`lease_dirs` too.
        """
        return self._store().leases

    def lease_dirs(self, project: str | None = None) -> tuple[list[Path], bool]:
        """Return every lease directory an omp process may be using.

        The variables that place omp's leases may come from its environment or
        from any ``.env`` file it loads — ``<cwd>/.env`` differing from one
        process to the next — so one directory is not enough.

        Args:
            project: The session's recorded working directory, whose ``.env``
                an omp started there loads first.

        Returns:
            ``(dirs, unknowable)``: :meth:`lease_dir`, the default
            ``~/.omp/run/session-owners``, ``~/.local/state/omp/run/
            session-owners`` when ``~/.local/state/omp`` exists, and every
            directory a value in the environment, a ``.env`` file or the
            project's ``.env`` names; and True when one of them cannot be
            placed, or a ``.env`` file cannot be read the way omp reads it, or
            sets one of the variables ``sx`` follows in a form Bun's parser and
            Node's read differently. :meth:`lease_state` also asks the
            directories running omp processes hold leases in, and watches the
            test runtime's.
        """
        return self._lease_dirs_of(self._store(), project)

    def _lease_dirs_of(self, store: _Store, project: str | None) -> tuple[list[Path], bool]:
        """Return :meth:`lease_dirs` for a store already resolved.

        Args:
            store: The store.
            project: The session's recorded working directory, if known.

        Returns:
            ``(dirs, unknowable)``, as :meth:`lease_dirs`; a project ``.env``
            ``sx`` cannot read, or whose reading it cannot be sure of, makes
            them unknowable.
        """
        dirs, unknowable = list(store.lease_dirs), store.lease_unknowable
        if project and os.path.isabs(project):
            dotenv = self._dotenv(Path(project) / ".env")
            values = dotenv.values if dotenv is not None else {}
            extra, guess = _lease_dirs_from(store.home, [values])
            dirs = list(_dedupe([*dirs, *extra]))
            unknowable = unknowable or bool(guess) or dotenv is None or bool(dotenv.unsure)
        return dirs, unknowable

    def archive_dir(self) -> Path:
        """Return where ``omp gc`` archives sessions — outside every store root.

        sx never lists, moves or deletes anything there.
        """
        return self.sessions_dir().parent.joinpath(*_ARCHIVE_PARTS)

    def _archived(self, bucket: Path, stem: str, archive: Path | None = None) -> Path | None:
        """Return the archived transcript of session ``stem``, if one exists.

        omp 18.15 archives to :meth:`archive_dir`, under the bucket's own name;
        a ``S.jsonl.gz`` inside the bucket itself is checked too, so a session
        compressed in place by anything else is never taken for debris.

        Args:
            bucket: The bucket the session's files are in.
            stem: The transcript's name without ``.jsonl``.
            archive: :meth:`archive_dir`, when the caller already has it.

        Returns:
            The archive's path, or ``None``.
        """
        name = f"{stem}{_ARCHIVE_SUFFIX}"
        root = archive if archive is not None else self.archive_dir()
        for candidate in (bucket / name, root / bucket.name / name):
            if os.path.lexists(candidate):
                return candidate
        return None

    def store_roots(self) -> list[Path]:
        """Return the sessions and memories directories — all ``sx`` may change."""
        return [self.sessions_dir(), self.memories_dir()]

    def protected_paths(self) -> list[Path]:
        """Return the other memory backends' directories, never deletable."""
        memories = self.memories_dir()
        return [memories / backend for backend in _MEMORY_BACKENDS]

    def _yields_to(self) -> str | None:
        """Return ``Pi`` when ``PI_CODING_AGENT_DIR`` points omp at a store that is Pi's.

        Both harnesses read that variable, so with it set and no profile they
        resolve to the same agent directory. The directory is omp's only when
        it shows omp's marks (:func:`~sx.adapters.pi.omp_markers`) — exactly
        when Pi's adapter leaves it alone. The default agent directory, or a
        profile's, is omp's by definition.

        Returns:
            ``"Pi"``, or ``None``.
        """
        store = self._store()
        if not store.overridden:
            return None
        return None if omp_markers(store.agent, store.sessions) else "Pi"

    def store_notes(self) -> list[str]:
        """Return the overrides that moved omp's store, and the profile in use."""
        store = self._store()
        notes = list(store.notes)
        if store.profile:
            notes.insert(
                0,
                f"{store.profile_variable} selects Oh My Pi profile {store.profile!r}; reading "
                f"sessions from {short_path(store.sessions)}",
            )
        archive = store.sessions.parent.joinpath(*_ARCHIVE_PARTS)
        try:
            archived = next(archive.glob(f"*/*{_ARCHIVE_SUFFIX}"), None) is not None
        except OSError:
            archived = False
        if archived:
            notes.append(
                f"Oh My Pi's gc has archived sessions in {short_path(archive)}; sx does not "
                "list, move or delete them"
            )
        return notes

    def store_warnings(self) -> list[str]:
        """Return problems with where omp's sessions are, or may be, written.

        Returns:
            Ignored overrides, an invalid profile, a custom session directory,
            relative XDG directories, a ``.env`` file ``sx`` cannot read or be
            sure of, other profiles' sessions, which are not listed; then
            either a store left to Pi, or a lease directory that
            is nowhere to be found next to sessions and the files ``sx`` does
            not recognize.
        """
        store = self._store()
        warnings = list(store.warnings)
        others: list[tuple[str, Path]] = []
        try:
            candidates = sorted((store.root / "profiles").iterdir())
        except OSError:
            candidates = []
        for entry in candidates:
            if entry.name != store.profile:
                others.append((f"profile {entry.name!r}", entry / "agent" / "sessions"))
        if store.profile:
            others.append(("default profile", store.root / "agent" / "sessions"))
        for label, directory in others:
            if directory.is_dir() and not _same_place(directory, store.sessions):
                warnings.append(
                    f"Oh My Pi's {label} keeps sessions in {short_path(directory)}, which sx "
                    "does not list; select that profile with OMP_PROFILE to see them."
                )
        if self._yields_to() is not None:
            warnings.append(
                f"{AGENT_DIR_ENV} points Oh My Pi at {short_path(store.agent)}, but nothing "
                "there shows Oh My Pi has used it (no agent.db, history.db, models.db, "
                "terminal-sessions, blobs or omp-*.log, and no transcript or backup starting "
                "with a title slot, nor an artifacts directory, in its sessions), so sx takes it "
                "for Pi's store: Oh My Pi lists, moves and deletes nothing there."
            )
            return warnings
        lost = self._lost_leases()
        if lost is not None:
            warnings.append(lost)
        unrecognized = self.unrecognized_warning()
        if unrecognized is not None:
            warnings.append(unrecognized)
        return warnings

    def _lost_leases(self) -> str | None:
        """Return a warning when no lease directory exists although sessions do.

        omp creates its lease directory before it first writes a session, so
        sessions without one mean omp keeps its leases somewhere ``sx`` does
        not look.

        Returns:
            The warning, or ``None``.
        """
        if not _LEASES_ARE_FILES or _FLOCK_LAYOUT is None:
            return None
        dirs, unknowable = self.lease_dirs()
        if unknowable or any(os.path.lexists(directory) for directory in dirs):
            return None
        if not self._holds_sessions():
            return None
        places = ", ".join(short_path(directory) for directory in dirs)
        return (
            f"{short_path(self.sessions_dir())} holds sessions, but none of the places Oh My Pi "
            f"keeps its session leases exists ({places}). Oh My Pi creates its lease directory "
            "before it first writes a session, so it keeps them somewhere sx does not look — "
            f"another HOME, XDG_STATE_HOME or {LEASE_DIR_ENV} in its environment: every Oh My "
            "Pi session counts as open, and none can be deleted or moved."
        )

    def _holds_sessions(self) -> bool:
        """Return True if any bucket holds a ``.jsonl`` file, stopping at the first."""
        try:
            with os.scandir(self.sessions_dir()) as buckets:
                folders = [entry.path for entry in buckets if entry.is_dir(follow_symlinks=False)]
        except OSError:
            return False
        for folder in folders:
            try:
                with os.scandir(folder) as entries:
                    if any(is_transcript(Path(entry.path)) for entry in entries):
                        return True
            except OSError:
                continue
        return False

    # --- folder encoding --------------------------------------------------

    def folder_for(self, project: str) -> str:
        """Return the bucket omp uses today for ``project``."""
        return _buckets(project, home()).encoded

    def folder_names(self, project: str) -> set[str]:
        """Return today's bucket for ``project`` and every older one omp migrates.

        Args:
            project: A project directory.

        Returns:
            Today's name; the ``--<abs>--`` names of its real and lexical paths;
            the hashed name; and the two home-relative names a project shadowed
            by a temp directory inside home had.
        """
        names = _buckets(project, home())
        found = {
            names.encoded,
            names.hashed,
            encode_session_folder(names.real),
            encode_session_folder(names.resolved),
        }
        for extra in (names.shadowed_home, names.shadowed_home_hashed):
            if extra:
                found.add(extra)
        return found

    # --- format hooks ----------------------------------------------------

    def read_header(self, path: Path) -> dict | None:
        """Return a transcript's session header, after its title slot if any.

        Args:
            path: A ``.jsonl`` file.

        Returns:
            The header — the first record, or the second when the first is a
            title slot — or ``None`` when the file holds no session.
        """
        return _slot_and_header(path)[1]

    def is_fragment(self, path: Path) -> bool:
        """Return True if a file without a session header is debris omp left.

        omp's writer opens a transcript with ``O_CREAT|O_APPEND``, so one that
        reopens a path deleted or moved under it leaves an empty file, then
        entries with no title slot and no header — debris omp itself refuses
        to resume. Every full write omp makes (title slot, header and entries)
        goes through a temp file renamed into place, so a file that starts
        with a title slot but holds no header is not omp's debris, and neither
        is a header line omp could only read by scanning its text: those are
        not fragments, and :meth:`transcript_kind` calls them unrecognized.

        So a fragment is an empty file, or one whose first line — itself, not
        a later one — parses as an entry omp writes: a string ``id``, a type
        in :data:`OMP_ENTRY_TYPES` and a ``parentId`` that is a string or
        ``null``. Either way it must carry the name omp gives a session file
        (``<ts>_<id>.jsonl``): omp writes to no other file, and only that name
        says whose lease protects it, so any other file is one ``sx`` does not
        recognize — listed, never offered for cleanup.

        Args:
            path: A file :meth:`read_header` found no header in.

        Returns:
            Whether it is such a fragment.

        Raises:
            OSError: The file cannot be read.
        """
        if _name_id(path.name) is None:
            return False
        return super().is_fragment(path)

    def is_entry(self, obj: dict) -> bool:
        """Return True for a tree entry: neither the header nor the title slot."""
        return obj.get("type") not in ("session", "title")

    def title_for_file(self, path: Path, header: dict) -> str:
        """Return a session's title, as omp's session list shows it.

        omp reads the first 4096 bytes. The slot's title wins when the file has
        a slot — a blank one means no title, even if the header still carries
        an old one — and without a slot the header's title is used. With no
        title at all, the last compaction's ``shortSummary`` in those bytes
        stands in. Then comes the first user message in them (found by
        scanning the raw text when none parses whole, as omp does), and for an
        empty session ``Untitled · HH:MM`` from its start time. Whichever is
        used shows its first line only, without control characters; a first
        line that is blank counts as no title. A session that continues
        another is marked as such.

        Args:
            path: The transcript.
            header: Its session header.

        Returns:
            The title, cut at 80 characters.
        """
        text, records = _head_records(path)
        slot = records[0] if records and records[0].get("type") == "title" else None
        if slot is not None and isinstance(slot.get("title"), str):
            recorded = slot["title"] if _js_trim(slot["title"]) else None
        else:
            value = header.get("title")
            recorded = value if isinstance(value, str) else None
        if recorded is None:
            summaries = [
                record["shortSummary"]
                for record in records[1:]
                if record.get("type") == "compaction" and isinstance(record.get("shortSummary"), str)
            ]
            recorded = summaries[-1] if summaries else None
        title = _first_line(recorded) if recorded else None
        if title is None:
            first = self._first_message(text, records)
            title = _first_line(first) if first and first != _NO_MESSAGES else None
        if title is None:
            title = self._untitled(path, header)
        else:
            title = _truncate(title, _TITLE_LIMIT)
        continued = header.get("continuationOf")
        if isinstance(continued, str) and continued:
            title = f"{title} · continued"
        return title

    @staticmethod
    def _first_message(text: str, records: list[dict]) -> str | None:
        """Return the first message omp's session list finds in a file's first bytes.

        Args:
            text: The first 4096 bytes of the transcript.
            records: The records that parse whole in them.

        Returns:
            The first user message with any text, as omp joins it; else what
            omp's text scan finds; else ``None``.
        """
        for record in records[1:]:
            message = record.get("message")
            if record.get("type") != "message" or not isinstance(message, dict):
                continue
            if message.get("role") == "user":
                found = _text_blocks(message.get("content"))
                if found:
                    return found
        return _scanned_message(text)

    @staticmethod
    def _untitled(path: Path, header: dict) -> str:
        """Return omp's placeholder for a session without a title or prompt.

        Args:
            path: The transcript, whose mtime stands in for a missing start time.
            header: Its session header.

        Returns:
            ``Untitled · HH:MM`` in local time and the clock of the user's
            locale (see :func:`_clock`), or ``Untitled``.
        """
        moment = parse_ts(header.get("timestamp"))
        if moment is not None and moment.tzinfo is not None:
            moment = moment.astimezone()
        if moment is None:
            moment = file_mtime(path)
        return f"Untitled · {_clock(moment)}" if moment is not None else "Untitled"

    def current_leaf(self, entries: list[dict]) -> str | None:
        """Return the entry omp resumes from.

        The last entry's ``activeLeafId`` wins when it names an entry; ``null``
        means the session resumes empty, which is returned as a stand-in id so
        the current branch shows nothing.

        Args:
            entries: The tree entries, in file order.

        Returns:
            The leaf id, or ``None`` when there are no entries.
        """
        leaf = _resume_leaf(entries)
        if leaf is None and entries:
            return _ROOT_LEAF
        return leaf

    def _tree(self, entries: list[dict]) -> Transcript:
        """Arrange the entries, with an empty current branch for an empty leaf.

        omp's empty leaf (``activeLeafId: null``) is its root: the session
        resumes before every entry, and an entry with no parent hangs off that
        root. So a stand-in root entry, which shows nothing, becomes the whole
        current branch, and every parentless entry is re-parented onto it in
        this view: each branch then forks from the start of the (empty)
        current branch rather than standing as a separate root, and the
        file-order view marks a jump back to the root as continuing from the
        start.

        Args:
            entries: Tree entries in file order, each with a string ``id``.

        Returns:
            The :class:`~sx.model.Transcript`.
        """
        if (
            entries
            and self.current_leaf(entries) == _ROOT_LEAF
            and not any(entry.get("id") == _ROOT_LEAF for entry in entries)
        ):
            root = {"type": _ROOT_ENTRY_TYPE, "id": _ROOT_LEAF, "parentId": None}
            entries = [
                root,
                *(
                    {**entry, "parentId": _ROOT_LEAF} if _parent_of(entry) is None else entry
                    for entry in entries
                ),
            ]
        return super()._tree(entries)

    def entry_message(self, entry: dict) -> Message | None:
        """Render one entry, including omp's own roles and entry types.

        Developer turns are skipped; a Python run shows like a bash run; an
        ``@file`` mention lists its files; ``model_change`` holds one
        ``provider/model`` string. omp's bookkeeping entries are skipped, and
        images — inline or ``blob:sha256:`` references — show as ``[image]``
        without the blob store being read.

        Args:
            entry: One tree entry.

        Returns:
            The message, or ``None``.
        """
        kind = entry.get("type")
        if kind in _SKIPPED_ENTRIES or kind == _ROOT_ENTRY_TYPE:
            return None
        timestamp = parse_ts(entry.get("timestamp"))
        message = entry.get("message")
        if kind == "message" and isinstance(message, dict):
            role = message.get("role")
            if role == "developer":
                return None
            if role == "pythonExecution":
                return _python_message(message, timestamp)
            if role == "fileMention":
                return _file_mention(message, timestamp)
        if kind == "model_change":
            model = entry.get("model")
            if isinstance(model, str) and model.strip():
                text = f"model → {model.strip()}"
                role = entry.get("role")
                if isinstance(role, str) and role and role != "default":
                    text = f"{text} ({role})"
                return Message(Role.SYSTEM, text=text, timestamp=timestamp)
        return super().entry_message(entry)

    def header_update(
        self, header: dict, old: Path, new: Path, *, moved_from: Path | None
    ) -> dict | None:
        """Return the header a move writes, as omp's ``moveTo`` leaves it.

        ``cwd`` is re-pointed. As in ``moveTo``, ``additionalDirectories``
        loses any entry equal to the new ``cwd`` and goes once it is empty,
        and its other entries are left as they are — even ones under the old
        directory, which ``moveTo`` does not re-point either. A relocated
        transcript's old absolute path is appended to
        ``previousSessionFiles``, without duplicates.

        Where ``sx`` differs from ``moveTo`` is outside the header: it moves
        the session's backups with it (``moveTo`` leaves them, and omp's next
        listing of the old bucket would restore the session there), refuses a
        file whose format it does not know, and refuses a session whose lease
        is held or cannot be checked.

        Args:
            header: The transcript's current session header.
            old: The project directory being moved away from.
            new: The project directory being moved to.
            moved_from: The transcript's path before the move, when the move
                relocates it.

        Returns:
            The replacement header, or ``None`` when nothing changes; ``header``
            itself is never edited.
        """
        updated = dict(header)
        cwd = repoint(header.get("cwd", ""), old, new)
        if cwd is not None:
            updated["cwd"] = cwd
            directories = header.get("additionalDirectories")
            if isinstance(directories, list):
                kept = [item for item in directories if item != cwd]
                if not kept:
                    del updated["additionalDirectories"]
                elif kept != directories:
                    updated["additionalDirectories"] = kept
        if moved_from is not None:
            previous = header.get("previousSessionFiles")
            files: list = []
            for item in previous if isinstance(previous, list) else []:
                if item not in files:
                    files.append(item)
            location = os.path.abspath(moved_from)
            if location not in files:
                files.append(location)
            if files != previous:
                updated["previousSessionFiles"] = files
        return updated if updated != header else None

    # --- liveness ----------------------------------------------------------

    def lease_state(self, session_id: str, project: str | None = None) -> str:
        """Return whether a running omp holds a session's ownership lease.

        Every directory :meth:`lease_dirs` names is asked, without taking the
        lease, and the safest answer wins: held if any holds it, unknown if
        any cannot be checked — or if no lease directory exists at all, since
        omp creates one before it first writes a session.

        Args:
            session_id: The session id.
            project: The session's recorded working directory, if known.

        Returns:
            :data:`LEASE_FREE`, :data:`LEASE_HELD` or :data:`LEASE_UNKNOWN`. An
            id omp would hash rather than use as a file name is unknown, and so
            is every id where leases cannot be probed.
        """
        return self._lease_states([session_id], project)[0]

    def _lease_states(self, ids: list, project: str | None = None) -> list[str]:
        """Return :meth:`lease_state` for several ids, finding the lease directories once.

        Two sources are asked. The kernel, for every lease directory
        :meth:`lease_dirs` names and every one a running omp holds a lease in
        (:func:`_scan_processes`) — and, for a held lease only, the test
        runtime's directory, which nothing says omp uses; and the running omp
        processes themselves: a lease file one of them has open is held,
        wherever it is — a place only omp's own environment names included. A
        lease that would read free reads unknown while an omp process is
        running whose open files cannot be listed.

        Args:
            ids: Session ids.
            project: Their session's recorded working directory, if known.

        Returns:
            One state per id, in order.
        """
        if not ids:
            return []
        if not _LEASES_ARE_FILES or fcntl is None or _FLOCK_LAYOUT is None:
            return [LEASE_UNKNOWN] * len(ids)
        store = self._store()
        dirs, unknowable = self._lease_dirs_of(store, project)
        scan = self._process_scan()
        dirs = list(_dedupe([*dirs, *scan.dirs]))
        watch = [directory for directory in store.lease_watch if directory not in dirs]
        states: list[str] = []
        for session_id in ids:
            if not isinstance(session_id, str) or not _LEASE_ID_RE.fullmatch(session_id):
                states.append(LEASE_UNKNOWN)
                continue
            if session_id in scan.held:
                states.append(LEASE_HELD)
                continue
            found = {_probe_lease(directory, session_id) for directory in dirs}
            if any(_probe_lease(directory, session_id) == LEASE_HELD for directory in watch):
                found.add(LEASE_HELD)
            if LEASE_HELD in found:
                states.append(LEASE_HELD)
            elif unknowable or scan.blind or LEASE_UNKNOWN in found or LEASE_FREE not in found:
                states.append(LEASE_UNKNOWN)
            else:
                states.append(LEASE_FREE)
        return states

    def _process_scan(self) -> _ProcessScan:
        """Return the scan of the running omp processes a lease check uses now.

        Inside :meth:`_fresh_scan`, the one scan that operation took — taken at
        its first lease check, never reused from before. Inside
        :meth:`_reused_scan`, the adapter's last scan while it is young
        enough — a blind one too — else a new scan whose listings may take
        :data:`_BADGE_TIMEOUT_SECONDS` at most. Anywhere else, a new scan.

        Returns:
            The scan.
        """
        state = self._thread_state()
        if getattr(state, "pinned", False):
            if state.scan is None:
                state.scan = self._new_scan()
            return state.scan
        reuse = self._reuse_window()
        if not reuse:
            return self._new_scan()
        last = self.__dict__.get("_sx_scan")
        if last is not None and time.monotonic() - last.taken < reuse:
            return last
        return self._new_scan(_BADGE_TIMEOUT_SECONDS)

    def _new_scan(self, timeout: float | None = None) -> _ProcessScan:
        """Scan the running omp processes, and keep the scan for the live badge.

        Args:
            timeout: How long ``ps`` and ``lsof`` may each take; ``None`` for
                :data:`_SCAN_TIMEOUT_SECONDS`.

        Returns:
            The scan.
        """
        token = _SCAN_LIMIT.set(timeout)
        try:
            scan = _scan_processes()
        finally:
            _SCAN_LIMIT.reset(token)
        self.__dict__["_sx_scan"] = scan
        return scan

    @contextmanager
    def _fresh_scan(self) -> Iterator[None]:
        """Make every lease check of one destructive operation use one new scan.

        A delete, an orphan or leftover delete, a move's planning and each of
        its units run inside one, so what they decide on is never older than
        the operation itself, and the processes are listed once for it. An
        operation inside another shares the outer one's scan.

        Yields:
            Nothing; the scan is taken at the first lease check that needs it.
        """
        state = self._thread_state()
        if getattr(state, "pinned", False):
            yield
            return
        state.pinned, state.scan = True, None
        try:
            yield
        finally:
            state.pinned, state.scan = False, None

    @contextmanager
    def _reused_scan(self) -> Iterator[None]:
        """Let lease checks reuse a scan up to :data:`_SCAN_REUSE_SECONDS` old.

        For the live badge, asked about every session the TUI shows: the
        processes are listed, the store resolved and each ``.env`` file read at
        most once in that time, and a listing that does not answer within
        :data:`_BADGE_TIMEOUT_SECONDS` leaves the scan blind. Inside
        :meth:`_fresh_scan` it changes nothing.

        Yields:
            Nothing.
        """
        state = self._thread_state()
        before = getattr(state, "reuse", 0.0)
        state.reuse = _SCAN_REUSE_SECONDS
        try:
            yield
        finally:
            state.reuse = before

    def _lease_ids(self, path: Path, session_id: object) -> list[str]:
        """Return the ids a transcript's lease may be held under.

        Args:
            path: The transcript.
            session_id: The id its header records, if any.

        Returns:
            That id and, when different, the id in the file name.
        """
        ids = [session_id] if isinstance(session_id, str) and session_id else []
        named = _name_id(path.name)
        if named and named not in ids:
            ids.append(named)
        return ids

    def _lease_reason(self, ids: list[str], project: str | None = None) -> str | None:
        """Return why a session counts as open, judged by its leases.

        Args:
            ids: The ids its lease may be held under; none at all is unknown.
            project: Its recorded working directory, if known.

        Returns:
            A refusal reason, or ``None`` when every lease is free.
        """
        states = self._lease_states(ids, project)
        if LEASE_HELD in states:
            return _OPEN
        if not states or any(state != LEASE_FREE for state in states):
            return _UNSURE
        return None

    def _subagent_reason(self, artifacts: Path, project: str | None = None) -> str | None:
        """Return why a session counts as open, judged by its sub-agents.

        Sub-agents write their own transcripts inside the session's artifacts
        directory and hold their own leases; omp's ``gc`` checks them too.

        Args:
            artifacts: The session's ``S/`` directory.
            project: The session's recorded working directory, if known.

        Returns:
            A refusal reason, or ``None``.
        """
        if artifacts.is_symlink() or not artifacts.is_dir():
            return None
        try:
            nested = sorted(artifacts.rglob("*.jsonl"))
        except OSError:
            return "cannot read its artifacts directory to check its sub-agents (refused)"
        for path in nested:
            if path.is_symlink() or not path.is_file():
                continue
            header = self.read_header(path)
            ids = self._lease_ids(path, header.get("id") if header else None)
            states = self._lease_states(ids, project)
            if LEASE_HELD in states:
                return "a sub-agent of this session is running in Oh My Pi (refused)"
            if any(state != LEASE_FREE for state in states):
                return "cannot tell whether a sub-agent of this session is running (refused)"
            if _lock_reason(path) is not None:
                return "a sub-agent of this session is being written by Oh My Pi (refused)"
        return None

    def _live_reason(self, path: Path, ids: list[str], project: str | None) -> str | None:
        """Return why omp may have ``path`` open, or ``None`` if it does not.

        Args:
            path: A transcript.
            ids: The ids its lease may be held under (:meth:`_lease_ids`).
            project: Its recorded working directory, if known.

        Returns:
            A refusal reason: a held or unknown lease (its own or a sub-agent's),
            or a lock file showing omp is writing it.
        """
        reason = self._lease_reason(ids, project)
        if reason is not None:
            return reason
        reason = _lock_reason(path)
        if reason is not None or not path.name.endswith(".jsonl"):
            return reason
        return self._subagent_reason(path.parent / path.name[: -len(".jsonl")], project)

    def _transcript_live_reason(self, path: Path, header: dict | None) -> str | None:
        """Return :meth:`_live_reason` for a transcript as its header describes it."""
        ids = self._lease_ids(path, header.get("id") if header else None)
        return self._live_reason(path, ids, _header_cwd(header) if header else None)

    def last_activity(self, session: Session) -> datetime | None:
        """Return now for a session omp holds, else its transcript's mtime.

        Args:
            session: The session to probe.

        Returns:
            The current time when a lease is held or cannot be checked,
            otherwise the transcript's modification time.
        """
        path = session.primary_path
        if path is None:
            return None
        with self._reused_scan():
            states = self._lease_states(
                self._lease_ids(path, session.session_id), session.project_path
            )
        if not states or any(state != LEASE_FREE for state in states):
            return datetime.now()
        return file_mtime(path)

    def busy_transcript(self, path: Path) -> bool:
        """Return True if ``path`` must stay where it is on a move.

        Covers what :meth:`destructive_refusal` refuses for a move, so a refused
        session never travels with its folder: a file ``sx`` does not
        recognize, omp holding or writing it, and a file whose format a
        rewrite cannot be trusted with. A fragment, or a file already gone,
        stays only while the lease its name gives is held or cannot be
        checked, or a lock file shows omp writing it.

        Args:
            path: A transcript.

        Returns:
            Whether it is in use or unsafe to move.
        """
        kind, header, _ = self._classify(path)
        if kind is TranscriptKind.UNRECOGNIZED:
            return True
        if header is not None and (
            self.format_refusal(path, header) is not None
            or self._shape_problem(path, header, rewriting=True) is not None
        ):
            return True
        return self._transcript_live_reason(path, header) is not None

    def format_refusal(self, path: Path, header: dict) -> str | None:
        """Return why a session file is in a format ``sx`` does not know, if it is.

        Args:
            path: A transcript with a session header.
            header: That header.

        Returns:
            A refusal reason when the file does not begin the way omp writes one
            (a valid 256-byte title slot then the header, or a legacy file that
            starts with the header), or records a format version ``sx`` does
            not know; else ``None``.
        """
        layout = _layout(path)
        if layout == _UNKNOWN:
            return _LAYOUT_REFUSAL
        if layout == _SLOT_SIZE:
            return _SLOT_REFUSAL
        return super().format_refusal(path, header)

    # --- session files ------------------------------------------------------

    @staticmethod
    def _artifacts(path: Path) -> Path | None:
        """Return a transcript's artifacts directory, when it is a real one."""
        if not path.name.endswith(".jsonl"):
            return None
        artifacts = path.parent / path.name[: -len(".jsonl")]
        return artifacts if artifacts.is_dir() and not artifacts.is_symlink() else None

    @staticmethod
    def _session_backups(path: Path, *, links: bool = False) -> list[Path]:
        """Return every backup omp's own delete removes with ``path``.

        That is each ``S.jsonl.<anything>.bak`` file — the ones omp restores
        (``S.jsonl.<id>.bak``, its id holding no dot) and the rest alike.

        Args:
            path: A transcript.
            links: Return symbolic links named as backups instead of files.

        Returns:
            The backups, by name built from the transcript's own.
        """
        try:
            entries = sorted(path.parent.iterdir())
        except OSError:
            return []
        found = []
        for entry in entries:
            if not _owns_backup(path.name, entry.name):
                continue
            if links:
                if entry.is_symlink():
                    found.append(entry)
            elif not entry.is_symlink() and entry.is_file():
                found.append(entry)
        return found

    @staticmethod
    def _stale_temps(path: Path) -> list[Path]:
        """Return rewrite temp files of ``path`` past omp's grace period."""
        try:
            entries = sorted(path.parent.iterdir())
        except OSError:
            return []
        now = time.time()
        return [
            entry
            for entry in entries
            if _temp_owner(entry.name) == path.name
            and not entry.is_symlink()
            and entry.is_file()
            and not _modified_within(entry, _TEMP_GRACE_SECONDS, now)
        ]

    def session_sidecars(self, session: Session) -> list[Path]:
        """Return what travels with a transcript: ``S/`` and every backup.

        A backup left behind would be renamed back to ``S.jsonl`` by omp's next
        listing of the old bucket, bringing the moved session back as a copy.
        The move's units carry these with the transcript, so a session moves
        whole or not at all.

        Args:
            session: The session being relocated.

        Returns:
            The existing artifacts directory and backups.
        """
        path = session.primary_path
        if path is None:
            return []
        artifacts = self._artifacts(path)
        return ([artifacts] if artifacts is not None else []) + self._session_backups(path)

    def correlated_paths(self, session: Session) -> list[Path]:
        """Return what omp's own delete removes beside the transcript, and debris.

        That is ``S/``, every ``S.jsonl.*.bak`` — left behind, omp would restore
        the session from one — and rewrite temp files past omp's grace period.
        Each is built from the transcript's own name, never searched for by
        pattern. Lock files are never included.

        Args:
            session: The session being deleted.

        Returns:
            The paths.
        """
        path = session.primary_path
        if path is None or not path.name.endswith(".jsonl"):
            return []
        return self.session_sidecars(session) + self._stale_temps(path)

    # --- guards ------------------------------------------------------------

    def _shape_problem(self, path: Path, header: dict, *, rewriting: bool) -> str | None:
        """Return why a session's files are not what ``sx`` can delete or rewrite whole.

        Args:
            path: The transcript.
            header: Its session header.
            rewriting: True for a move, which also needs a header it can
                rewrite faithfully and no archived copy sharing ``S/``.

        Returns:
            A refusal reason, or ``None``.
        """
        if path.name.endswith(".jsonl"):
            artifacts = path.parent / path.name[: -len(".jsonl")]
            if artifacts.is_symlink() or self._session_backups(path, links=True):
                return "its artifacts directory or a backup is a symbolic link (refused)"
        if not rewriting:
            return None
        for key in ("previousSessionFiles", "additionalDirectories"):
            value = header.get(key)
            if value is not None and not isinstance(value, list):
                return f"records {key} in a form sx does not know (refused)"
        if path.name.endswith(".jsonl"):
            archive = self._archived(path.parent, path.name[: -len(".jsonl")])
            if archive is not None:
                return (
                    f"Oh My Pi's gc also holds an archived copy, {short_path(archive)}, that "
                    "shares its artifacts (refused)"
                )
        return None

    def destructive_refusal(self, session: Session, *, rewriting: bool) -> str | None:
        """Return why ``session`` must not be deleted or moved now, if it must not.

        Asked again at the moment of the operation. The session's lease — by
        the id it was listed under and the one in its file name — and the
        publish lock are checked first, whether or not the transcript is there:
        omp's rewrite can move the transcript aside for a moment while it holds
        both. Then, for a transcript that is there: a file that does not begin
        the way omp writes one, a title slot that is not 256 bytes, a header
        that now names another session or records a format version ``sx``
        does not know, a symlinked artifacts directory or backup, and a sub-
        agent or lock file showing omp at work all refuse, for a delete and a
        move alike. A move also refuses a header it could not rewrite
        faithfully. A file ``sx`` does not recognize is always refused.

        Args:
            session: The session about to be deleted or moved.
            rewriting: True for a move, False for a delete.

        Returns:
            A refusal reason, or ``None``. A transcript already gone is left
            to the caller to report.
        """
        reason = _unrecognized_refusal(session)
        if reason is not None:
            return reason
        path = session.primary_path
        if path is None:
            return None
        project = session.project_path
        reason = self._lease_reason(self._lease_ids(path, session.session_id), project)
        if reason is None:
            reason = _lock_reason(path)
        if reason is not None or not os.path.lexists(path):
            return reason
        layout = _layout(path)
        if layout == _UNKNOWN:
            return _LAYOUT_REFUSAL
        if layout == _SLOT_SIZE:
            return _SLOT_REFUSAL
        header = self.read_header(path)
        if header is None:
            return "no longer holds a session header; the format may have changed (refused)"
        if (header.get("id") or path.stem) != session.session_id:
            return "now holds a different session (refused)"
        reason = _unknown_version(header) or self._shape_problem(path, header, rewriting=rewriting)
        if reason is not None or not path.name.endswith(".jsonl"):
            return reason
        return self._subagent_reason(path.parent / path.name[: -len(".jsonl")], project)

    # --- delete --------------------------------------------------------------

    def delete(self, session: Session, *, dry_run: bool = False) -> DeleteResult:
        """Delete a session's files, backups first, stopping at the first that will not go.

        omp's ``deleteSessionWithArtifacts`` removes the transcript, then
        ``S/``, then every ``S.jsonl.*.bak``. ``sx`` goes the other way round
        and stops at the first thing it cannot remove, so a backup never
        outlives its transcript — it would bring the session back: every
        backup first, then the transcript, then its artifacts directory and
        stale rewrite temp files. A failure part-way can therefore leave some
        backups gone and the rest of the session in place; it is never all or
        nothing. Shared stores and lock files stay, and so does the artifacts
        directory when ``omp gc`` also holds an archived copy of the session,
        which it still belongs to. When the transcript is missing, nothing is
        deleted. Whether omp has the session open is decided on a fresh look
        at the running omp processes (:meth:`_fresh_scan`).

        Args:
            session: The session to remove.
            dry_run: If True, report what would happen and change nothing.

        Returns:
            A :class:`DeleteResult`. Whatever a failure left in place is listed
            as refused, with why.
        """
        with self._fresh_scan():
            return self._delete_session(session, dry_run=dry_run)

    def _delete_session(self, session: Session, *, dry_run: bool) -> DeleteResult:
        """Carry out :meth:`delete`.

        Args:
            session: The session to remove.
            dry_run: If True, report what would happen and change nothing.

        Returns:
            A :class:`DeleteResult`.
        """
        reason = self._refusal(session, rewriting=False)
        path = session.primary_path
        if reason is not None or path is None:
            target = path or Path(f"<{self.name}:{session.session_id}>")
            skipped = {target: reason} if reason is not None else {}
            return DeleteResult(dry_run=dry_run, skipped=skipped)
        result = DeleteResult(dry_run=dry_run)
        artifacts = self._artifacts(path)
        backups = self._session_backups(path)
        try:
            os.lstat(path)
        except (FileNotFoundError, NotADirectoryError):
            if artifacts is None and not backups:
                result.skipped[path] = "does not exist"
            else:
                result.skipped[path] = (
                    "the transcript is missing, so nothing of this session is deleted: Oh My Pi "
                    "may be rewriting it, and would restore it from a backup (refused)"
                )
            return result
        except (OSError, ValueError):
            pass  # the guard reports why it cannot be checked
        rest: list[Path] = []
        if artifacts is not None:
            archive = self._archived(artifacts.parent, artifacts.name)
            if archive is None:
                rest.append(artifacts)
            else:
                result.skipped[artifacts] = (
                    f"also belongs to this session's archived copy {short_path(archive)} (kept)"
                )
        rest.extend(self._stale_temps(path))
        steps = [
            (backups, "Oh My Pi would restore the session from it if the rest went"),
            (list(session.paths), "its artifacts stay with it"),
            (rest, ""),
        ]
        for index, (targets, consequence) in enumerate(steps):
            part = self._delete_paths(targets, dry_run=dry_run)
            _merge(result, part)
            stuck = next((p for p in targets if p in part.refused), None)
            if stuck is None:
                continue
            why = (
                f"not deleted: {stuck.name} could not be removed, and {consequence} (refused)"
            )
            for later, _ in steps[index + 1 :]:
                for leftover in later:
                    result.skipped.setdefault(leftover, why)
            break
        return result

    # --- move ----------------------------------------------------------------

    def plan_move(self, sessions: list[Session], old: Path, new: Path) -> MovePlan:
        """Plan a move: the inherited plan, cut into whole sessions, plus memory.

        Each session's transcript, ``S/`` and backups form one unit
        (:meth:`session_sidecars`), so a session moves whole or not at all.
        Memory folders are planned on their own (:meth:`_plan_memory`). The
        running omp processes are looked at once, afresh, for the whole plan.

        Args:
            sessions: Sessions selected for the move.
            old: The project directory being moved away from.
            new: The project directory being moved to.

        Returns:
            A :class:`MovePlan`.
        """
        with self._fresh_scan():
            plan = super().plan_move(sessions, old, new)
            if self._yields_to() is None:
                self._plan_memory(plan)
        return plan

    def _plan_memory(self, plan: MovePlan) -> None:
        """Plan the project's memory folders, which omp names for each cwd.

        One for the moved directory and one for every distinct session ``cwd``
        under it, each relocated to the name for its re-pointed path — merged
        entry by entry into a folder already there, collisions refused. A
        folder stays when a session recording that directory is not being
        re-pointed, when another directory's sessions encode to the same name,
        and when anything in it was written in the last 90 seconds — omp's
        memory pipeline runs in the background after a session ends, and a
        leftover delete refuses the same folder for the same reason. Each
        relocation runs as a unit of its own.

        Memory is planned the way the sessions are: a folder that is itself
        moving is vacated before anything moves into it, and each relocation
        is judged against the store as the ones before it leave it
        (:meth:`_namespace_of`). So in a chained move — ``/x/app`` to
        ``/x/app-v1`` while ``/x/app/v1`` has memory, whose folder name
        ``/x/app-v1`` shares — ``/x/app/v1``'s folder moves on first and
        ``/x/app``'s takes its name whole, rather than being merged into it
        and split. Two folders that would trade names stay where they are.

        Args:
            plan: The plan to extend (mutated in place).
        """
        memories = self.memories_dir()
        rewritten = set(plan.rewrites)
        cwds = {str(plan.old)}
        staying: set[str] = set()
        for session in plan.sessions:
            cwd = session.project_path
            if not cwd or not is_under(cwd, plan.old):
                continue
            cwds.add(cwd)
            if session.primary_path not in rewritten:
                staying.add(cwd)
        by_folder: dict[Path, list[str]] = {}
        for cwd in sorted(cwds):
            by_folder.setdefault(memories / memory_folder_name(cwd), []).append(cwd)
        owners: dict[str, set[str]] | None = None
        wanted: list[_MemoryMove] = []
        for source, members in by_folder.items():
            target = repoint(members[0], plan.old, plan.new)
            if source.is_symlink() or not source.is_dir() or target is None:
                continue
            destination = memories / memory_folder_name(target)
            if destination == source:
                continue
            stays = next((cwd for cwd in members if cwd in staying), None)
            if stays is not None:
                plan.blocked[source] = (
                    f"a session in {stays} is not being moved, so its memory stays (refused)"
                )
                continue
            if owners is None:
                owners = self._memory_owners()
            others = sorted(c for c in owners.get(source.name, ()) if not is_under(c, plan.old))
            if others:
                plan.blocked[source] = (
                    f"also the memory folder of {others[0]}, whose path encodes alike (refused)"
                )
                continue
            if not _safe_idle(source, _MEMORY_IDLE_SECONDS):
                plan.blocked[source] = (
                    f"was written in the last {_MEMORY_IDLE_SECONDS} seconds: Oh My Pi may still "
                    "be writing this project's memory (refused)"
                )
                continue
            wanted.append(_MemoryMove(source, destination))
        ordered, cyclic = self._ordered(wanted)
        for move in cyclic:
            plan.blocked[move.folder] = (
                "would trade names with another memory folder in the same move (refused)"
            )
        namespace = self._namespace_of(plan)
        roots = self.store_roots()
        moved = 0
        for move in ordered:
            before = len(plan.relocations)
            self._plan_relocation(move.folder, move.destination, plan, roots)
            pairs = plan.relocations[before:]
            if pairs:
                moved += 1
                self._apply([MoveUnit(MoveUnit.ENTRY, relocations=pairs)], namespace)
        if moved:
            detail = f"{moved} Oh My Pi memory folder(s) move too"
            plan.note = f"{plan.note}; {detail}" if plan.note else detail

    def _plan_relocation(self, src: Path, dst: Path, plan: MovePlan, roots: list[Path]) -> None:
        """Plan moving a folder whole — or, for a bucket holding lock files, around them.

        omp's lock files belong where omp created them, so a bucket holding
        any moves entry by entry, merging as usual, and its lock files stay —
        unless the destination is the bucket itself under another spelling (a
        case-only rename on a case-insensitive file system), which is renamed
        whole, its lock files with it, since they stay where they are.
        Everything is judged against the store as the plan leaves it
        (:meth:`_namespace_of`), not as the disk is now.

        Args:
            src: The folder to move.
            dst: Where it should end up.
            plan: The plan to record into.
            roots: The adapter's store roots, for the containment guard.
        """
        namespace = self._namespace_of(plan)
        names = namespace.names(src) if src.parent == self.sessions_dir() else None
        if (
            src == dst
            or not isinstance(names, set)
            or not any(map(_is_lock_name, names))
            or same_entry(src, dst)
        ):
            super()._plan_relocation(src, dst, plan, roots)
            return
        if namespace.lexists(dst) and not namespace.is_dir(dst):
            plan.blocked[dst] = "destination exists and is not a directory (refused)"
            return
        for name in sorted(names):
            if not _is_lock_name(name):
                self._plan_entry(src / name, dst / name, plan, roots)

    def _foreign_transcripts(self, folder: Path, selected: list[Session], old: Path) -> list[Path]:
        """Return what in ``folder`` must not travel with it, session data included.

        On top of Pi's rule (another project's transcripts, files ``sx`` does
        not recognize, and transcripts in use), anything that holds session
        data no transcript in the bucket accounts for stays put, which keeps
        the bucket from moving whole: a backup without its transcript (omp
        would restore it wherever it lands), an artifacts directory without
        its transcript — an unsent draft, or a session ``omp gc`` archived —
        a directory ``omp session unpack`` set aside (``S.unpack-<id>.bak/``),
        and an archived transcript kept in the bucket.

        Args:
            folder: The folder about to be relocated.
            selected: The sessions being moved out of it.
            old: The project directory being moved away from.

        Returns:
            The paths that stay put.
        """
        stay = set(super()._foreign_transcripts(folder, selected, old))
        if folder.parent != self.sessions_dir():
            return sorted(stay)
        try:
            entries = sorted(folder.iterdir())
        except OSError:
            return sorted(stay)
        transcripts = {entry.name for entry in entries if is_transcript(entry)}
        for entry in entries:
            name = entry.name
            if name in transcripts or name.startswith(".") or _is_lock_name(name):
                continue
            if name.endswith(_ARCHIVE_SUFFIX):
                stay.add(entry)
            elif not entry.is_symlink() and entry.is_dir():
                if f"{name}.jsonl" not in transcripts:
                    stay.add(entry)
            elif name.endswith(".bak") and not any(
                _owns_backup(transcript, name) for transcript in transcripts
            ):
                stay.add(entry)
        return sorted(stay)

    def _unit_check(
        self, unit: MoveUnit, plan: MovePlan, simulation: Namespace | None
    ) -> tuple[tuple[Path, str] | None, bool]:
        """Check a unit as Pi does, and probe the sessions an entry is named for.

        An entry moving on its own out of a bucket — no transcript of its
        travels in the same unit — is checked against the lease of the
        session its name gives (an artifacts directory ``<ts>_<id>``, a
        backup ``<ts>_<id>.jsonl.<x>.bak``): omp may be writing that session
        with no transcript on disk yet. Every unit of a real move looks at the
        running omp processes afresh, right before it runs.

        Args:
            unit: The unit about to run.
            plan: The plan being executed.
            simulation: The dry run's namespace, or ``None`` to ask the disk.

        Returns:
            ``(problem, merge)``, as :meth:`EncodedFolderAdapter._unit_check`.
        """
        with self._fresh_scan():
            return self._check_unit(unit, plan, simulation)

    def _check_unit(
        self, unit: MoveUnit, plan: MovePlan, simulation: Namespace | None
    ) -> tuple[tuple[Path, str] | None, bool]:
        """Carry out :meth:`_unit_check`.

        Args:
            unit: The unit about to run.
            plan: The plan being executed.
            simulation: The dry run's namespace, or ``None`` to ask the disk.

        Returns:
            ``(problem, merge)``.
        """
        problem, merge = super()._unit_check(unit, plan, simulation)
        if problem is not None or merge or unit.kind != MoveUnit.ENTRY:
            return problem, merge
        root = self.sessions_dir()
        for src, _ in unit.relocations:
            if src.parent.parent != root or is_transcript(src):
                continue
            named = _entry_session_id(src.name)
            if named is None:
                continue
            reason = self._lease_reason([named])
            if reason is not None:
                return (src, reason), False
        return None, False

    def _memory_owners(self) -> dict[str, set[str]]:
        """Map each memory folder name to the recorded directories that produce it."""
        owners: dict[str, set[str]] = {}
        for path in self.session_files():
            cwd = self.recorded_project(path)
            if cwd:
                owners.setdefault(memory_folder_name(cwd), set()).add(cwd)
        return owners

    def move(self, plan: MovePlan, *, dry_run: bool = False) -> MoveResult:
        """Carry out a plan, then remove memory folders a merge left empty.

        A real move looks at the running omp processes afresh for each unit
        (:meth:`_unit_check`); a dry run looks once, for all of them.

        Args:
            plan: The plan produced by :meth:`plan_move`.
            dry_run: If True, report what would happen and change nothing.

        Returns:
            A :class:`MoveResult`; relocated memory is listed in ``moved``.
        """
        if dry_run:
            with self._fresh_scan():
                return super().move(plan, dry_run=True)
        result = super().move(plan, dry_run=dry_run)
        if not dry_run:
            memories = self.memories_dir()
            for source, _ in result.moved:
                folder = source.parent
                if folder.parent != memories or folder.is_symlink() or not folder.is_dir():
                    continue
                try:
                    folder.rmdir()  # only when empty: anything left keeps it
                except OSError:
                    pass
        return result

    # --- lock files ------------------------------------------------------------

    @staticmethod
    def _bucket_locks(folder: Path) -> list[Path]:
        """Return the lock files directly inside a bucket."""
        try:
            return sorted(entry for entry in folder.iterdir() if _is_lock_name(entry.name))
        except OSError:
            return []

    def _delete_paths(self, paths: list[Path], *, dry_run: bool) -> DeleteResult:
        """Remove paths through the inherited guards, never one of omp's lock files.

        A lock file named directly is refused, and a bucket holding lock files
        is emptied around them instead of removed — as omp's own delete leaves
        a session's lock files where they are. Lock files inside an artifacts
        directory go with it, as they do in omp's delete. Nothing at all is
        removed from a store this adapter leaves to Pi.

        Args:
            paths: Files/dirs to remove.
            dry_run: If True, only compute the result; remove nothing.

        Returns:
            A :class:`DeleteResult`.
        """
        owner = self._yields_to()
        if owner is not None:
            reason = self._yield_reason(owner)
            return DeleteResult(dry_run=dry_run, skipped=dict.fromkeys(paths, reason))
        root = self.sessions_dir()
        targets: list[Path] = []
        kept: dict[Path, str] = {}
        for path in paths:
            if path.parent.parent == root and _is_lock_name(path.name):
                kept[path] = "an Oh My Pi lock file (refused)"
            elif path.parent == root and not path.is_symlink() and self._bucket_locks(path):
                try:
                    entries = sorted(path.iterdir())
                except OSError as exc:
                    kept[path] = f"cannot read: {exc}"
                    continue
                targets.extend(entry for entry in entries if not _is_lock_name(entry.name))
            else:
                targets.append(path)
        result = super()._delete_paths(targets, dry_run=dry_run)
        result.skipped.update(kept)
        return result

    # --- orphans -------------------------------------------------------------

    def _kept_reason(self, folder: Path) -> str | None:
        """Return why a bucket holds session data sx cannot show, which must stay.

        Such a bucket is never offered or removed whole. It holds a backup no
        transcript in it accounts for — any ``*.bak`` that is not one of a
        present transcript's ``S.jsonl.*.bak`` files (those go with their
        session, as a session delete takes them): omp restores
        ``S.jsonl.<id>.bak``, and ``omp session unpack`` sets a session aside
        as ``S.jsonl.unpack-<id>.bak`` and ``S.unpack-<id>.bak/`` — an
        archived transcript, the artifacts of a session ``omp gc`` archived, or
        an unsent draft whose transcript is not there.

        Args:
            folder: A bucket.

        Returns:
            A refusal reason, or ``None``.
        """
        try:
            entries = sorted(folder.iterdir())
        except OSError as exc:
            return f"cannot read: {exc}"
        names = {entry.name for entry in entries}
        transcripts = {name for name in names if name.endswith(".jsonl") and not name.startswith(".")}
        archive = self.archive_dir()
        for entry in entries:
            name = entry.name
            if name.endswith(".bak"):
                session = any(_owns_backup(t, name) for t in transcripts)
                if not session or entry.is_symlink() or not entry.is_file():
                    return f"holds a backup Oh My Pi keeps ({name}) (refused)"
                continue
            if name.endswith(_ARCHIVE_SUFFIX):
                return f"holds an archived session ({name}) (refused)"
            if name.startswith(".") or entry.is_symlink() or not entry.is_dir():
                continue
            if self._archived(folder, name, archive) is not None:
                return f"holds the artifacts of a session Oh My Pi archived ({name}) (refused)"
            if f"{name}.jsonl" not in names and (entry / _DRAFT_MARKER).exists():
                return f"holds an unsent draft ({name}) (refused)"
        return None

    def _archive_note(self, folder: Path) -> str | None:
        """Return a note that a bucket's archived sessions stay, if it has any."""
        archived = self.archive_dir() / folder.name
        try:
            found = next(archived.glob(f"*{_ARCHIVE_SUFFIX}"), None) is not None
        except OSError:
            found = False
        if not found:
            return None
        return f"its archived sessions in {short_path(archived)} are not affected"

    def _writing_in(self, folder: Path) -> str | None:
        """Return why omp may be writing in a bucket right now.

        Args:
            folder: A bucket.

        Returns:
            A reason when a lock file, a rewrite temp file or a ``gc --compress``
            scratch directory is recent, or an artifacts directory without its
            transcript is not yet a day idle.
        """
        try:
            entries = sorted(folder.iterdir())
        except OSError as exc:
            return f"cannot read: {exc}"
        names = {entry.name for entry in entries}
        now = time.time()
        for entry in entries:
            name = entry.name
            if _is_lock_name(name) and _modified_within(entry, _LOCK_WINDOW_SECONDS, now):
                return f"Oh My Pi is writing in this folder ({name}) (refused)"
            if _temp_owner(name) is not None and _modified_within(
                entry, _TEMP_GRACE_SECONDS, now
            ):
                return f"Oh My Pi is rewriting a session in this folder ({name}) (refused)"
            if name.startswith(_COMPRESS_SCRATCH) and _modified_within(
                entry, _TEMP_GRACE_SECONDS, now
            ):
                return f"Oh My Pi's gc is compressing in this folder ({name}) (refused)"
            if (
                _name_id(f"{name}.jsonl") is not None
                and f"{name}.jsonl" not in names
                and not entry.is_symlink()
                and entry.is_dir()
                and not _safe_idle(entry, _ARTIFACT_IDLE_SECONDS)
            ):
                return f"{name} may belong to a session Oh My Pi has not saved yet (refused)"
        return None

    def _stale_artifacts(self, folder: Path) -> list[Orphan]:
        """Return artifacts directories whose session is gone.

        Only a directory named the way omp names sessions qualifies, and only
        when its transcript is missing, no backup could restore it, ``omp gc``
        did not archive its session, it holds no unsent draft, and nothing in
        it changed for a day. A directory set aside as ``*.bak`` never does.

        Args:
            folder: A bucket.

        Returns:
            One :attr:`OrphanKind.STALE_SESSION` orphan per directory.
        """
        try:
            entries = sorted(folder.iterdir())
        except OSError:
            return []
        names = {entry.name for entry in entries}
        restorable = {_backup_owner(name) for name in names} - {None}
        archive = self.archive_dir()
        orphans: list[Orphan] = []
        for entry in entries:
            transcript = f"{entry.name}.jsonl"
            if (
                _name_id(transcript) is None
                or entry.name.endswith(".bak")
                or transcript in names
                or transcript in restorable
                or entry.is_symlink()
                or not entry.is_dir()
                or (entry / _DRAFT_MARKER).exists()
                or self._archived(folder, entry.name, archive) is not None
                or not _safe_idle(entry, _ARTIFACT_IDLE_SECONDS)
            ):
                continue
            orphans.append(
                Orphan(
                    self.name,
                    OrphanKind.STALE_SESSION,
                    [entry],
                    f"artifacts of a deleted session ({transcript} is gone)",
                    dir_size(entry),
                )
            )
        return orphans

    def _folder_orphans(self, folder: Path) -> list[Orphan]:
        """Return a bucket's orphans, keeping what omp can still bring back.

        A bucket is offered whole — empty, or for a dead project — only when it
        holds no session data sx cannot show (see :meth:`_kept_reason`) and omp
        is not writing in it; otherwise only its stale artifacts directories
        are offered, each on its own. Backups and archives never are. A bucket
        holding nothing but lock files is not offered at all — they are never
        removed — and one holding more says they stay.

        Args:
            folder: A bucket directly under the sessions directory.

        Returns:
            Its orphans, possibly none.
        """
        stale = self._stale_artifacts(folder)
        if self._kept_reason(folder) is not None or self._writing_in(folder) is not None:
            found = super()._folder_orphans(folder)
            return [o for o in found if o.paths != [folder]] + stale
        locks = self._bucket_locks(folder)
        if not _folder_transcripts(folder):
            try:
                only_locks = len(locks) == sum(1 for _ in folder.iterdir())
            except OSError:
                only_locks = True
            if locks and only_locks:
                return []
            found = super()._folder_orphans(folder)
        else:
            found = super()._folder_orphans(folder)
            if not any(orphan.kind is OrphanKind.DEAD_PROJECT for orphan in found):
                return found + stale
        notes = [f"its {len(locks)} lock file(s) stay"] if locks else []
        archived = self._archive_note(folder)
        if archived:
            notes.append(archived)
        if not notes:
            return found
        return [
            replace(orphan, reason="; ".join([orphan.reason, *notes]))
            if orphan.paths == [folder]
            else orphan
            for orphan in found
        ]

    def _orphan_revalidation(self, orphan: Orphan) -> dict[Path, str]:
        """Ask again, at the moment of deletion, whether each path is still an orphan.

        :meth:`HarnessAdapter.delete_orphan` has already refused anything that
        changed since the scan, was written in the last 90 seconds, or left the
        store. On top of Pi's own re-checks — a fragment must still be one, and
        a dead project's bucket must still hold nothing but sessions whose
        recorded directories are definitely gone — omp's are made: a
        transcript, or any session named by an artifacts directory in a
        bucket, that omp holds or is writing; a session that appeared in an
        empty bucket; a backup, draft or archive; and artifacts whose session
        came back.

        Whether omp holds a session is decided on a fresh look at the running
        omp processes, taken for this orphan alone (:meth:`_fresh_scan`).

        Args:
            orphan: The orphan about to be deleted.

        Returns:
            Path -> refusal reason, for the paths that must stay.
        """
        with self._fresh_scan():
            refused = super()._orphan_revalidation(orphan)
            if self._yields_to() is not None:
                return refused
            for path in orphan.paths:
                if path in refused or not os.path.lexists(path):
                    continue
                reason = self._orphan_refusal(orphan.kind, path)
                if reason is not None:
                    refused[path] = reason
            return refused

    def _orphan_refusal(self, kind: OrphanKind, path: Path) -> str | None:
        """Return why one orphan path must stay, or ``None`` if it may go.

        Args:
            kind: The orphan's kind.
            path: One of its paths.

        Returns:
            A refusal reason, or ``None``; a missing path is left to the guard.
        """
        root = self.sessions_dir()
        if is_transcript(path) and not path.is_symlink() and path.is_file():
            return self._live_reason(path, self._lease_ids(path, None), None)
        if path.is_symlink() or not path.is_dir():
            return None
        if path.parent == root:
            return self._bucket_refusal(path, kind)
        if path.parent.parent == root:
            return self._artifacts_refusal(path)
        return None

    def _bucket_refusal(self, folder: Path, kind: OrphanKind) -> str | None:
        """Return why a whole bucket must stay.

        Every transcript in it is checked for omp holding or writing it — an
        empty bucket must hold no session at all, and a dead project's none
        in a format ``sx`` does not know — and so is every session an
        artifacts directory without its transcript is named for; then the
        bucket must still hold nothing omp keeps, and omp must not be writing
        in it.

        Args:
            folder: The bucket.
            kind: Why it was offered.

        Returns:
            A refusal reason, or ``None``.
        """
        try:
            names = {entry.name for entry in folder.iterdir()}
        except OSError as exc:
            return f"cannot read: {exc}"
        for path in _folder_transcripts(folder):
            header = self.read_header(path)
            if kind is OrphanKind.EMPTY and header is not None:
                return f"now holds a session ({path.name}) (refused)"
            problem = self.format_refusal(path, header) if header is not None else None
            if problem is not None:
                return f"{path.name}: {problem}"
            reason = self._transcript_live_reason(path, header)
            if reason is not None:
                return f"{path.name}: {reason}"
        for name in sorted(names):
            named = _name_id(f"{name}.jsonl")
            if named is None or f"{name}.jsonl" in names or not (folder / name).is_dir():
                continue
            reason = self._lease_reason([named])
            if reason is not None:
                return f"{name}: {reason}"
        return self._kept_reason(folder) or self._writing_in(folder)

    def _artifacts_refusal(self, directory: Path) -> str | None:
        """Return why a stale artifacts directory must stay after all.

        Args:
            directory: An artifacts directory offered as stale.

        Returns:
            A refusal reason, or ``None``.
        """
        transcript = directory.parent / f"{directory.name}.jsonl"
        if os.path.lexists(transcript):
            return "its session exists again (refused)"
        if directory.name.endswith(".bak"):
            return "Oh My Pi set it aside as a backup (refused)"
        archive = self._archived(directory.parent, directory.name)
        if archive is not None:
            return f"its session is archived in {short_path(archive)} (refused)"
        try:
            names = [entry.name for entry in directory.parent.iterdir()]
        except OSError as exc:
            return f"cannot read: {exc}"
        if any(_backup_owner(name) == transcript.name for name in names):
            return "Oh My Pi can still restore its session from a backup (refused)"
        if (directory / _DRAFT_MARKER).exists():
            return "holds an unsent draft (refused)"
        if not _safe_idle(directory, _ARTIFACT_IDLE_SECONDS):
            return "was written in the last day (refused)"
        named = _name_id(transcript.name)
        reason = self._lease_reason([named]) if named else _UNSURE
        return reason or self._subagent_reason(directory)

    # --- project-scoped state ------------------------------------------------

    def _project_memory(self, project: str) -> Path | None:
        """Return the project's memory folder, when it is the project's alone.

        Args:
            project: The project directory.

        Returns:
            The folder, or ``None`` when there is none or another directory's
            sessions encode to the same name.
        """
        folder = self.memories_dir() / memory_folder_name(project)
        if folder.is_symlink() or not folder.is_dir():
            return None
        owners = self._memory_owners().get(folder.name, set())
        return None if owners - {project} else folder

    def project_leftovers(self, project: str) -> ProjectLeftovers | None:
        """Offer the project's bucket and memory folder once its last session goes.

        The bucket is offered as Pi's is, except when it holds session data sx
        cannot show — a backup, an archive, the artifacts of an archived
        session, an unsent draft. The memory folder is listed with its Markdown
        documents. Nothing is offered from a store left to Pi.

        Args:
            project: The project directory.

        Returns:
            A :class:`ProjectLeftovers`, or ``None`` when there is nothing.
        """
        if self._yields_to() is not None:
            return None
        leftovers = super().project_leftovers(project)
        if leftovers is not None and self._kept_reason(leftovers.paths[0]) is not None:
            leftovers = None
        memory = self._project_memory(project)
        if memory is None:
            return leftovers
        if leftovers is None:
            leftovers = ProjectLeftovers(project_path=project)
        leftovers.memory_files.extend(sorted(p for p in memory.rglob("*.md") if p.is_file()))
        leftovers.paths.append(memory)
        leftovers.size_bytes += dir_size(memory)
        return leftovers

    def delete_project_leftovers(
        self, leftovers: ProjectLeftovers, *, dry_run: bool = False
    ) -> DeleteResult:
        """Remove what :meth:`project_leftovers` offered, if still left over.

        The bucket is checked as Pi's is, and as a bucket offered as an orphan
        is (:meth:`_still_in_use`); the memory folder must be this project's
        own, no session may record the project any more, and the memory must
        not have been written in the last 90 seconds.

        Args:
            leftovers: The plan produced by :meth:`project_leftovers`.
            dry_run: If True, report what would happen and change nothing.

        Returns:
            A :class:`DeleteResult`.
        """
        with self._fresh_scan():
            return self._delete_leftovers(leftovers, dry_run=dry_run)

    def _delete_leftovers(self, leftovers: ProjectLeftovers, *, dry_run: bool) -> DeleteResult:
        """Carry out :meth:`delete_project_leftovers`.

        Args:
            leftovers: The plan produced by :meth:`project_leftovers`.
            dry_run: If True, report what would happen and change nothing.

        Returns:
            A :class:`DeleteResult`.
        """
        memory = self.memories_dir() / memory_folder_name(leftovers.project_path)
        others = [path for path in leftovers.paths if path != memory]
        result = super().delete_project_leftovers(replace(leftovers, paths=others), dry_run=dry_run)
        if memory not in leftovers.paths:
            return result
        owner = self._yields_to()
        reason = (
            self._yield_reason(owner)
            if owner is not None
            else self._memory_refusal(leftovers.project_path, memory)
        )
        if reason is not None:
            result.skipped[memory] = reason
            return result
        _merge(result, self._delete_paths([memory], dry_run=dry_run))
        return result

    def _memory_refusal(self, project: str, memory: Path) -> str | None:
        """Return why a project's memory folder must stay after all.

        Args:
            project: The project directory.
            memory: Its memory folder.

        Returns:
            A refusal reason, or ``None``.
        """
        if not memory.exists():
            return None
        if memory.is_symlink():
            return "the memory folder is a symbolic link (refused)"
        for path in self.session_files():
            cwd = self.recorded_project(path)
            if cwd == project:
                return f"Oh My Pi still has a session for this project ({path.name}) (refused)"
            if cwd and memory_folder_name(cwd) == memory.name:
                return f"also the memory folder of {cwd}, whose path encodes alike (refused)"
        if not _safe_idle(memory, _MEMORY_IDLE_SECONDS):
            return f"was written in the last {_MEMORY_IDLE_SECONDS} seconds (refused)"
        return None

    def _still_in_use(self, folder: Path) -> str | None:
        """Return why a leftover bucket must stay, adding omp's own reasons.

        Args:
            folder: The bucket about to be removed.

        Returns:
            Pi's reasons; then what keeps an empty bucket offered as an orphan
            (:meth:`_bucket_refusal`): a fragment or an artifacts directory
            whose session omp holds or is writing, session data sx cannot show
            (see :meth:`_kept_reason`), or omp writing in the folder. ``None``
            when it may go.
        """
        reason = super()._still_in_use(folder)
        if reason is not None or not folder.exists():
            return reason
        return self._bucket_refusal(folder, OrphanKind.EMPTY)
