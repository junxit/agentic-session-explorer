"""Conservative filesystem checks shared by every adapter.

Two questions decide whether ``sx`` may treat something on disk as gone, or as
debris it can remove, and both are answered here so every harness answers them
the same way:

* :func:`project_state` — is a project directory a session recorded still
  there? Only a definite "no such file or directory" counts as missing; any
  other answer, including an unmounted volume, is :data:`UNKNOWN`, and unknown
  is never an orphan.
* :func:`survey` — what is at a path right now, as a fingerprint to compare
  with an earlier one, and when was anything there last written?

Everything here uses ``lstat`` and never follows a symbolic link to decide what
a path *is*: a link is reported as a link.
"""

from __future__ import annotations

import errno
import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

#: The answer :func:`project_state` gives.
ProjectState = Literal["present", "missing", "unknown"]

#: The directory exists (``lstat`` succeeded).
PRESENT: Final[ProjectState] = "present"
#: The directory is definitely gone: its own absence, not an unreadable or
#: unmounted parent, made ``lstat`` fail with ``ENOENT`` or ``ENOTDIR``.
MISSING: Final[ProjectState] = "missing"
#: Nothing can be said: a permission error, an unmounted volume, a dangling
#: symbolic link on the way, a relative or empty path. Never an orphan.
UNKNOWN: Final[ProjectState] = "unknown"

#: The only errors that mean "not there".
_GONE: Final = frozenset({errno.ENOENT, errno.ENOTDIR})

#: Directories under which removable and network volumes are mounted. A path
#: below one is only missing when a volume is actually mounted above it.
_MEDIA_PARENTS: Final = ("/Volumes", "/mnt", "/media", "/run/media")


def project_state(path: str | os.PathLike[str] | None) -> ProjectState:
    """Return whether a recorded project directory is present, missing, or unknown.

    ``Path.exists()`` cannot answer this: it returns ``False`` for *any*
    ``OSError``, so a live project behind a parent ``sx`` cannot traverse
    (``EACCES``, or a macOS privacy-protected folder) would look deleted — and
    its sessions would be offered for permanent deletion as a dead project's.

    Args:
        path: The directory a session recorded, as a string or path.

    Returns:
        :data:`PRESENT` when ``lstat`` succeeds; :data:`MISSING` only when it
        fails with ``ENOENT``/``ENOTDIR`` because the directory itself is gone;
        :data:`UNKNOWN` for every other error, for a path on a volume that is
        not mounted, for a path that runs through a dangling symbolic link, and
        for a value that is not an absolute path.
    """
    if path is None:
        return UNKNOWN
    try:
        text = os.fspath(path)
    except TypeError:
        return UNKNOWN
    if not isinstance(text, str) or not text or not os.path.isabs(text):
        return UNKNOWN
    try:
        os.lstat(text)
    except OSError as exc:
        if exc.errno not in _GONE:
            return UNKNOWN
        return MISSING if _gone_for_certain(text) else UNKNOWN
    except ValueError:  # an embedded NUL byte
        return UNKNOWN
    return PRESENT


def project_missing(path: str | os.PathLike[str] | None) -> bool:
    """Return True only when :func:`project_state` says the directory is missing.

    Args:
        path: The directory a session recorded.

    Returns:
        True for :data:`MISSING`; False for :data:`PRESENT` and :data:`UNKNOWN`.
    """
    return project_state(path) == MISSING


def _gone_for_certain(text: str) -> bool:
    """Decide whether an ``ENOENT``/``ENOTDIR`` from ``lstat`` is the path's own absence.

    Args:
        text: An absolute path that ``lstat`` could not find.

    Returns:
        True when the deepest ancestor that exists is an ordinary directory and
        the path is not on a volume that is away; False when the answer is in
        doubt.
    """
    ancestor = os.path.dirname(text)
    while True:
        try:
            info = os.lstat(ancestor)
        except OSError as exc:
            if exc.errno not in _GONE:
                return False
        except ValueError:
            return False
        else:
            break
        parent = os.path.dirname(ancestor)
        if parent == ancestor:
            return False  # not even the root answered: nothing can be judged
        ancestor = parent
    if stat.S_ISLNK(info.st_mode):
        # The path runs through this link. If it dangles, what it points at —
        # an unplugged drive, typically — is away, not the project.
        try:
            os.stat(ancestor)
        except (OSError, ValueError):
            return False
    try:
        real = os.path.realpath(text)
    except (OSError, ValueError):
        return False
    return not any(_volume_away(candidate) for candidate in {text, real})


def _volume_away(text: str) -> bool:
    """Return True if ``text`` lies on a removable volume that is not mounted.

    A path under ``/Volumes``, ``/mnt``, ``/media`` or ``/run/media`` counts as
    on a mounted volume only when the deepest part of it that exists sits on a
    different device than that parent — that is, when something is mounted
    there. An empty mount-point directory left behind, or no volume directory at
    all, means the volume is away.

    Args:
        text: An absolute path.

    Returns:
        True when the path's volume is, or may be, absent.
    """
    for parent in _MEDIA_PARENTS:
        if not text.startswith(parent + "/"):
            continue
        try:
            parent_device = os.lstat(parent).st_dev
        except (OSError, ValueError):
            return True
        current = text
        while current != parent:
            try:
                info = os.lstat(current)
            except OSError as exc:
                if exc.errno not in _GONE:
                    return True
                current = os.path.dirname(current)
                continue
            except ValueError:
                return True
            return info.st_dev == parent_device
        return True
    return False


@dataclass(frozen=True, slots=True)
class Survey:
    """What :func:`survey` found at a path.

    Attributes:
        fingerprint: A digest of the path and, for a directory, every entry
            under it — relative path, type, size and modification time, each
            read with ``lstat``. Two surveys of an unchanged tree agree.
        newest_ns: The newest modification time, in nanoseconds, of the path
            and everything under it; ``None`` when the path is absent or
            unreadable.
    """

    fingerprint: str
    newest_ns: int | None


def _kind(mode: int) -> str:
    """Return a one-letter type for an ``st_mode``: ``d``, ``f``, ``l`` or ``o``."""
    if stat.S_ISDIR(mode):
        return "d"
    if stat.S_ISREG(mode):
        return "f"
    if stat.S_ISLNK(mode):
        return "l"
    return "o"


def _digest(records: list[tuple]) -> str:
    """Return a short stable digest of a list of survey records."""
    payload = repr(sorted(records)).encode("utf-8", "surrogateescape")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def survey(path: Path | str) -> Survey:
    """Fingerprint ``path`` and find when anything under it was last written.

    A file contributes its type, size and modification time; a directory
    contributes the same for itself and for every entry beneath it, by path
    relative to it. Symbolic links are recorded as links and never followed, so
    a directory swapped for a link — or a link swapped for anything else —
    changes the fingerprint. An entry that cannot be read is recorded as such
    rather than skipped.

    Args:
        path: The file or directory to survey.

    Returns:
        A :class:`Survey`. Never raises.
    """
    try:
        info = os.lstat(path)
    except (OSError, ValueError) as exc:
        code = getattr(exc, "errno", None)
        return Survey(_digest([(".", "absent" if code in _GONE else "unreadable", code)]), None)
    records: list[tuple] = [(".", _kind(info.st_mode), info.st_size, info.st_mtime_ns)]
    newest = info.st_mtime_ns
    if stat.S_ISDIR(info.st_mode):
        pending: list[tuple[str, str]] = [(os.fspath(path), "")]
        while pending:
            directory, relative = pending.pop()
            try:
                with os.scandir(directory) as entries:
                    children = list(entries)
            except OSError as exc:
                records.append((relative or ".", "unreadable", exc.errno))
                continue
            for child in children:
                name = f"{relative}/{child.name}" if relative else child.name
                try:
                    child_info = child.stat(follow_symlinks=False)
                except OSError as exc:
                    records.append((name, "unreadable", exc.errno))
                    continue
                records.append(
                    (name, _kind(child_info.st_mode), child_info.st_size, child_info.st_mtime_ns)
                )
                newest = max(newest, child_info.st_mtime_ns)
                if stat.S_ISDIR(child_info.st_mode):
                    pending.append((child.path, name))
    return Survey(_digest(records), newest)
