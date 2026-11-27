"""Ask whether another process holds a file, without ever taking it.

Several harnesses mark a session in use with a lock file — Oh My Pi's session
lease, Antigravity's presence lock — and keep the session's files open while
they write. ``sx`` must find out without disturbing them: a probe that took the
lock even for an instant could make the harness's own attempt fail at that
moment. Everything here only asks.

* :func:`flock_state` queries a lock file with ``F_GETLK``, which reports a
  conflicting lock and acquires nothing. On macOS ``flock`` and ``fcntl`` locks
  share one lock table, so it sees a ``flock`` too (reported with pid -1).
* :func:`list_processes` and :func:`list_open_files` run ``ps`` and ``lsof``
  by absolute path, never from ``PATH``, with a time limit.

Moved here from :mod:`sx.adapters.omp`, which keeps its own names for them.
"""

from __future__ import annotations

import os
import re
import stat
import struct
import subprocess
import sys
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - not on the platforms sx supports
    fcntl = None

#: What :func:`flock_state` reports.
FREE = "free"
HELD = "held"
UNKNOWN = "unknown"
#: Nothing at the path.
ABSENT = "absent"

#: ``struct flock`` as ``F_GETLK`` reads and writes it, on the platforms where
#: its layout was checked: macOS's ``<sys/fcntl.h>`` declares ``off_t l_start``,
#: ``off_t l_len``, ``pid_t l_pid``, ``short l_type``, ``short l_whence`` — 24
#: bytes, verified on Darwin 27 (arm64) with ``offsetof`` and against a live
#: lock. Elsewhere the layout differs (Linux and FreeBSD put other fields
#: first, or add one), so ``None``: every lock is unknown.
FLOCK_LAYOUT = "@qqihh" if sys.platform == "darwin" else None

#: How long ``ps`` or ``lsof`` may take before a scan gives up, which counts
#: as not knowing.
SCAN_TIMEOUT_SECONDS = 10.0

#: macOS's process and open-file listers, by absolute path: never looked up on
#: ``PATH``.
PS = "/bin/ps"
LSOF = "/usr/sbin/lsof"

#: The locale ``lsof`` runs in, so it prints a UTF-8 name as it is.
LSOF_LOCALE = "en_US.UTF-8"

#: What ``lsof`` escapes in a name it prints: ``\n``, ``\t``, ``\r``, ``\b``,
#: ``\f``, ``\\`` for a backslash, and ``\xNN`` for a byte it does not take for
#: a printable character in its locale. (It prints other control characters
#: as ``^`` and a letter, which cannot be told from those two characters.)
_LSOF_ESCAPE_RE = re.compile(rb"\\(?:x([0-9a-fA-F]{2})|([ntrbf\\]))")
_LSOF_ESCAPES = {b"n": b"\n", b"t": b"\t", b"r": b"\r", b"b": b"\b", b"f": b"\f", b"\\": b"\\"}


def flock_state(path: Path, *, layout: str | None = FLOCK_LAYOUT) -> str:
    """Ask the kernel whether anyone holds a lock on ``path``, taking none.

    The file is opened read-only, without following a link or creating
    anything, and queried with ``F_GETLK`` for an exclusive lock over the whole
    file: any lock another holder has — a ``flock``, a ``lockf``, shared or
    exclusive — conflicts with that and is reported. Nothing is acquired, so a
    holder trying to take the lock at the same moment always gets it.

    Args:
        path: A lock file.
        layout: The ``struct`` format of ``struct flock`` on this platform, or
            ``None`` where it is not known.

    Returns:
        :data:`HELD`, :data:`FREE`, :data:`ABSENT` when nothing is there, or
        :data:`UNKNOWN` — for anything but a regular file, a file replaced
        while it was opened, an error, an answer ``sx`` cannot read, and a
        platform whose ``struct flock`` ``sx`` does not know.
    """
    if fcntl is None or layout is None:
        return UNKNOWN
    try:
        before = os.lstat(path)
    except FileNotFoundError, NotADirectoryError:
        return ABSENT
    except OSError, ValueError:
        return UNKNOWN
    if not stat.S_ISREG(before.st_mode):
        return UNKNOWN
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    try:
        fd = os.open(path, flags)
    except FileNotFoundError, NotADirectoryError:
        return ABSENT
    except OSError:
        return UNKNOWN
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            return UNKNOWN
        query = struct.pack(layout, 0, 0, 0, fcntl.F_WRLCK, os.SEEK_SET)
        try:
            answer = fcntl.fcntl(fd, fcntl.F_GETLK, query)
        except OSError:
            return UNKNOWN
    finally:
        os.close(fd)
    kind = struct.unpack(layout, answer)[3]
    if kind == fcntl.F_UNLCK:
        return FREE
    if kind in (fcntl.F_RDLCK, fcntl.F_WRLCK):
        return HELD
    return UNKNOWN


def parse_ps(output: bytes) -> list[tuple[int, int, str]]:
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


def list_processes(*, timeout: float = SCAN_TIMEOUT_SECONDS) -> list[tuple[int, int, str]] | None:
    """List every process with ``ps``, never running anything else.

    Args:
        timeout: Seconds ``ps`` may take.

    Returns:
        ``(pid, uid, command line)`` per process; ``None`` when ``ps`` cannot
        run, fails or does not answer in time.
    """
    try:
        done = subprocess.run(
            [PS, "-axwwo", "pid=,uid=,args="],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except OSError, subprocess.SubprocessError, ValueError:
        return None
    return parse_ps(done.stdout) if done.returncode == 0 else None


def _lsof_unescape(match: re.Match[bytes]) -> bytes:
    """Return the byte one of ``lsof``'s escapes (:data:`_LSOF_ESCAPE_RE`) stands for."""
    code, letter = match.groups()
    return bytes([int(code, 16)]) if code is not None else _LSOF_ESCAPES[letter]


def parse_lsof(output: bytes) -> dict[int, list[str]]:
    """Parse ``lsof -F n0`` output: NUL-terminated fields, one set per line.

    ``lsof`` escapes some characters in the names it prints
    (:data:`_LSOF_ESCAPE_RE`); each name is unescaped, so a path holding a
    newline, a tab, a backslash or a character ``lsof`` printed as ``\\xNN``
    is the path on disk.

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


def list_open_files(
    pids: list[int], *, timeout: float = SCAN_TIMEOUT_SECONDS
) -> dict[int, list[str]] | None:
    """List what each process has open, with ``lsof``.

    ``lsof`` runs with ``-b``, so it never waits on a file system that does not
    answer (a hung network mount), and in a UTF-8 locale, so it prints a
    non-ASCII name as it is rather than as escapes.

    Args:
        pids: The processes.
        timeout: Seconds ``lsof`` may take.

    Returns:
        Each process ``lsof`` could list -> the names of its open files; a
        process missing from it could not be listed, or has exited. ``None``
        when ``lsof`` cannot run or does not answer in time.
    """
    if not pids:
        return {}
    try:
        done = subprocess.run(
            [LSOF, "-nP", "-w", "-b", "-F", "n0", "-p", ",".join(str(pid) for pid in pids)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
            check=False,
            env={**os.environ, "LC_ALL": LSOF_LOCALE},
        )
    except OSError, subprocess.SubprocessError, ValueError:
        return None
    return parse_lsof(done.stdout)  # exit status 1 only says some pid was not found
