"""A rewrite never clobbers a concurrent write, and never leaves its copy behind.

:func:`sx.util.rewrite_jsonl` re-stats the source before it swaps its copy in,
and refuses if the size *or* the modification time changed. Each half catches
a write the other cannot see:

* only mtime moves when Oh My Pi rewrites its fixed-width 256-byte title slot
  in place, which leaves the size as it was;
* only size moves when a harness appends on a filesystem whose timestamps are
  too coarse to tell the two writes apart.

Whenever the copy is not swapped in, it is removed. A stray ``.sx-move-*`` file
is a full copy of the transcript, which would outlive a later delete of the
session. A copy that is swapped in takes the file's mode, and a rewrite that
changes nothing swaps nothing in, so a harness appending to its open file
keeps appending to the live one.
"""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path

import pytest

from sx.util import dumps_record, rewrite_jsonl, write_text_atomic

REFUSED = (0, "changed while being rewritten (refused)")
HEADER = b'{"type":"session","version":3,"id":"S1","timestamp":"t","cwd":"/old/proj"}\n'


def _slot(title: str) -> bytes:
    """Build a title slot line, padded to 256 bytes as Oh My Pi pads it.

    Args:
        title: The session title.

    Returns:
        The line, newline included.
    """
    fields = {"type": "title", "v": 1, "title": title, "updatedAt": "2026-05-01T10:00:00.000Z"}
    bare = dumps_record({**fields, "pad": ""}) + b"\n"
    line = dumps_record({**fields, "pad": " " * (256 - len(bare))}) + b"\n"
    assert len(line) == 256
    return line


def _repoint(obj: dict) -> dict | None:
    """Re-point the session header's ``cwd``; leave every other record alone."""
    return {**obj, "cwd": "/new/proj"} if obj.get("type") == "session" else None


def _leftovers(directory: Path) -> list[str]:
    """Return the names of rewrite temp files left in ``directory``."""
    return sorted(p.name for p in directory.iterdir() if p.name.startswith(".sx-move-"))


# --- each half of the race check ---------------------------------------------


def test_a_same_size_rewrite_in_place_is_not_overwritten(tmp_path: Path):
    """Comparing size alone would lose Oh My Pi retitling the session mid-rewrite."""
    path = tmp_path / "s.jsonl"
    path.write_bytes(_slot("old") + HEADER)
    before = path.stat()
    retitled = _slot("new")

    def transform(obj: dict) -> dict | None:
        """Rewrite the title slot in place, as omp does, then re-point the header."""
        if obj.get("type") == "session":
            with path.open("r+b") as fh:
                fh.write(retitled)
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 5_000_000_000))
        return _repoint(obj)

    assert rewrite_jsonl(path, transform) == REFUSED
    assert path.stat().st_size == before.st_size
    assert path.read_bytes() == retitled + HEADER
    assert _leftovers(tmp_path) == []


def test_an_append_the_clock_cannot_see_is_not_overwritten(tmp_path: Path):
    """Comparing mtime alone would lose a turn appended within the same clock tick."""
    path = tmp_path / "s.jsonl"
    path.write_bytes(HEADER)
    before = path.stat()
    turn = b'{"type":"message","id":"m1"}\n'
    appended = False

    def transform(obj: dict) -> dict | None:
        """Append one turn, keeping the mtime as a coarse clock would."""
        nonlocal appended
        if not appended:
            appended = True
            with path.open("ab") as fh:
                fh.write(turn)
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        return _repoint(obj)

    assert rewrite_jsonl(path, transform) == REFUSED
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    assert path.read_bytes() == HEADER + turn
    assert _leftovers(tmp_path) == []


# --- the copy never outlives the rewrite -------------------------------------


def test_an_abandoned_rewrite_removes_its_copy(tmp_path: Path):
    """Refusing because the harness wrote meanwhile also removes the temp file."""
    path = tmp_path / "s.jsonl"
    path.write_bytes(HEADER)
    appended = False

    def transform(obj: dict) -> dict | None:
        """Append one turn mid-rewrite, as a running harness would."""
        nonlocal appended
        if not appended:
            appended = True
            with path.open("ab") as fh:
                fh.write(b'{"type":"message"}\n')
        return _repoint(obj)

    assert rewrite_jsonl(path, transform) == REFUSED
    assert _leftovers(tmp_path) == []


def test_a_failed_write_removes_its_copy(tmp_path: Path):
    """An I/O error mid-write is reported, and the half-written copy is removed."""
    path = tmp_path / "s.jsonl"
    path.write_bytes(HEADER + HEADER)
    seen = 0

    def transform(obj: dict) -> dict | None:
        """Fail as a full disk would, after the first line was written."""
        nonlocal seen
        seen += 1
        if seen == 2:
            raise OSError(errno.ENOSPC, "No space left on device")
        return _repoint(obj)

    assert rewrite_jsonl(path, transform) == (
        0,
        "write failed: [Errno 28] No space left on device",
    )
    assert path.read_bytes() == HEADER + HEADER
    assert _leftovers(tmp_path) == []


def test_a_failed_swap_removes_its_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A complete copy that cannot be swapped in is removed, not left beside the file."""
    path = tmp_path / "s.jsonl"
    path.write_bytes(HEADER)

    def refuse(src: object, dst: object) -> None:
        """Fail the swap, as an immutable file would."""
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "replace", refuse)

    assert rewrite_jsonl(path, _repoint) == (0, "write failed: [Errno 1] Operation not permitted")
    assert path.read_bytes() == HEADER
    assert _leftovers(tmp_path) == []


@pytest.mark.parametrize(
    "error", [RuntimeError("transform bug"), KeyboardInterrupt()], ids=["bug", "ctrl-c"]
)
def test_an_exception_mid_rewrite_propagates_and_removes_its_copy(
    tmp_path: Path, error: BaseException
):
    """Not only OSError: a crashing transform, or Ctrl-C, must not strand a copy."""
    path = tmp_path / "s.jsonl"
    path.write_bytes(HEADER + HEADER)
    seen = 0

    def transform(obj: dict) -> dict | None:
        """Re-point the first header, then raise on the second."""
        nonlocal seen
        seen += 1
        if seen == 2:
            raise error
        return _repoint(obj)

    with pytest.raises(type(error)):
        rewrite_jsonl(path, transform)
    assert path.read_bytes() == HEADER + HEADER
    assert _leftovers(tmp_path) == []


# --- a rewrite keeps the file as the harness made it ----------------------------------


def test_a_rewrite_keeps_the_files_mode(tmp_path: Path):
    """The copy is made ``0600``; the transcript keeps the ``0644`` its harness gave it."""
    path = tmp_path / "s.jsonl"
    path.write_bytes(HEADER)
    os.chmod(path, 0o644)

    assert rewrite_jsonl(path, _repoint) == (1, None)

    assert stat.S_IMODE(path.stat().st_mode) == 0o644


def test_an_atomic_registry_write_keeps_the_files_mode(tmp_path: Path):
    """``projects.json`` or ``.claude.json``, rewritten, is as readable as it was."""
    path = tmp_path / "registry.json"
    path.write_text('{"a": 1}\n', encoding="utf-8")
    os.chmod(path, 0o644)

    assert write_text_atomic(path, '{"a": 2}\n', expect=path.stat()) is None

    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    assert path.read_text(encoding="utf-8") == '{"a": 2}\n'


def test_a_rewrite_that_changes_nothing_leaves_the_file_in_place(tmp_path: Path):
    """Not replaced by an identical copy: the harness, holding it open, keeps appending to it."""
    path = tmp_path / "s.jsonl"
    path.write_bytes(HEADER)
    inode = path.stat().st_ino
    appended = b'{"type":"message","id":"m1"}\n'

    with path.open("ab") as harness:
        assert rewrite_jsonl(path, lambda obj: None) == (0, None)
        harness.write(appended)

    assert path.stat().st_ino == inode
    assert path.read_bytes() == HEADER + appended
    assert _leftovers(tmp_path) == []
