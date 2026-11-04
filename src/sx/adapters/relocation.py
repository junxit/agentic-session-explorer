"""The moving parts of an encoded-folder move, kept apart from the adapters.

:class:`~sx.adapters.base.EncodedFolderAdapter` plans and runs a move as atomic
units; what it needs to do that safely, and nothing harness-specific, lives
here:

* :class:`Namespace` — the store as the relocations planned (or dry-run) so far
  will leave it, so a step that depends on an earlier one is planned right;
* :class:`Recorder` — wraps a rewrite's transform and remembers each change,
  so a unit that fails later can put its transcripts back;
* :func:`rename_no_clobber` — a rename that never replaces what is already
  at the destination and never copies across file systems; with
  :func:`same_entry`, which tells a case-only rename on a case-insensitive
  file system from a destination that is really taken, and :func:`landed`,
  which settles a rename Ctrl-C cut short.
"""

from __future__ import annotations

import errno
import json
import os
import stat
from collections.abc import Callable
from pathlib import Path


class Namespace:
    """The store as the relocations planned, or dry-run, so far will leave it.

    Planning a whole move against the disk as it is now goes wrong as soon as
    one step depends on another: a folder an earlier step creates, or empties,
    is not yet there, or still full. Planning and dry runs ask this instead.
    Directory listings are read lazily, one level at a time, and every rename
    and new directory is applied on top.
    """

    #: A listing that could not be read: everything in it may exist.
    UNREADABLE = frozenset()

    def __init__(self) -> None:
        """Start with nothing read and nothing changed."""
        self._names: dict[Path, set[str] | frozenset | None] = {}
        self._original: dict[Path, set[str]] = {}
        self._dirs: set[Path] = set()

    def names(self, directory: Path) -> set[str] | frozenset | None:
        """Return the entry names in ``directory``.

        Args:
            directory: A directory, as the plan sees it.

        Returns:
            The names; ``None`` when there is no such directory; or
            :attr:`UNREADABLE` when it exists but cannot be listed.
        """
        if directory not in self._names:
            try:
                names: set[str] | frozenset | None = set(os.listdir(directory))
            except (FileNotFoundError, NotADirectoryError):
                names = None
            except (OSError, ValueError):
                names = self.UNREADABLE
            self._names[directory] = names
            self._original[directory] = set(names) if isinstance(names, set) else set()
        return self._names[directory]

    def lexists(self, path: Path) -> bool:
        """Return True if anything — a dangling link included — will be at ``path``.

        Args:
            path: The path to test.

        Returns:
            True when present, or when its folder cannot be listed.
        """
        names = self.names(path.parent)
        if names is None:
            return False
        if names is self.UNREADABLE:
            return True
        return path.name in names

    def readable(self, directory: Path) -> bool:
        """Return False if ``directory`` exists but cannot be listed."""
        return self.names(directory) is not self.UNREADABLE

    def is_dir(self, path: Path) -> bool:
        """Return True if ``path`` will be a real directory (not a link to one).

        Args:
            path: The path to test.

        Returns:
            True for a directory the plan created or moved there, or one on disk
            that is not a symbolic link.
        """
        if path in self._dirs:
            return True
        if path in self._names and self._names[path] is None:
            return False
        try:
            return stat.S_ISDIR(os.lstat(path).st_mode)
        except (OSError, ValueError):
            return False

    def preexisting(self, directory: Path) -> set[str]:
        """Return the names in ``directory`` that were there before this move.

        Args:
            directory: A destination folder.

        Returns:
            Names read from disk at that path and still there, excluding
            anything this move brought in.
        """
        names = self.names(directory)
        if not isinstance(names, set):
            return set()
        return names & self._original.get(directory, set())

    def rename(self, src: Path, dst: Path) -> None:
        """Record that ``src`` now lives at ``dst``, a directory's listing included.

        Args:
            src: The path being moved.
            dst: Where it goes.
        """
        was_dir = self.is_dir(src)
        if was_dir:
            self.names(src)  # read it now, while it is still at its old path
        source_names = self.names(src.parent)
        if isinstance(source_names, set):
            source_names.discard(src.name)
        target_names = self.names(dst.parent)
        if isinstance(target_names, set):
            target_names.add(dst.name)
        moving = [key for key in self._names if key == src or src in key.parents]
        carried = {key: self._names.pop(key) for key in moving}
        for key in moving:
            self._original.pop(key, None)
        dirs = {key for key in self._dirs if key == src or src in key.parents}
        self._dirs -= dirs
        for key, names in carried.items():
            target = dst / key.relative_to(src)
            self._names[target] = names
            self._original[target] = set()
        self._dirs.update(dst / key.relative_to(src) for key in dirs)
        if was_dir:
            self._dirs.add(dst)
        self._names[src] = None
        self._original[src] = set()

    def mkdir(self, directory: Path) -> None:
        """Record that an empty ``directory`` now exists.

        Args:
            directory: The new directory.
        """
        names = self.names(directory.parent)
        if isinstance(names, set):
            names.add(directory.name)
        self._names[directory] = set()
        self._original[directory] = set()
        self._dirs.add(directory)

    def rmdir(self, directory: Path) -> None:
        """Record that ``directory`` is gone, if it is an empty directory.

        Args:
            directory: The directory to remove.
        """
        names = self.names(directory)
        if not isinstance(names, set) or names:
            return
        parent = self.names(directory.parent)
        if isinstance(parent, set):
            parent.discard(directory.name)
        self._names[directory] = None
        self._original[directory] = set()
        self._dirs.discard(directory)


class Recorder:
    """Wrap a move transform, remembering each change so it can be undone.

    Attributes:
        changes: ``(call number, new record, original record)`` for every record
            the transform replaced, in file order.
    """

    def __init__(self, transform: Callable[[dict], object]) -> None:
        """Wrap ``transform``."""
        self._transform = transform
        self._calls = 0
        self.changes: list[tuple[int, dict, dict]] = []

    def __call__(self, obj: dict) -> object:
        """Apply the transform to one record and remember a replacement."""
        index = self._calls
        self._calls += 1
        updated = self._transform(obj)
        if isinstance(updated, dict):
            self.changes.append((index, updated, obj))
        return updated


def canonical(obj: object) -> str:
    """Serialize a record for comparison: key order and spacing do not matter."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def read_records(path: Path) -> list[dict]:
    """Return the records of a JSONL file in order, as a rewrite sees them.

    Args:
        path: The file to read.

    Returns:
        Every line that parses as a JSON object; blank, torn and non-object
        lines are skipped, exactly as :func:`sx.util.rewrite_jsonl` skips them.
    """
    found: list[dict] = []
    try:
        with path.open("rb") as fh:
            for raw in fh:
                body = raw.rstrip(b"\r\n")
                if not body:
                    continue
                try:
                    obj = json.loads(body.decode("utf-8"))
                except (UnicodeDecodeError, ValueError, RecursionError):
                    continue
                if isinstance(obj, dict):
                    found.append(obj)
    except OSError:
        pass
    return found


def is_transcript(path: Path) -> bool:
    """Return True for a ``*.jsonl`` name that is not a hidden temporary file."""
    return path.name.endswith(".jsonl") and not path.name.startswith(".")


#: Errors that mean a file system cannot hard-link, so a plain rename is used.
_NO_LINKS = frozenset(
    code
    for code in (
        getattr(errno, "EPERM", None),
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
        getattr(errno, "EMLINK", None),
        getattr(errno, "ENOSYS", None),
    )
    if code is not None
)


def _crosses_devices(src: Path) -> str:
    """Return the refusal for a rename that would cross file systems."""
    return (
        f"{src} and its destination are on different file systems; sx moves by "
        "renaming and never copies (refused)"
    )


def same_entry(src: Path, dst: Path) -> bool:
    """Return True if ``dst`` is the directory entry ``src`` is, spelled another way.

    On a case-insensitive file system — APFS and HFS+ by default — ``/x/App``
    and ``/x/app`` name one entry, so the destination of a case-only rename
    "exists" although nothing else is there. Two hard links to one file are
    two entries, not one: they live in different folders, or both names are
    listed in the same one. A symbolic link is never its target's entry.

    Args:
        src: The path being moved.
        dst: Where it goes.

    Returns:
        True only when both paths reach the same file through the same folder
        and the folder does not list them as two names; False when anything is
        in doubt.
    """
    try:
        if not os.path.samestat(os.lstat(src), os.lstat(dst)):
            return False
        if not os.path.samestat(os.stat(src.parent), os.stat(dst.parent)):
            return False  # two links to one file, in two folders
        names = set(os.listdir(dst.parent))
    except (OSError, ValueError):
        return False
    return src.name == dst.name or not {src.name, dst.name} <= names


def _finish_link(src: Path, dst: Path) -> str | None:
    """Unlink ``src`` once it is linked at ``dst``, or take the new link back.

    Whatever stops the unlink — an error, or Ctrl-C — the session must not be
    left in two places: while ``src`` is still there, ``dst`` is removed again
    before the error is reported or the interruption is let through.

    Args:
        src: The original name.
        dst: The new link to the same file.

    Returns:
        ``None`` when ``src`` is gone, otherwise why the move failed.
    """
    try:
        os.unlink(src)
    except BaseException as exc:
        try:
            if os.path.samestat(os.lstat(src), os.lstat(dst)):
                os.unlink(dst)  # still both there: take the new link back
        except OSError:
            pass
        if isinstance(exc, OSError):
            return f"move failed: {exc}"
        raise
    return None


def rename_no_clobber(src: Path, dst: Path) -> str | None:
    """Move ``src`` to ``dst`` by renaming, never replacing anything already there.

    A regular file is hard-linked into place and then unlinked, because
    ``link`` fails when the name is taken where ``rename`` would silently
    replace it. A directory is renamed: onto a non-empty directory that fails,
    and onto an empty one it replaces it, which loses nothing. Nothing is ever
    copied: a move across file systems (``EXDEV``) is refused.

    A destination that is ``src`` itself under another spelling (see
    :func:`same_entry`) is a case-only rename on a case-insensitive file
    system: it is renamed in place when its own name changes, and refused when
    only a folder above it is spelled differently, since there is nothing to
    move.

    Args:
        src: The path to move.
        dst: Where it goes; its parent must exist.

    Returns:
        ``None`` on success, otherwise why nothing was moved.
    """
    try:
        info = os.lstat(src)
    except FileNotFoundError:
        return "does not exist"
    except OSError as exc:
        return f"cannot read: {exc} (refused)"
    if os.path.lexists(dst):
        if same_entry(src, dst):
            if src.name == dst.name:
                return f"{dst} is this same file under another spelling (refused)"
            try:
                os.rename(src, dst)  # only the case of the name changes
            except OSError as exc:
                return f"move failed: {exc}"
            return None
        try:
            other = os.lstat(dst)
        except OSError:
            other = None
        if (
            other is not None
            and stat.S_ISREG(info.st_mode)
            and (other.st_dev, other.st_ino) == (info.st_dev, info.st_ino)
        ):
            # Our own link from an interrupted move: finish it.
            try:
                os.unlink(src)
            except OSError as exc:
                return f"move failed: {exc}"
            return None
        return f"already exists at destination: {dst} (refused)"
    try:
        if stat.S_ISREG(info.st_mode):
            try:
                os.link(src, dst)
            except OSError as exc:
                if exc.errno == errno.EEXIST:
                    return f"already exists at destination: {dst} (refused)"
                if exc.errno == errno.EXDEV:
                    return _crosses_devices(src)
                if exc.errno not in _NO_LINKS:
                    return f"move failed: {exc}"
            else:
                return _finish_link(src, dst)
            if os.path.lexists(dst):
                return f"already exists at destination: {dst} (refused)"
        os.rename(src, dst)
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            return _crosses_devices(src)
        if exc.errno in (errno.EEXIST, errno.ENOTEMPTY):
            return f"already exists at destination: {dst} (refused)"
        return f"move failed: {exc}"
    return None


def landed(src: Path, dst: Path) -> bool:
    """Settle a :func:`rename_no_clobber` that was interrupted, and say where it left ``src``.

    A half-done link — ``dst`` linked, ``src`` not yet unlinked — is taken
    back, so the file is in one place only.

    Args:
        src: The path that was being moved.
        dst: Where it was going.

    Returns:
        True when ``src`` is at ``dst`` now: the rename finished before the
        interruption. False when it is still at ``src``, or cannot be found.
    """
    try:
        before = os.lstat(src)
    except FileNotFoundError:
        return os.path.lexists(dst)
    except (OSError, ValueError):
        return False
    try:
        after = os.lstat(dst)
    except (OSError, ValueError):
        return False
    if not os.path.samestat(before, after):
        return False
    if same_entry(src, dst):
        # A case-only rename: done once the folder lists the new spelling.
        try:
            return src.name != dst.name and dst.name in os.listdir(dst.parent)
        except (OSError, ValueError):
            return False
    if stat.S_ISREG(before.st_mode):
        try:
            os.unlink(dst)
        except OSError:
            pass
    return False
