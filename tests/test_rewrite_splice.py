"""A rewritten line changes only where its record did.

A move re-points a path recorded inside JSONL transcripts. The harnesses write
those lines with ``JSON.stringify`` or ``serde_json``, whose number formatting
is not Python's (``1.5e-7`` where Python prints ``1.5e-07``, ``0.00001`` where
it prints ``1e-05``), and other writers add spaces or escape ``/``. The README
promises that the inverse move restores a transcript byte for byte, so
:func:`sx.util.rewrite_jsonl` must keep every byte of a line except the values
the transform changed.

Every fixture is valid JSON that Python's ``json`` would not write back the
same way, so an implementation that re-serializes a line, whether the changed
one or any other, fails here. Each test asserts exact bytes.

The transforms mirror the shapes of the adapters' own, which are being
reworked separately: Pi re-points its header's ``cwd``; Oh My Pi also
replaces or deletes ``additionalDirectories`` and replaces or appends
``previousSessionFiles``; Claude re-points top-level ``cwd`` values; Codex
re-points ``payload.cwd``.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from sx import util
from sx.util import dumps_record, repoint, rewrite_jsonl

OLD = "/old/proj"
NEW = "/new/proj"

Transform = Callable[[dict], object]

#: Numbers as JSON.stringify, serde_json and others print them. Python's json
#: would write the first eight back as 1e-05, 1.5e-07, 4.5e-05, 100000.0,
#: 100000.0, 2000.0, 1.5 and 0; the two big integers are regression guards.
NUMBERS = (
    b'{"score":0.00001,"p":1.5e-7,"cost":0.000045,"e5":1e5,"E5":1E5,"plus":2e+3,'
    b'"trail":1.50,"negzero":-0,"big":10000000000000000,"bigger":12345678901234567890}'
)

#: Escapes Python's json would canonicalize: ``\\/`` to ``/`` and ``\\u00e9`` to a
#: raw ``é``. The quote value holds escaped quotes, a brace, a bracket, a comma
#: and a colon, and ends with an escaped backslash, so a scanner that does not
#: understand strings loses its place in it.
ESCAPES = (
    b'{"url":"https:\\/\\/example.com\\/a","name":"caf\\u00e9",'
    b'"quote":"say \\"hi\\" {not} [an], object: \\\\"}'
)


def _swap(data: bytes, before: bytes, after: bytes) -> bytes:
    """Return ``data`` with its single occurrence of ``before`` replaced.

    Args:
        data: The original bytes.
        before: A fragment that must occur exactly once in ``data``.
        after: Its replacement.

    Returns:
        The edited bytes.
    """
    assert data.count(before) == 1, before
    return data.replace(before, after)


def _rewrite(tmp_path: Path, data: bytes, transform: Transform) -> bytes:
    """Rewrite ``data`` as a JSONL file and return the file's new bytes.

    Args:
        tmp_path: The test's scratch directory.
        data: The file's original contents.
        transform: The rewrite's transform.

    Returns:
        The file's contents after the rewrite.
    """
    path = tmp_path / "s.jsonl"
    path.write_bytes(data)
    changed, error = rewrite_jsonl(path, transform)
    assert error is None
    assert changed > 0
    assert [p.name for p in tmp_path.iterdir()] == ["s.jsonl"]
    return path.read_bytes()


def _claude(old: str, new: str) -> Transform:
    """Build Claude's transform: copy the record, re-point its top-level ``cwd``."""

    def transform(obj: dict) -> dict | None:
        """Return the re-pointed record, or ``None``."""
        updated = repoint(obj.get("cwd", ""), old, new)
        if updated is None:
            return None
        replacement = dict(obj)
        replacement["cwd"] = updated
        return replacement

    return transform


def _codex(old: str, new: str) -> Transform:
    """Build Codex's transform: re-point ``payload.cwd`` in a copied payload."""

    def transform(obj: dict) -> dict | None:
        """Return the re-pointed record, or ``None``."""
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            return None
        updated = repoint(payload.get("cwd", ""), old, new)
        if updated is None:
            return None
        replacement = dict(obj)
        replacement["payload"] = {**payload, "cwd": updated}
        return replacement

    return transform


def _pi(old: str, new: str) -> Transform:
    """Build Pi's transform: re-point the ``cwd`` of the first session header."""
    seen = False

    def transform(obj: dict) -> dict | None:
        """Return the re-pointed header, or ``None`` for every other record."""
        nonlocal seen
        if seen or obj.get("type") != "session":
            return None
        seen = True
        updated = repoint(obj.get("cwd", ""), old, new)
        return None if updated is None else {**obj, "cwd": updated}

    return transform


def _omp(old: str, new: str, moved_from: str) -> Transform:
    """Build a transform shaped like Oh My Pi's header rewrite, for the splicer.

    ``cwd`` is re-pointed in place; ``additionalDirectories`` is re-pointed, or
    deleted once nothing but the new ``cwd`` is left in it; the transcript's old
    location is added to ``previousSessionFiles``, appending the key when the
    header has none. This exercises every edit the splicer supports: replacing
    a value, deleting a key and adding one. The adapter itself no longer
    re-points ``additionalDirectories`` (it follows omp's ``moveTo``, which only
    filters the new ``cwd`` out), but its edits are a subset of these.

    Args:
        old: The project directory being moved away from.
        new: The project directory being moved to.
        moved_from: The transcript's location before the move.

    Returns:
        The transform.
    """

    def transform(header: dict) -> dict | None:
        """Return the moved header, or ``None`` for every other record."""
        if header.get("type") != "session":
            return None
        updated = dict(header)
        cwd = repoint(header.get("cwd", ""), old, new)
        if cwd is not None:
            updated["cwd"] = cwd
        directories = header.get("additionalDirectories")
        if isinstance(directories, list):
            moved = [repoint(item, old, new) or item for item in directories]
            if cwd is not None:
                moved = [item for item in moved if item != cwd]
            if not moved and cwd is not None:
                del updated["additionalDirectories"]
            elif moved != directories:
                updated["additionalDirectories"] = moved
        previous = header.get("previousSessionFiles")
        files: list = []
        for item in previous if isinstance(previous, list) else []:
            if item not in files:
                files.append(item)
        if moved_from not in files:
            files.append(moved_from)
        if files != previous:
            updated["previousSessionFiles"] = files
        return updated if updated != header else None

    return transform


# --- values outside the change keep their bytes ------------------------------


def test_numbers_keep_their_writers_form_on_a_changed_line(tmp_path: Path):
    """``1.5e-7`` must not come back as ``1.5e-07`` because ``cwd`` moved."""
    line = b'{"type":"user","cwd":"/old/proj","usage":' + NUMBERS + b',"after":1e5}\n'

    out = _rewrite(tmp_path, line, _claude(OLD, NEW))

    assert out == _swap(line, b'"cwd":"/old/proj"', b'"cwd":"/new/proj"')


def test_escapes_and_spacing_survive_on_a_changed_line(tmp_path: Path):
    """Spaced separators, ``\\/``, ``\\u00e9`` and escaped quotes stay as written."""
    line = (
        b'{"quote": "say \\"hi\\" {not} [an], object: \\\\", "url": "https:\\/\\/example.com", '
        b'"name": "caf\\u00e9", "cwd": "/old/proj", "text": "cd /old/proj && ls", "n": 1e5}\n'
    )

    out = _rewrite(tmp_path, line, _claude(OLD, NEW))

    # The path in prose is history, not structure: only the cwd value moves.
    assert out == _swap(line, b'"cwd": "/old/proj"', b'"cwd": "/new/proj"')


def test_a_key_spelled_with_an_escape_keeps_its_spelling(tmp_path: Path):
    """The splicer matches keys by decoded value, then leaves the key's bytes alone."""
    line = b'{"c\\u0077d":"/old/proj","n":1E5}\n'

    out = _rewrite(tmp_path, line, _claude(OLD, NEW))

    assert out == b'{"c\\u0077d":"/new/proj","n":1E5}\n'


def test_a_nested_value_is_spliced_in_place(tmp_path: Path):
    """Codex's ``payload.cwd`` moves; the objects and arrays around it do not."""
    line = (
        b'{"timestamp":"t","type":"turn_context","payload":{"sandbox_policy":'
        b'{"mode":"workspace-write","writable_roots":["\\/tmp", 1e5]},'
        b'"cwd":"/old/proj/sub","effort":1e0,"usage":' + NUMBERS + b',"model":"gpt"},"n":1.5e-7}\n'
    )

    out = _rewrite(tmp_path, line, _codex(OLD, NEW))

    assert out == _swap(line, b'"cwd":"/old/proj/sub"', b'"cwd":"/new/proj/sub"')


def test_crlf_lines_keep_their_endings_and_bytes(tmp_path: Path):
    """CRLF endings survive on changed and untouched lines alike."""
    data = (
        b'{"type":"user","cwd":"/old/proj","n":1e5}\r\n'
        b'{"type":"user","cwd":"/elsewhere","p":1.5e-7}\r\n'
        b'{"type": "user", "cwd": "/old/proj/sub", "url": "https:\\/\\/x"}\r\n'
    )

    out = _rewrite(tmp_path, data, _claude(OLD, NEW))

    assert out == (
        b'{"type":"user","cwd":"/new/proj","n":1e5}\r\n'
        b'{"type":"user","cwd":"/elsewhere","p":1.5e-7}\r\n'
        b'{"type": "user", "cwd": "/new/proj/sub", "url": "https:\\/\\/x"}\r\n'
    )


def test_lines_the_transform_leaves_alone_stay_byte_identical(tmp_path: Path):
    """Untouched lines are copied, never re-serialized, however they were written.

    The changed line is already in ``dumps_record`` form, so this isolates the
    lines the transform returned ``None`` for.
    """
    untouched = (
        b'{"type":"assistant","usage":' + NUMBERS + b"}\n"
        + ESCAPES + b"\n"
        + b'{"type": "user", "message": {"content": [1, {"a": 1e5}], "n": 10000000000000000}}\n'
        + b'{"type":"user","cwd":"/elsewhere","n":1E5}\r\n'
        + b'{"type":"user","cwd":"/old/projects","p":1.5e-7}\n'
    )
    data = b'{"type":"user","cwd":"/old/proj"}\n' + untouched

    out = _rewrite(tmp_path, data, _claude(OLD, NEW))

    assert out == b'{"type":"user","cwd":"/new/proj"}\n' + untouched


def test_a_replacement_equal_to_the_record_keeps_the_line(tmp_path: Path):
    """A transform that returns an equal copy still counts, and changes no byte."""
    line = b'{"n": 1e5, "cwd": "/old/proj"}\n'
    path = tmp_path / "s.jsonl"
    path.write_bytes(line)

    assert rewrite_jsonl(path, dict) == (1, None)
    assert path.read_bytes() == line


# --- the three edits ---------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (b'{"cwd":"/old/proj","n":1e5}', b'{"cwd":"/new/proj","n":1e5}'),
        (b'{"n":1e5,"cwd":"/old/proj","p":1.5e-7}', b'{"n":1e5,"cwd":"/new/proj","p":1.5e-7}'),
        (b'{"n":1e5,"cwd":"/old/proj"}', b'{"n":1e5,"cwd":"/new/proj"}'),
        (b'{ "n" : 1e5 ,\t"cwd" : "/old/proj" }', b'{ "n" : 1e5 ,\t"cwd" : "/new/proj" }'),
        (b'  {"cwd":"/old/proj","n":1e5}  ', b'  {"cwd":"/new/proj","n":1e5}  '),
    ],
    ids=["first", "middle", "last", "spaced", "padded-line"],
)
def test_a_replaced_value_is_overwritten_in_place(tmp_path: Path, line: bytes, expected: bytes):
    """Only the value's own span changes, wherever the key sits."""
    out = _rewrite(tmp_path, line + b"\n", lambda obj: {**obj, "cwd": NEW})

    assert out == expected + b"\n"


@pytest.mark.parametrize(
    ("line", "keys", "expected"),
    [
        (b'{"x":[],"a":1e5,"b":2}', ("x",), b'{"a":1e5,"b":2}'),
        (b'{"a":1e5,"x":[],"b":2}', ("x",), b'{"a":1e5,"b":2}'),
        (b'{"a":1e5,"b":2,"x":[]}', ("x",), b'{"a":1e5,"b":2}'),
        (b'{"x":[]}', ("x",), b"{}"),
        (b'{ "x": [], "a": 1e5 }', ("x",), b'{ "a": 1e5 }'),
        (b'{"a": 1e5, "x": [], "b": 2}', ("x",), b'{"a": 1e5, "b": 2}'),
        (b'{"a": 1e5, "x": []}', ("x",), b'{"a": 1e5}'),
        (b'{"x":1,"y":2,"a":1e5,"z":3}', ("x", "y", "z"), b'{"a":1e5}'),
        (b'{"a":1e5,"x":1,"y":2,"b":2}', ("x", "y"), b'{"a":1e5,"b":2}'),
        (b'{"x":1,"a":1e5,"y":2,"b":2}', ("x", "y"), b'{"a":1e5,"b":2}'),
        (b'{"x":1, "y":2}', ("x", "y"), b"{}"),
        (b'{ "x":1, "y":2 }', ("x", "y"), b"{  }"),
    ],
    ids=[
        "first",
        "middle",
        "last",
        "only",
        "spaced-first",
        "spaced-middle",
        "spaced-last",
        "lead-and-tail",
        "adjacent",
        "interleaved",
        "all",
        "all-spaced",
    ],
)
def test_a_deleted_key_takes_exactly_one_comma(
    tmp_path: Path, line: bytes, keys: tuple[str, ...], expected: bytes
):
    """The key, its value and one adjacent comma go; nothing else moves."""

    def transform(obj: dict) -> dict:
        """Return the record without ``keys``."""
        return {key: value for key, value in obj.items() if key not in keys}

    out = _rewrite(tmp_path, line + b"\n", transform)

    assert out == expected + b"\n"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (b'{"a":1e5}', b'{"a":1e5,"k":["/v"]}'),
        (b'{"a": 1e5 }', b'{"a": 1e5,"k":["/v"] }'),
        (b"{}", b'{"k":["/v"]}'),
        (b"{ }", b'{ "k":["/v"]}'),
    ],
    ids=["compact", "spaced", "empty", "empty-spaced"],
)
def test_an_added_key_goes_last_in_its_object(tmp_path: Path, line: bytes, expected: bytes):
    """A new key is inserted before the closing brace, as JavaScript orders it."""
    out = _rewrite(tmp_path, line + b"\n", lambda obj: {**obj, "k": ["/v"]})

    assert out == expected + b"\n"


def test_edits_combine_within_and_across_objects(tmp_path: Path):
    """A delete, an add and a nested add land together without disturbing the rest."""
    line = b'{"a":1e5,"x":[1.5e-7],"p":{"q":2e+3},"z":"\\/"}\n'

    def transform(obj: dict) -> dict:
        """Delete ``x``, add a nested key to ``p`` and a top-level ``k``."""
        return {"a": obj["a"], "p": {**obj["p"], "r": 1}, "z": obj["z"], "k": True}

    out = _rewrite(tmp_path, line, transform)

    assert out == b'{"a":1e5,"p":{"q":2e+3,"r":1},"z":"\\/","k":true}\n'


def test_emptying_a_nested_object_keeps_the_rest_of_the_line(tmp_path: Path):
    """Deleting every key of an object leaves it empty, and its siblings as written."""
    line = b'{"n":1e5,"p":{"x":1.5e-7,"y":[2]},"z":"\\/"}\n'

    out = _rewrite(tmp_path, line, lambda obj: {**obj, "p": {}})

    assert out == b'{"n":1e5,"p":{},"z":"\\/"}\n'


def test_a_changed_value_is_written_as_dumps_record_writes_it(tmp_path: Path):
    """The spliced value is compact, keeps non-ASCII raw, and escapes a lone surrogate."""
    line = b'{"cwd":"/old/proj","n":1e5}\n'
    moved = '/new/café "q" \\ \t \ud83d'

    out = _rewrite(tmp_path, line, lambda obj: {**obj, "cwd": moved, "dirs": [moved, 1.5e-7]})

    value = b'"/new/caf\xc3\xa9 \\"q\\" \\\\ \\t \\ud83d"'
    assert dumps_record(moved) == value
    assert out == b'{"cwd":' + value + b',"n":1e5,"dirs":[' + value + b",1.5e-07]}\n"


@pytest.mark.parametrize(
    ("before", "after", "written"),
    [(b"1", True, b"true"), (b"0.0", -0.0, b"-0.0"), (b"1", 1.0, b"1.0")],
    ids=["int-to-bool", "zero-sign", "int-to-float"],
)
def test_a_value_changed_only_in_type_is_rewritten(
    tmp_path: Path, before: bytes, after: object, written: bytes
):
    """``1 == True`` and ``0.0 == -0.0`` in Python, but they are written differently."""
    line = b'{"v":' + before + b',"n":1e5}\n'

    out = _rewrite(tmp_path, line, lambda obj: {**obj, "v": after})

    assert out == b'{"v":' + written + b',"n":1e5}\n'


# --- each harness's transform ------------------------------------------------


def test_pi_re_points_its_header_in_place(tmp_path: Path):
    """Pi's header ``cwd`` moves; a message quoting the old path does not."""
    header = b'{"type": "session", "version": 3, "id": "S1", "timestamp": "t", "cwd": "/old/proj"}\n'
    message = b'{"type":"message","message":{"content":"cd /old/proj","cost":0.000045}}\n'

    out = _rewrite(tmp_path, header + message, _pi(OLD, NEW))

    assert out == _swap(header, b'"/old/proj"', b'"/new/proj"') + message


def test_omp_replaces_values_and_appends_previous_session_files(tmp_path: Path):
    """``cwd`` and ``additionalDirectories`` change in place; the new key goes last."""
    slot = b'{"type":"title","v":1,"title":"caf\\u00e9","pad":"  "}\n'
    header = (
        b'{"type":"session","version":3,"id":"S1","timestamp":"t","cwd":"/old/proj",'
        b'"additionalDirectories":["/old/proj/lib", "/other"],"title":"caf\\u00e9","n":1e5}\n'
    )

    out = _rewrite(tmp_path, slot + header, _omp(OLD, NEW, "/store/--old-proj--/S1.jsonl"))

    assert out == slot + (
        b'{"type":"session","version":3,"id":"S1","timestamp":"t","cwd":"/new/proj",'
        b'"additionalDirectories":["/new/proj/lib","/other"],"title":"caf\\u00e9","n":1e5,'
        b'"previousSessionFiles":["/store/--old-proj--/S1.jsonl"]}\n'
    )


def test_omp_deletes_emptied_directories_and_replaces_previous_session_files(tmp_path: Path):
    """An emptied ``additionalDirectories`` goes with one comma; the list it had grows."""
    header = (
        b'{"type":"session","version":3,"id":"S1","cwd":"/old/proj",'
        b'"additionalDirectories":["/new/proj"],"previousSessionFiles":["\\/a.jsonl"],"n":1e5}\n'
    )

    out = _rewrite(tmp_path, header, _omp(OLD, NEW, "/store/S1.jsonl"))

    # The replaced array is written whole, by dumps_record rules; the rest keeps its bytes.
    assert out == (
        b'{"type":"session","version":3,"id":"S1","cwd":"/new/proj",'
        b'"previousSessionFiles":["/a.jsonl","/store/S1.jsonl"],"n":1e5}\n'
    )


#: A Claude transcript: subdirectory and sibling cwds, CRLF, and a torn line.
CLAUDE_FILE = b"".join(
    [
        b'{"type":"user","cwd":"/old/proj","usage":' + NUMBERS + b"}\n",
        (
            b'{"type": "assistant", "cwd": "/old/proj/sub", "text": "cd /old/proj", '
            b'"url": "https:\\/\\/x", "name": "caf\\u00e9"}\r\n'
        ),
        b'{"type":"user","cwd":"/old/projects","n":1E5}\n',
        b"{ half-written\n",
        b'{"type":"user","cwd":"/old/proj","last":1.5e-7}\n',
    ]
)

#: A Codex rollout: the cwd is nested in each record's payload.
CODEX_FILE = b"".join(
    [
        b'{"type":"session_meta","payload":{"id":"S1","cwd":"/old/proj","n":0.00001}}\n',
        (
            b'{"type":"turn_context","payload":{"cwd":"/old/proj/sub","effort":1.5e-7,'
            b'"roots":["\\/tmp"]}}\n'
        ),
    ]
)

#: A Pi session: a spaced header, then a message with Node-printed costs.
PI_FILE = b"".join(
    [
        b'{"type": "session", "version": 3, "id": "S1", "cwd": "/old/proj", "n": 1e5}\n',
        b'{"type":"message","message":{"cost":{"total":0.000081}}}\n',
    ]
)


@pytest.mark.parametrize(
    ("data", "make"),
    [(CLAUDE_FILE, _claude), (CODEX_FILE, _codex), (PI_FILE, _pi)],
    ids=["claude", "codex", "pi"],
)
def test_a_move_and_its_inverse_restore_every_byte(
    tmp_path: Path, data: bytes, make: Callable[[str, str], Transform]
):
    """There and back again is the identity, as the README promises."""
    path = tmp_path / "s.jsonl"
    path.write_bytes(data)

    assert rewrite_jsonl(path, make(OLD, NEW))[1] is None
    moved = path.read_bytes()
    assert moved != data
    assert len(moved.splitlines()) == len(data.splitlines())

    assert rewrite_jsonl(path, make(NEW, OLD))[1] is None
    assert path.read_bytes() == data


def test_a_changed_last_line_without_a_terminator_gains_one(tmp_path: Path):
    """The one byte a rewrite adds, so a harness's next append starts a new line."""
    data = b'{"cwd":"/elsewhere"}\n{"cwd":"/old/proj","n":1e5}'

    out = _rewrite(tmp_path, data, _claude(OLD, NEW))

    assert out == b'{"cwd":"/elsewhere"}\n{"cwd":"/new/proj","n":1e5}\n'


# --- the fallback: re-serialize the whole line -------------------------------


def test_a_duplicate_key_falls_back_to_the_whole_line(tmp_path: Path):
    """Which duplicate counts is the parser's choice, so the splicer does not guess."""
    line = b'{"cwd":"/old/proj/a","n":1e5,"cwd":"/old/proj"}\n'

    out = _rewrite(tmp_path, line, _claude(OLD, NEW))

    assert out == b'{"cwd":"/new/proj","n":100000.0}\n'


def test_reordered_keys_fall_back_to_the_transforms_order(tmp_path: Path):
    """Key edits cannot reorder an object, so the record is written as returned."""
    line = b'{"n":1e5,"cwd":"/old/proj"}\n'

    out = _rewrite(tmp_path, line, lambda obj: {"cwd": NEW, "n": obj["n"]})

    assert out == b'{"cwd":"/new/proj","n":100000.0}\n'


def test_a_nested_reorder_replaces_only_that_object(tmp_path: Path):
    """The fallback is as narrow as it can be: one object, not the line."""
    line = b'{"n":1e5,"payload":{"a":1.5e-7,"cwd":"/old/proj"}}\n'

    def transform(obj: dict) -> dict:
        """Re-point ``payload.cwd``, moving it to the front of ``payload``."""
        return {**obj, "payload": {"cwd": NEW, "a": obj["payload"]["a"]}}

    out = _rewrite(tmp_path, line, transform)

    assert out == b'{"n":1e5,"payload":{"cwd":"/new/proj","a":1.5e-07}}\n'


def test_a_transform_that_edits_its_argument_still_writes_its_record(tmp_path: Path):
    """Editing in place hides the change from the diff; the check catches it."""
    line = b'{"n":1e5,"cwd":"/old/proj"}\n'

    def transform(obj: dict) -> dict:
        """Re-point ``cwd`` on the record itself."""
        obj["cwd"] = NEW
        return obj

    out = _rewrite(tmp_path, line, transform)

    assert out == b'{"n":100000.0,"cwd":"/new/proj"}\n'


@pytest.mark.parametrize(
    "bogus",
    [
        '{"n":1e5,"cwd":"/somewhere/else"}',
        '{"n":1e5,',
        '{"cwd":"/new/proj","n":1e5}',
        '{"n":100000,"cwd":"/new/proj"}',
        '{"n":1e5,"cwd":"/new/proj","extra":1}',
    ],
    ids=["wrong-value", "not-json", "wrong-order", "wrong-type", "extra-key"],
)
def test_a_splice_that_does_not_decode_to_the_record_is_never_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bogus: str
):
    """Whatever the splicer returns must decode to exactly the transform's record."""
    monkeypatch.setattr(util, "_splice_record", lambda text, old, new: bogus)
    line = b'{"n":1e5,"cwd":"/old/proj"}\n'

    out = _rewrite(tmp_path, line, lambda obj: {**obj, "cwd": NEW})

    assert out == b'{"n":100000.0,"cwd":"/new/proj"}\n'


def test_nesting_deeper_than_python_recursion_falls_back(tmp_path: Path):
    """A deep copy of a deeply nested record cannot be diffed; it is still written."""
    depth = sys.getrecursionlimit() + 100
    deep = b"[" * depth + b"]" * depth
    line = b'{"cwd":"/old/proj","deep":' + deep + b',"n":1e5}\n'

    def transform(obj: dict) -> dict:
        """Re-point ``cwd`` on a deep copy, so no value is shared with ``obj``."""
        copy = json.loads(json.dumps(obj))
        copy["cwd"] = NEW
        return copy

    out = _rewrite(tmp_path, line, transform)

    assert out == b'{"cwd":"/new/proj","deep":' + deep + b',"n":100000.0}\n'
