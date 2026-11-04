"""Tests for sx.util: human_size, is_within, iter_jsonl, parse_ts."""

from __future__ import annotations

from pathlib import Path

from sx.util import human_size, is_within, iter_jsonl, parse_ts, rewrite_jsonl, sanitize_text


def test_c1_controls_are_neutralized_like_c0_ones():
    """A C1 control never reaches the terminal: U+009B alone acts as ESC [.

    Each is shown as the sequence it stands for, led by the inert ␛ glyph, and
    ordinary text around it, tabs and newlines included, is untouched.
    """
    raw = "a\u009b31mb\u009d0;title\u009c\tc\n\u0085d"
    assert sanitize_text(raw) == "a␛[31mb␛]0;title␛\\\tc\n␛Ed"
    assert sanitize_text("\x1b[2J\x07") == "␛[2J␇"
    assert sanitize_text("é ✓  ") == "é ✓  "


def test_human_size_units():
    """Byte counts format to the expected short human strings."""
    assert human_size(0) == "0 B"
    assert human_size(1536) == "1.5 KB"
    assert human_size(5 * 1024 * 1024) == "5.0 MB"


def test_is_within_inside_root(tmp_path: Path):
    """A path inside one of the roots is reported as within."""
    inner = tmp_path / "sub" / "file.txt"
    inner.parent.mkdir(parents=True)
    inner.write_text("x")
    assert is_within(inner, [tmp_path]) is True


def test_is_within_outside_root(tmp_path: Path):
    """A path well outside the root is reported as not within."""
    assert is_within(Path("/etc/passwd"), [tmp_path]) is False


def test_is_within_nonexistent_path(tmp_path: Path):
    """A non-existent path under a root still resolves as within (no crash)."""
    missing = tmp_path / "does-not-exist.txt"
    assert is_within(missing, [tmp_path]) is True
    # A non-existent path outside the root is not within.
    assert is_within(Path("/nope/also-missing"), [tmp_path]) is False


def test_iter_jsonl_skips_blank_and_malformed(tmp_path: Path):
    """iter_jsonl yields only dict lines, skipping blanks/garbage/non-dicts."""
    f = tmp_path / "mixed.jsonl"
    f.write_text(
        '{"a": 1}\n'
        "\n"
        "   \n"
        "not json at all\n"
        "{bad json}\n"
        "[1, 2, 3]\n"  # valid JSON but not a dict -> skipped
        '"just a string"\n'  # valid JSON but not a dict -> skipped
        '{"b": 2}\n',
        encoding="utf-8",
    )
    objs = list(iter_jsonl(f))
    assert objs == [{"a": 1}, {"b": 2}]


def test_deeply_nested_lines_are_skipped_not_fatal(tmp_path: Path):
    """A line nested past the parser's limit is skipped, not raised.

    ``json`` raises :class:`RecursionError` for it, which is not a
    ``ValueError`` — uncaught, one crafted line aborted loading a transcript
    and left a move's temporary file behind.
    """
    nested = "[" * 200_000 + "]" * 200_000
    f = tmp_path / "deep.jsonl"
    original = f'{{"cwd":"/old"}}\n{nested}\n'.encode()
    f.write_bytes(original)

    assert list(iter_jsonl(f)) == [{"cwd": "/old"}]

    changed, error = rewrite_jsonl(f, lambda obj: {**obj, "cwd": "/new"})
    assert (changed, error) == (1, None)
    assert f.read_bytes() == original.replace(b"/old", b"/new")
    assert [p.name for p in tmp_path.iterdir()] == ["deep.jsonl"]


def test_iter_jsonl_missing_file(tmp_path: Path):
    """Reading a non-existent file yields nothing rather than raising."""
    assert list(iter_jsonl(tmp_path / "absent.jsonl")) == []


def test_parse_ts_valid_iso():
    """A real ISO-8601 timestamp parses to a datetime (not None)."""
    assert parse_ts("2026-05-30T00:35:37.638Z") is not None


def test_parse_ts_invalid_inputs():
    """Empty strings and non-strings return None."""
    assert parse_ts("") is None
    assert parse_ts(None) is None
    assert parse_ts(123) is None
    assert parse_ts("not-a-timestamp") is None
