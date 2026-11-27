"""The schema-less protobuf reader: fields, nesting, strictness, timestamps."""

from __future__ import annotations

from datetime import datetime

import pytest

from sx.protowire import Message, ProtoError, parse, timestamp
from tests.agy_store import f, ts, varint


def test_fields_come_back_in_wire_order():
    """Varints, strings, raw bytes and fixed-width values, repeats included."""
    data = f(1, 150) + f(2, "héllo") + f(2, "again") + b"\x1d\x01\x02\x03\x04" + b"\x21" + bytes(8)
    assert parse(data) == [
        (1, 0, 150),
        (2, 2, "héllo".encode()),
        (2, 2, b"again"),
        (3, 5, b"\x01\x02\x03\x04"),
        (4, 1, bytes(8)),
    ]


def test_accessors_read_optional_fields_without_raising():
    """The last scalar wins; a missing field is empty; a nested message is parsed on demand."""
    msg = Message(
        f(1, 3) + f(1, 7) + f(2, "a") + f(3, f(1, "inner") + f(2, f(5, 9))) + f(4, b"\xff\xfe")
    )
    assert msg.varint(1) == 7 and msg.varints(1) == [3, 7]
    assert msg.text(2) == "a" and msg.text(99) == ""
    assert msg.get(3).text(1) == "inner"
    assert msg.get(3, 2).varint(5) == 9
    assert msg.get(3, 7) is None and msg.get(99) is None
    assert msg.texts(4) == []  # not UTF-8
    assert msg.children(2) == []  # "a" is not a well-formed message


@pytest.mark.parametrize(
    "data",
    [b"\x08", b"\x0a\x05ab", b"\x00\x01", b"\x0b", b"\x08" + b"\xff" * 11, b"\x15\x01\x02"],
    ids=[
        "truncated varint",
        "length past end",
        "field zero",
        "group",
        "long varint",
        "short fixed32",
    ],
)
def test_malformed_bytes_are_refused(data):
    """Nothing is guessed: a malformed message raises, and maybe() says None."""
    with pytest.raises(ProtoError):
        parse(data)
    assert Message.maybe(data) is None


def test_maybe_accepts_only_bytes():
    """A database column that is not a blob is not a message."""
    assert Message.maybe("text") is None and Message.maybe(None) is None
    assert Message.maybe(memoryview(f(1, 1))).varint(1) == 1


def test_timestamps_are_local_and_aware():
    """A Timestamp becomes a local, timezone-aware datetime; zero and absurd values do not."""
    moment = timestamp(Message(ts(1768478400, 420660000)))
    assert moment == datetime.fromtimestamp(1768478400.42066).astimezone()
    assert moment.tzinfo is not None
    assert timestamp(Message(b"")) is None
    assert timestamp(Message(ts(1 << 41))) is None
    assert timestamp(None) is None


def test_varint_encoding_round_trips():
    """The test writer and the reader agree, multi-byte values included."""
    for value in (0, 1, 127, 128, 300, 1 << 35):
        assert Message(f(1, value)).varint(1) == value
    assert varint(300) == b"\xac\x02"
