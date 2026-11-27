"""Read protocol-buffer messages without a schema or a dependency.

Antigravity keeps each step of a conversation as a serialized protobuf message.
``sx`` only reads a handful of fields, so instead of compiling the harness's
``.proto`` files it walks the wire format directly: every field is a tag
(field number and wire type) followed by a varint, a fixed-width value, or a
length-delimited payload. A length-delimited payload may be a string, raw
bytes or a nested message; the caller decides which, by asking for it.

The reader is strict. A truncated buffer, a varint longer than ten bytes, a
field number of zero, a length that runs past the end, or a wire type this
format does not use (groups, which no current encoder emits, included) raises
:class:`ProtoError` rather than guessing. Callers that read untrusted bytes
catch it and treat the message as unreadable.
"""

from __future__ import annotations

from datetime import UTC, datetime

#: Wire types, from the protobuf encoding specification.
VARINT = 0
FIXED64 = 1
LEN = 2
FIXED32 = 5

#: How deep :meth:`Message.get` may descend; real messages are far shallower,
#: and the bound keeps a crafted payload from recursing without end.
MAX_DEPTH = 64


class ProtoError(ValueError):
    """Raised for bytes that are not a well-formed protobuf message."""


def read_varint(data: bytes, pos: int) -> tuple[int, int]:
    """Decode one base-128 varint.

    Args:
        data: The buffer.
        pos: Where the varint starts.

    Returns:
        ``(value, position after it)``.

    Raises:
        ProtoError: If the buffer ends mid-varint or the varint is longer than
            ten bytes.
    """
    value = 0
    shift = 0
    for _ in range(10):
        if pos >= len(data):
            raise ProtoError("truncated varint")
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7
    raise ProtoError("varint longer than ten bytes")


def parse(data: bytes) -> list[tuple[int, int, int | bytes]]:
    """Split a serialized message into its fields, in wire order.

    Args:
        data: A serialized message.

    Returns:
        ``(field number, wire type, value)`` per field: an ``int`` for varints,
        ``bytes`` for every other wire type (eight or four raw bytes for the
        fixed types).

    Raises:
        ProtoError: If ``data`` is not a well-formed message.
    """
    fields: list[tuple[int, int, int | bytes]] = []
    pos = 0
    end = len(data)
    while pos < end:
        key, pos = read_varint(data, pos)
        number, wire = key >> 3, key & 7
        if number == 0:
            raise ProtoError("field number 0")
        if wire == VARINT:
            value, pos = read_varint(data, pos)
            fields.append((number, wire, value))
        elif wire == LEN:
            size, pos = read_varint(data, pos)
            if size > end - pos:
                raise ProtoError("length runs past the end")
            fields.append((number, wire, bytes(data[pos : pos + size])))
            pos += size
        elif wire == FIXED64:
            if pos + 8 > end:
                raise ProtoError("truncated fixed64")
            fields.append((number, wire, bytes(data[pos : pos + 8])))
            pos += 8
        elif wire == FIXED32:
            if pos + 4 > end:
                raise ProtoError("truncated fixed32")
            fields.append((number, wire, bytes(data[pos : pos + 4])))
            pos += 4
        else:
            raise ProtoError(f"unsupported wire type {wire}")
    return fields


class Message:
    """A parsed message, read by field number.

    Parsing is done once, on construction; nested messages are parsed only
    when asked for. Every accessor returns an empty result, never raises, for
    a field that is absent or holds the wrong wire type, so callers can read
    optional fields directly.

    Args:
        data: A serialized message.

    Raises:
        ProtoError: If ``data`` is not a well-formed message.
    """

    __slots__ = ("_fields",)

    def __init__(self, data: bytes) -> None:
        """Parse ``data``."""
        self._fields = parse(data)

    @classmethod
    def maybe(cls, data: object) -> Message | None:
        """Parse ``data`` if it is a well-formed message, else return ``None``.

        Args:
            data: Bytes from a database column, or anything else.

        Returns:
            The message, or ``None`` for non-bytes and malformed bytes.
        """
        if not isinstance(data, (bytes, bytearray, memoryview)):
            return None
        try:
            return cls(bytes(data))
        except ProtoError:
            return None

    def numbers(self) -> list[int]:
        """Return the field numbers present, in wire order (repeats included)."""
        return [number for number, _, _ in self._fields]

    def varints(self, number: int) -> list[int]:
        """Return every varint stored under ``number``."""
        return [v for n, w, v in self._fields if n == number and w == VARINT]  # type: ignore[misc]

    def varint(self, number: int, default: int | None = None) -> int | None:
        """Return the last varint under ``number`` (protobuf's rule for scalars)."""
        values = self.varints(number)
        return values[-1] if values else default

    def raw(self, number: int) -> list[bytes]:
        """Return every length-delimited payload under ``number``."""
        return [v for n, w, v in self._fields if n == number and w == LEN]  # type: ignore[misc]

    def texts(self, number: int) -> list[str]:
        """Return every payload under ``number`` that decodes as UTF-8 text."""
        out: list[str] = []
        for payload in self.raw(number):
            try:
                out.append(payload.decode("utf-8"))
            except UnicodeDecodeError:
                continue
        return out

    def text(self, number: int) -> str:
        """Return the last UTF-8 string under ``number``, or ``""``."""
        values = self.texts(number)
        return values[-1] if values else ""

    def children(self, number: int) -> list[Message]:
        """Return every payload under ``number`` that parses as a message."""
        out: list[Message] = []
        for payload in self.raw(number):
            try:
                out.append(Message(payload))
            except ProtoError:
                continue
        return out

    def child(self, number: int) -> Message | None:
        """Return the last message under ``number``, or ``None``.

        Protobuf merges repeated occurrences of a singular message field; the
        last occurrence is close enough for the fields ``sx`` reads, which a
        conforming writer emits once.
        """
        values = self.children(number)
        return values[-1] if values else None

    def get(self, *path: int) -> Message | None:
        """Follow a path of nested message fields.

        Args:
            *path: Field numbers, outermost first.

        Returns:
            The message at the end of the path, or ``None`` if any step is
            missing or malformed (or the path is deeper than :data:`MAX_DEPTH`).
        """
        if len(path) > MAX_DEPTH:
            return None
        current: Message | None = self
        for number in path:
            if current is None:
                return None
            current = current.child(number)
        return current


def timestamp(message: Message | None) -> datetime | None:
    """Convert a ``google.protobuf.Timestamp`` to a local, timezone-aware datetime.

    Harnesses record UTC; like :func:`sx.util.parse_ts`, the result is in the
    local time zone so it displays on the same clock as file times.

    Args:
        message: A Timestamp message (``seconds`` = 1, ``nanos`` = 2), or ``None``.

    Returns:
        The instant, or ``None`` when absent, zero, or out of range.
    """
    if message is None:
        return None
    seconds = message.varint(1, 0) or 0
    nanos = message.varint(2, 0) or 0
    if seconds <= 0 or seconds >= 1 << 40:
        return None
    try:
        return datetime.fromtimestamp(seconds + nanos / 1e9, tz=UTC).astimezone()
    except OverflowError, OSError, ValueError:
        return None
