"""Export a normalized transcript to a Markdown file.

Like :mod:`sx.render`, this module operates only on
:class:`~sx.model.Message` and :class:`~sx.model.Transcript`, so it works for
every harness — including those that keep a session's abandoned branches, which
can be exported in any :class:`~sx.model.BranchMode`. Export is offered
both as a standalone action and as an optional step before deletion, giving a
durable archive of a session that is about to be permanently removed.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

from sx.model import BranchMode, Message, Role, Session, Transcript
from sx.render import marker_label
from sx.util import sanitize_text


class SessionGoneError(Exception):
    """Raised by :func:`export_session` for a session that no longer exists.

    Nothing is written: a session whose file was deleted reads as one with no
    messages, and an archive of that would pass for a copy of it.
    """


def session_gone(adapter, session: Session) -> bool:
    """Return True if ``session`` no longer exists.

    Its main file must be gone, and its adapter must see no activity for it.
    The file alone does not settle it: an OpenCode session lives in a database
    and the file it lists is an optional sidecar, and Oh My Pi renames a
    transcript away while it rewrites it, holding the session's lease — which
    its adapter reports as activity — while it does.

    Args:
        adapter: The session's adapter, or ``None``. One without
            ``last_activity`` is judged by the file alone.
        session: The session to check.

    Returns:
        True only when the session is known to be gone; a session without
        files, or one its adapter cannot be asked about, counts as present.
    """
    path = session.primary_path
    if path is None or os.path.lexists(path):
        return False
    probe = getattr(adapter, "last_activity", None)
    if probe is None:
        return True
    try:
        return probe(session) is None
    except Exception:  # noqa: BLE001 - a failing probe proves nothing either way
        return False


#: Emoji/label shown for each role in exported Markdown.
_ROLE_HEADING: dict[Role, str] = {
    Role.USER: "🧑 user",
    Role.ASSISTANT: "🤖 assistant",
    Role.TOOL: "🔧 tool",
    Role.SYSTEM: "⚙ system",
    Role.OTHER: "· note",
}


def _slug(text: str, limit: int = 50) -> str:
    """Turn a title into a filesystem-safe slug.

    Args:
        text: Arbitrary text (typically a session title).
        limit: Maximum slug length.

    Returns:
        A lowercase, hyphen-separated slug; ``"session"`` if nothing usable
        remains.
    """
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    if len(text) > limit:
        text = text[:limit].rstrip("-")
    return text or "session"


def _id_slug(session_id: str, limit: int = 8) -> str:
    """Turn a session id into a filesystem-safe filename part.

    Ids are read from the session files themselves (a Pi header's ``id`` can
    hold anything), and a ``/`` in one made the export path point into a
    directory that does not exist — which also aborted an export-before-delete.
    Ordinary ids (UUIDs, ``ses_…``) come out unchanged.

    Args:
        session_id: The session's id.
        limit: Maximum length of the result.

    Returns:
        The id with each run of characters other than ASCII letters, digits,
        ``_`` and ``-`` replaced by ``-``, cut to ``limit`` characters;
        ``"session"`` if nothing usable remains.
    """
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", session_id).strip("-")
    return slug[:limit].strip("-") or "session"


def _fmt_dt(value: datetime | None) -> str:
    """Format a datetime for the export header, or ``"—"`` when missing."""
    return value.strftime("%Y-%m-%d %H:%M") if value else "—"


def messages_to_markdown(
    session: Session,
    messages: list[Message],
    *,
    now: datetime | None = None,
    branches: str | None = None,
) -> str:
    """Render a session and its transcript to a Markdown document.

    Divider markers (``Message.marker``) become a horizontal rule followed by an
    italic line, never a role heading, and are left out of the message count.

    Args:
        session: The session being exported (used for the header).
        messages: The normalized transcript.
        now: Override for the "exported at" timestamp (for reproducible output
            in tests); defaults to the current local time.
        branches: How the export presents a session that branched — the
            :meth:`~sx.model.Transcript.describe` line — recorded as a
            ``Branches`` metadata line. ``None`` (a session that never
            branched) adds nothing.

    Returns:
        A complete Markdown document as a string.
    """
    stamp = now or datetime.now()
    lines: list[str] = []
    title = sanitize_text((session.title or session.session_id).replace("\n", " ").strip())
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- **Harness:** {session.harness}")
    lines.append(f"- **Project:** {sanitize_text(session.project_path or '(unknown)')}")
    lines.append(f"- **Session:** `{sanitize_text(session.session_id)}`")
    lines.append(f"- **Created:** {_fmt_dt(session.created)}")
    lines.append(f"- **Modified:** {_fmt_dt(session.modified)}")
    lines.append(f"- **Messages:** {sum(1 for m in messages if not m.marker)}")
    if branches:
        lines.append(f"- **Branches:** {sanitize_text(branches)}")
    if session.is_orphan:
        lines.append("- **Orphan:** yes (originating project no longer exists)")
    lines.append(f"- **Exported:** {stamp.strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    if not messages:
        lines.append("_(no messages)_")
        return "\n".join(lines) + "\n"

    for message in messages:
        if message.marker:
            # Every block above ends in a blank line, so this rule can never
            # turn the preceding paragraph into a setext heading.
            lines.append("---")
            lines.append("")
            label = marker_label(message)
            if label:
                lines.append(f"_{label}_")
                lines.append("")
            continue
        heading = _ROLE_HEADING.get(message.role, message.role.value)
        when = f" · {message.timestamp.strftime('%Y-%m-%d %H:%M')}" if message.timestamp else ""
        lines.append(f"### {heading}{when}")
        lines.append("")
        if message.tool_summary:
            lines.append(f"> ⚙ **{sanitize_text(message.tool_summary)}**")
            lines.append("")
        if message.thinking:
            thinking = sanitize_text(message.thinking)
            quoted = "\n".join("> " + ln for ln in thinking.splitlines())
            lines.append("> 💭 _thinking_")
            lines.append(quoted)
            lines.append("")
        if message.text:
            lines.append(sanitize_text(message.text))
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def default_export_dir() -> Path:
    """Return the default export directory (``./session-exports``)."""
    return Path.cwd() / "session-exports"


def export_filename(session: Session) -> str:
    """Build the export filename for a session.

    Args:
        session: The session being exported.

    Returns:
        A filename like ``claude-20260530-be0c3352-add-smtp-codes.md``.
    """
    when = session.modified or session.created
    date = when.strftime("%Y%m%d") if when else "nodate"
    return f"{session.harness}-{date}-{_id_slug(session.session_id)}-{_slug(session.title)}.md"


def _load_transcript(adapter, session: Session) -> Transcript:
    """Load ``session`` through ``adapter``, branches included when it has any.

    Args:
        adapter: A harness adapter. Anything exposing only ``load()`` is
            treated as a linear harness.
        session: The session to load.

    Returns:
        The session's :class:`~sx.model.Transcript`.
    """
    loader = getattr(adapter, "load_transcript", None)
    if loader is None:
        return Transcript(current=adapter.load(session))
    return loader(session)


def export_session(
    adapter,
    session: Session,
    dest_dir: Path | None = None,
    *,
    now: datetime | None = None,
    mode: BranchMode = BranchMode.CURRENT,
    transcript: Transcript | None = None,
) -> Path:
    """Load a session's transcript and write it as Markdown.

    For a session that never branched every ``mode`` writes the same document,
    byte for byte, as an export made before branches existed.

    Args:
        adapter: The harness adapter able to load the session.
        session: The session to export.
        dest_dir: Destination directory; defaults to :func:`default_export_dir`.
        now: Override for the export timestamp (tests).
        mode: How to present a session that branched.
        transcript: An already-loaded transcript to export instead of parsing
            the session again.

    Returns:
        The path of the written Markdown file.

    Raises:
        SessionGoneError: The transcript holds no messages because the session
            no longer exists (see :func:`session_gone`). No file is written.
    """
    if transcript is None:
        transcript = _load_transcript(adapter, session)
    if not (transcript.current or transcript.branched) and session_gone(adapter, session):
        raise SessionGoneError(f"the session's file no longer exists: {session.primary_path}")
    dest = dest_dir or default_export_dir()
    dest.mkdir(parents=True, exist_ok=True)
    markdown = messages_to_markdown(
        session,
        transcript.messages(mode),
        now=now,
        branches=transcript.describe(mode),
    )
    out_path = _unique_path(dest / export_filename(session))
    out_path.write_text(markdown, encoding="utf-8")
    return out_path


def _unique_path(path: Path) -> Path:
    """Return ``path``, or the first free ``name-2``, ``name-3``… variant.

    Export filenames are derived from harness, date, a truncated session id, and
    a title slug, so distinct sessions genuinely collide — a session duplicated
    across renamed project folders produces the same name every time. Since this
    function also backs "export before deleting", an overwrite would destroy the
    very archive the user asked for, so collisions are never overwritten.

    Args:
        path: The desired output path.

    Returns:
        A path that does not currently exist.
    """
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    counter = 2
    while True:
        candidate = path.with_name(f"{stem}-{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1
