"""Export a normalized transcript to a Markdown file.

Like :mod:`sx.render`, this module operates only on
:class:`~sx.model.Message`, so it works for every harness. Export is offered
both as a standalone action and as an optional step before deletion, giving a
durable archive of a session that is about to be permanently removed.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from sx.model import Message, Role, Session

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


def _fmt_dt(value: datetime | None) -> str:
    """Format a datetime for the export header, or ``"—"`` when missing."""
    return value.strftime("%Y-%m-%d %H:%M") if value else "—"


def messages_to_markdown(
    session: Session,
    messages: list[Message],
    *,
    now: datetime | None = None,
) -> str:
    """Render a session and its transcript to a Markdown document.

    Args:
        session: The session being exported (used for the header).
        messages: The normalized transcript.
        now: Override for the "exported at" timestamp (for reproducible output
            in tests); defaults to the current local time.

    Returns:
        A complete Markdown document as a string.
    """
    stamp = now or datetime.now()
    lines: list[str] = []
    title = (session.title or session.session_id).replace("\n", " ").strip()
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- **Harness:** {session.harness}")
    lines.append(f"- **Project:** {session.project_path or '(unknown)'}")
    lines.append(f"- **Session:** `{session.session_id}`")
    lines.append(f"- **Created:** {_fmt_dt(session.created)}")
    lines.append(f"- **Modified:** {_fmt_dt(session.modified)}")
    lines.append(f"- **Messages:** {len(messages)}")
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
        heading = _ROLE_HEADING.get(message.role, message.role.value)
        when = f" · {message.timestamp.strftime('%Y-%m-%d %H:%M')}" if message.timestamp else ""
        lines.append(f"### {heading}{when}")
        lines.append("")
        if message.tool_summary:
            lines.append(f"> ⚙ **{message.tool_summary}**")
            lines.append("")
        if message.thinking:
            quoted = "\n".join("> " + ln for ln in message.thinking.splitlines())
            lines.append("> 💭 _thinking_")
            lines.append(quoted)
            lines.append("")
        if message.text:
            lines.append(message.text)
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
    short = session.session_id[:8] or "session"
    return f"{session.harness}-{date}-{short}-{_slug(session.title)}.md"


def export_session(
    adapter,
    session: Session,
    dest_dir: Path | None = None,
    *,
    now: datetime | None = None,
) -> Path:
    """Load a session's transcript and write it as Markdown.

    Args:
        adapter: The harness adapter able to :meth:`load` the session.
        session: The session to export.
        dest_dir: Destination directory; defaults to :func:`default_export_dir`.
        now: Override for the export timestamp (tests).

    Returns:
        The path of the written Markdown file.
    """
    dest = dest_dir or default_export_dir()
    dest.mkdir(parents=True, exist_ok=True)
    messages = adapter.load(session)
    markdown = messages_to_markdown(session, messages, now=now)
    out_path = dest / export_filename(session)
    out_path.write_text(markdown, encoding="utf-8")
    return out_path
