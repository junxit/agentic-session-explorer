"""Render a normalized transcript (``list[Message]``) for display.

This module is intentionally harness-agnostic: it operates only on
:class:`~sx.model.Message` and :class:`~sx.model.Transcript`, so the same
rendering is reused everywhere a transcript is shown. The Markdown counterpart
lives in :mod:`sx.export`.
"""

from __future__ import annotations

from rich.text import Text

from sx.model import BranchMode, Message, Role, Transcript
from sx.util import sanitize_text

#: Rich style for the divider lines ``sx`` inserts where a branch starts.
MARKER_STYLE = "bold yellow"

#: Rich style applied to each role's header line.
ROLE_STYLE: dict[Role, str] = {
    Role.USER: "bold cyan",
    Role.ASSISTANT: "bold green",
    Role.TOOL: "yellow",
    Role.SYSTEM: "bold magenta",
    Role.OTHER: "dim",
}

#: Human-friendly label per role.
ROLE_LABEL: dict[Role, str] = {
    Role.USER: "user",
    Role.ASSISTANT: "assistant",
    Role.TOOL: "tool",
    Role.SYSTEM: "system",
    Role.OTHER: "·",
}


def _prefix_lines(text: str, prefix: str) -> str:
    """Prefix every line of ``text`` with ``prefix``.

    Args:
        text: The (possibly multi-line) text to indent.
        prefix: String prepended to each line.

    Returns:
        The prefixed text.
    """
    return "\n".join(prefix + line for line in text.splitlines())


def marker_label(message: Message) -> str:
    """Return a divider marker's text as one sanitized line.

    Shared with the Markdown exporter so a divider reads the same in the viewer
    and in an export.

    Args:
        message: A message with ``marker`` set.

    Returns:
        The marker text, newlines folded into spaces and control characters
        neutralized.
    """
    return sanitize_text(message.text.replace("\n", " ").strip())


def messages_to_text(messages: list[Message]) -> Text:
    """Render a transcript to a single styled :class:`rich.text.Text`.

    Building one ``Text`` (rather than many widgets) keeps rendering fast for
    large transcripts and is safe against markup injection from chat content,
    since the content is added as literal spans rather than parsed markup.

    A message with ``marker`` set is a divider ``sx`` inserted itself (where a
    branch starts, for instance), so it renders as a divider line rather than
    as a role turn.

    Args:
        messages: The normalized transcript.

    Returns:
        A styled ``Text`` ready to write into a ``RichLog`` or ``Static``.
    """
    out = Text()
    if not messages:
        out.append("(no messages)", style="dim italic")
        return out

    for index, message in enumerate(messages):
        if index:
            out.append("\n")

        if message.marker:
            out.append(f"── {marker_label(message)} ──", style=MARKER_STYLE)
            out.append("\n")
            continue

        style = ROLE_STYLE.get(message.role, "white")
        out.append("▌ ", style=style)
        out.append(ROLE_LABEL.get(message.role, message.role.value), style=style)
        if message.timestamp:
            out.append("  " + message.timestamp.strftime("%Y-%m-%d %H:%M"), style="dim")
        if message.tool_summary:
            out.append("  ⚙ " + sanitize_text(message.tool_summary), style="yellow")
        out.append("\n")

        if message.thinking:
            out.append(
                _prefix_lines(sanitize_text(message.thinking), "  💭 "),
                style="dim italic",
            )
            out.append("\n")

        if message.text:
            out.append(sanitize_text(message.text))
            out.append("\n")

    return out


def transcript_to_text(
    transcript: Transcript, mode: BranchMode, *, hint: str | None = None
) -> Text:
    """Render a transcript in ``mode``, headed by what the mode shows.

    A session that never branched renders exactly as :func:`messages_to_text`
    would render its messages: every mode presents the same thing, so there is
    nothing to describe.

    Args:
        transcript: The loaded transcript.
        mode: How to present the branches the session does not end on.
        hint: Optional dim text appended to the header line, such as the key
            that changes the view.

    Returns:
        A styled ``Text`` ready to write into a ``RichLog`` or ``Static``.
    """
    body = messages_to_text(transcript.messages(mode))
    description = transcript.describe(mode)
    if description is None:
        return body
    out = Text()
    out.append(f"⑂ {sanitize_text(description)}", style=MARKER_STYLE)
    if hint:
        out.append(f"  {sanitize_text(hint)}", style="dim")
    out.append("\n\n")
    out.append_text(body)
    return out
