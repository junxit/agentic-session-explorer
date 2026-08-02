"""Render a normalized transcript (``list[Message]``) for display.

This module is intentionally harness-agnostic: it operates only on
:class:`~sx.model.Message`, so the same rendering is reused everywhere a
transcript is shown. The Markdown renderer (added in the export milestone) will
live here too, beside :func:`messages_to_text`.
"""

from __future__ import annotations

from rich.text import Text

from sx.model import Message, Role
from sx.util import sanitize_text

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


def messages_to_text(messages: list[Message]) -> Text:
    """Render a transcript to a single styled :class:`rich.text.Text`.

    Building one ``Text`` (rather than many widgets) keeps rendering fast for
    large transcripts and is safe against markup injection from chat content,
    since the content is added as literal spans rather than parsed markup.

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
