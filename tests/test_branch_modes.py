"""Rendering and exporting a session that branched, in every ``BranchMode``.

A hand-built :class:`~sx.model.Transcript` stands in for a tree-shaped harness
(Pi, Oh My Pi): one prompt whose answer was retried, so the abandoned answer
sits on a side branch. Nothing here reads a real session store; files are only
ever written under ``tmp_path``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from sx.adapters.base import HarnessAdapter
from sx.export import export_session, messages_to_markdown
from sx.model import Branch, BranchMode, Capability, Message, Role, Session, Transcript
from sx.render import MARKER_STYLE, messages_to_text, transcript_to_text

NOW = datetime(2026, 5, 30, 13, 0)


def _session() -> Session:
    """A session with deterministic metadata for export tests."""
    return Session(
        harness="tmp",
        session_id="abcd1234",
        project_path="/some/proj",
        title="Branchy",
        modified=datetime(2026, 5, 30, 12, 0),
    )


def _branched() -> Transcript:
    """A prompt answered twice: the first answer was abandoned by a retry."""
    ask = Message(Role.USER, text="the prompt", timestamp=datetime(2026, 5, 30, 12, 0))
    abandoned = Message(Role.ASSISTANT, text="abandoned answer")
    kept = Message(Role.ASSISTANT, text="kept answer")
    return Transcript(
        current=[ask, kept],
        branches=[Branch(label="retried after: the prompt", messages=[abandoned])],
        tree=[
            ask,
            abandoned,
            Message(Role.OTHER, text="Branch 2 of 2 — retried after: the prompt", marker=True),
            kept,
        ],
        hidden=1,
    )


class _BranchingAdapter(HarnessAdapter):
    """Returns the branched transcript; its linear ``load`` must never be used."""

    name = "tmp"
    display = "Tmp"
    capabilities = Capability.BROWSE
    supports_branches = True

    def __init__(self, transcript: Transcript | None = None) -> None:
        """Remember the transcript to serve and count how often it is parsed."""
        self._transcript = transcript or _branched()
        self.calls = 0

    def store_roots(self) -> list[Path]:
        """No store is needed."""
        return []

    def discover(self):
        """Discovery is unused here."""
        return iter(())

    def load(self, session: Session) -> list[Message]:
        """A branching adapter is exported through ``load_transcript`` only."""
        raise AssertionError("load() must not be called for a branching adapter")

    def load_transcript(self, session: Session) -> Transcript:
        """Return the branched transcript."""
        self.calls += 1
        return self._transcript


# --- the viewer ------------------------------------------------------------


def test_current_mode_shows_the_current_branch_under_a_count():
    """CURRENT shows only the branch the session ended on, and says what it hides."""
    plain = transcript_to_text(_branched(), BranchMode.CURRENT).plain
    lines = plain.splitlines()
    assert lines[0] == "⑂ current branch only — 1 message(s) on 1 other branch(es) not shown"
    assert lines[1] == ""
    assert "the prompt" in plain and "kept answer" in plain
    assert "abandoned answer" not in plain


def test_appendix_mode_follows_the_current_branch_with_the_others():
    """APPENDIX adds each other branch after a divider naming where it forks."""
    plain = transcript_to_text(_branched(), BranchMode.APPENDIX).plain
    divider = "── Other branch 1 of 1 — retried after: the prompt ──"
    assert plain.splitlines()[0] == "⑂ current branch, then 1 other branch(es)"
    assert divider in plain.splitlines()
    assert plain.index("kept answer") < plain.index(divider) < plain.index("abandoned answer")


def test_tree_mode_shows_every_entry_in_file_order():
    """TREE is the file as written, with a divider where it jumps branches."""
    plain = transcript_to_text(_branched(), BranchMode.TREE).plain
    divider = "── Branch 2 of 2 — retried after: the prompt ──"
    assert plain.splitlines()[0] == "⑂ whole tree in the order it was written — 2 branches"
    order = [plain.index(s) for s in ("the prompt", "abandoned answer", divider, "kept answer")]
    assert order == sorted(order)


@pytest.mark.parametrize("mode", list(BranchMode))
def test_markers_render_as_dividers_never_as_role_turns(mode):
    """A marker is sx's own divider, so it never gets a ``▌ role`` header."""
    text = messages_to_text(_branched().messages(mode))
    # Role.OTHER's header would read "▌ ·"; the transcript has no real OTHER turn.
    assert "▌ ·" not in text.plain
    for line in text.plain.splitlines():
        if line.startswith("──"):
            start = text.plain.index(line)
            assert any(
                span.start <= start and span.end >= start + len(line)
                and str(span.style) == MARKER_STYLE
                for span in text.spans
            ), f"divider {line!r} is not styled as a marker"


def test_the_hint_follows_the_header_dimmed():
    """The viewer appends the key that changes the view, styled dim."""
    text = transcript_to_text(_branched(), BranchMode.CURRENT, hint="t: change view")
    first = text.plain.splitlines()[0]
    assert first.endswith("  t: change view")
    start = text.plain.index("t: change view")
    assert any(
        span.start <= start and span.end >= start + len("t: change view")
        and str(span.style) == "dim"
        for span in text.spans
    )


@pytest.mark.parametrize("mode", list(BranchMode))
def test_an_unbranched_transcript_renders_exactly_as_before(mode):
    """No header, no hint: a linear session looks the way it always did."""
    messages = [Message(Role.USER, text="hi"), Message(Role.ASSISTANT, text="hello")]
    rendered = transcript_to_text(Transcript(current=messages), mode, hint="t: change view")
    assert rendered == messages_to_text(messages)


# --- the export ------------------------------------------------------------


def test_markdown_markers_are_rules_and_italics_not_headings():
    """A divider becomes ``---`` and an italic line — never a ``###`` role heading."""
    transcript = _branched()
    md = messages_to_markdown(
        _session(),
        transcript.messages(BranchMode.APPENDIX),
        now=NOW,
        branches=transcript.describe(BranchMode.APPENDIX),
    )
    assert "\n\n---\n\n_Other branch 1 of 1 — retried after: the prompt_\n\n" in md
    assert "### · note" not in md
    assert md.index("kept answer") < md.index("_Other branch 1") < md.index("abandoned answer")


@pytest.mark.parametrize(
    ("mode", "count"),
    [(BranchMode.CURRENT, 2), (BranchMode.APPENDIX, 3), (BranchMode.TREE, 3)],
)
def test_the_message_count_leaves_markers_out(mode, count):
    """``Messages:`` counts what the harness recorded, not sx's dividers."""
    transcript = _branched()
    md = messages_to_markdown(
        _session(), transcript.messages(mode), now=NOW, branches=transcript.describe(mode)
    )
    assert f"\n- **Messages:** {count}\n" in md


@pytest.mark.parametrize("mode", list(BranchMode))
def test_the_branches_line_follows_the_message_count(mode):
    """The export records which view it holds, right under the count."""
    transcript = _branched()
    md = messages_to_markdown(
        _session(), transcript.messages(mode), now=NOW, branches=transcript.describe(mode)
    )
    lines = md.splitlines()
    index = lines.index(f"- **Branches:** {transcript.describe(mode)}")
    assert lines[index - 1].startswith("- **Messages:**")


#: ``messages_to_markdown`` output for :func:`_golden_session`, captured from the
#: implementation before branches existed. An unbranched export must not move.
GOLDEN = (
    "# Golden [session] line two\n\n"
    "- **Harness:** tmp\n"
    "- **Project:** /some/proj\n"
    "- **Session:** `abcd1234`\n"
    "- **Created:** 2026-05-30 11:00\n"
    "- **Modified:** 2026-05-30 12:00\n"
    "- **Messages:** 5\n"
    "- **Orphan:** yes (originating project no longer exists)\n"
    "- **Exported:** 2026-05-30 13:00\n\n"
    "---\n\n"
    "### 🧑 user · 2026-05-30 12:00\n\nhello *world*\n\n"
    "### 🤖 assistant · 2026-05-30 12:01\n\n"
    "> ⚙ **Bash(ls)**\n\n"
    "> 💭 _thinking_\n> ponder\n> more\n\n"
    "hi\nthere\n\n"
    "### 🔧 tool\n\n> ⚙ **tool: Read(x)**\n\n"
    "### ⚙ system\n\nsys\n\n"
    "### · note\n\nother note\n"
)


def _golden_session() -> Session:
    """The session :data:`GOLDEN` was captured from."""
    return Session(
        harness="tmp",
        session_id="abcd1234",
        project_path="/some/proj",
        title="Golden [session]\nline two",
        modified=datetime(2026, 5, 30, 12, 0),
        created=datetime(2026, 5, 30, 11, 0),
        is_orphan=True,
    )


def _golden_messages() -> list[Message]:
    """The transcript :data:`GOLDEN` was captured from: every kind of field."""
    return [
        Message(Role.USER, text="hello *world*", timestamp=datetime(2026, 5, 30, 12, 0)),
        Message(
            Role.ASSISTANT,
            text="hi\nthere",
            thinking="ponder\nmore",
            tool_summary="Bash(ls)",
            timestamp=datetime(2026, 5, 30, 12, 1),
        ),
        Message(Role.TOOL, tool_summary="tool: Read(x)"),
        Message(Role.SYSTEM, text="sys"),
        Message(Role.OTHER, text="other note"),
    ]


class _LoadOnly:
    """An adapter-like object from before ``load_transcript`` existed."""

    def load(self, session: Session) -> list[Message]:
        """Return the golden transcript."""
        return _golden_messages()


class _Linear(HarnessAdapter):
    """A linear harness: ``load_transcript`` is the inherited default."""

    name = "tmp"
    display = "Tmp"
    capabilities = Capability.BROWSE

    def store_roots(self) -> list[Path]:
        """No store is needed."""
        return []

    def discover(self):
        """Discovery is unused here."""
        return iter(())

    def load(self, session: Session) -> list[Message]:
        """Return the golden transcript."""
        return _golden_messages()


def test_unbranched_markdown_is_byte_identical_to_before():
    """No ``branches`` and no markers: the document is exactly what it was."""
    assert messages_to_markdown(_golden_session(), _golden_messages(), now=NOW) == GOLDEN


@pytest.mark.parametrize("adapter", [_LoadOnly(), _Linear()], ids=["load-only", "linear"])
@pytest.mark.parametrize("mode", list(BranchMode))
def test_unbranched_export_files_are_byte_identical_in_every_mode(tmp_path, adapter, mode):
    """Whatever mode is asked for, a linear session exports the same bytes."""
    out = export_session(adapter, _golden_session(), tmp_path, now=NOW, mode=mode)
    assert out.read_bytes() == GOLDEN.encode("utf-8")


def test_export_session_uses_a_supplied_transcript_without_parsing(tmp_path):
    """An already-loaded transcript is exported as is; the session is not re-read."""
    adapter = _BranchingAdapter()
    out = export_session(
        adapter, _session(), tmp_path, now=NOW, mode=BranchMode.TREE, transcript=_branched()
    )
    assert adapter.calls == 0
    md = out.read_text(encoding="utf-8")
    assert "- **Branches:** whole tree in the order it was written — 2 branches" in md
    assert "abandoned answer" in md


@pytest.mark.parametrize("mode", list(BranchMode))
def test_export_session_loads_the_transcript_and_honors_the_mode(tmp_path, mode):
    """Without a transcript, the adapter's ``load_transcript`` supplies one."""
    adapter = _BranchingAdapter()
    out = export_session(adapter, _session(), tmp_path, now=NOW, mode=mode)
    assert adapter.calls == 1
    md = out.read_text(encoding="utf-8")
    assert f"- **Branches:** {_branched().describe(mode)}" in md
    assert ("abandoned answer" in md) is (mode is not BranchMode.CURRENT)


def test_export_defaults_to_the_current_branch(tmp_path):
    """Callers that predate branches get the current branch, and are told so."""
    out = export_session(_BranchingAdapter(), _session(), tmp_path, now=NOW)
    md = out.read_text(encoding="utf-8")
    assert "- **Branches:** current branch only — 1 message(s) on 1 other branch(es)" in md
    assert "abandoned answer" not in md


# --- untrusted text --------------------------------------------------------


def _hostile() -> Transcript:
    """A branched transcript whose branch label carries terminal escapes."""
    label = "evil\x1b]0;pwned\x07 [bold]not markup[/bold]\nsecond line"
    transcript = _branched()
    transcript.branches[0].label = label
    return transcript


def test_marker_text_is_sanitized_in_the_viewer():
    """A divider cannot drive the terminal, parse as markup, or span lines."""
    transcript = _hostile()
    plain = transcript_to_text(transcript, BranchMode.APPENDIX, hint="t\x1b[31m").plain
    assert "\x1b" not in plain and "\x07" not in plain
    assert "␛" in plain and "␇" in plain
    divider = next(line for line in plain.splitlines() if line.startswith("──"))
    assert "[bold]not markup[/bold] second line" in divider
    assert divider.endswith(" ──")


def test_marker_text_is_sanitized_in_the_export():
    """Escapes in a divider, or in the Branches line, never reach the archive."""
    transcript = _hostile()
    md = messages_to_markdown(
        _session(),
        transcript.messages(BranchMode.APPENDIX),
        now=NOW,
        branches="current\x1b[2J branch",
    )
    assert "\x1b" not in md and "\x07" not in md
    assert "- **Branches:** current␛[2J branch" in md
    assert "_Other branch 1 of 1 — evil␛]0;pwned␇ [bold]not markup[/bold] second line_" in md


# --- a session whose current branch is empty -------------------------------


def _reset_to_root() -> Transcript:
    """A session that went back to its start: nothing is current, two runs elsewhere.

    This is what Oh My Pi leaves when its leaf is reset to the root: the tree's
    first run starts on another branch, so the tree as written opens with a
    divider, and an empty branch with nothing to show is passed along too.
    """
    ask = Message(Role.USER, text="the prompt")
    first = Message(Role.ASSISTANT, text="first answer")
    retry = Message(Role.USER, text="retried prompt")
    second = Message(Role.ASSISTANT, text="second answer")
    return Transcript(
        current=[],
        branches=[
            Branch(label="a separate root", messages=[ask, first]),
            Branch(label="nothing to show", messages=[]),
            Branch(label="forks from other branch 1 after message 1", messages=[retry, second]),
        ],
        tree=[
            Message(Role.OTHER, text="branch — a separate root (other branch)", marker=True),
            ask,
            first,
            Message(Role.OTHER, text="branch — continues after message 1", marker=True),
            Message(Role.OTHER, text="branch — continues after message 1 (other branch)", marker=True),
            retry,
            second,
            Message(Role.OTHER, text="branch — a trailing jump", marker=True),
        ],
        hidden=4,
    )


def test_the_tree_never_opens_or_closes_with_a_divider():
    """A divider with nothing before it divides nothing; a run of them keeps the last."""
    messages = _reset_to_root().messages(BranchMode.TREE)
    assert not messages[0].marker and not messages[-1].marker
    assert [m.text for m in messages] == [
        "the prompt",
        "first answer",
        "branch — continues after message 1 (other branch)",
        "retried prompt",
        "second answer",
    ]


def test_the_appendix_of_an_empty_branch_does_not_double_the_rule():
    """The header's rule is not followed by a second one for the first branch."""
    transcript = _reset_to_root()
    md = messages_to_markdown(
        _session(),
        transcript.messages(BranchMode.APPENDIX),
        now=NOW,
        branches=transcript.describe(BranchMode.APPENDIX),
    )
    assert "---\n\n---" not in md
    assert md.index("first answer") < md.index("_Other branch 2 of 2") < md.index("second answer")


def test_branch_counts_include_only_branches_with_messages():
    """An empty current branch, or an empty side branch, is not counted."""
    transcript = _reset_to_root()
    assert transcript.describe(BranchMode.TREE) == "whole tree in the order it was written — 2 branches"
    assert transcript.describe(BranchMode.APPENDIX) == "current branch (empty), then 2 other branch(es)"
    assert transcript.describe(BranchMode.CURRENT) == (
        "current branch only — 4 message(s) on 2 other branch(es) not shown"
    )
    assert "Other branch 3" not in transcript_to_text(transcript, BranchMode.APPENDIX).plain


def test_a_transcript_whose_only_side_branch_is_empty_never_branched():
    """Nothing lies off the current branch, so every mode shows the same thing."""
    ask = Message(Role.USER, text="the prompt")
    transcript = Transcript(current=[ask], branches=[Branch(label="empty", messages=[])])
    assert not transcript.branched and transcript.describe(BranchMode.TREE) is None
    assert transcript.messages(BranchMode.APPENDIX) == [ask]
