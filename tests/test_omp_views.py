"""How Oh My Pi sessions read: the empty current branch (T4) and titles (T11).

Every store is synthetic, under ``tmp_path``. The builders and the autouse
sandbox come from :mod:`tests.test_omp_adapter`; the sandbox sets
``LANG=en_GB.UTF-8``, a 24-hour locale.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from sx.adapters.omp import OmpAdapter
from sx.export import messages_to_markdown
from sx.model import BranchMode, Message
from tests import test_omp_adapter as base

sandbox = base.sandbox
old = base.old


def _shown(messages: list[Message]) -> list[str]:
    """Return each message's text, a divider as ``--<text>``."""
    return [f"--{m.text}" if m.marker else m.text for m in messages]


# --- T4: the empty current branch ---------------------------------------------------------


def _reset(old: Path):
    """Write a prompt answered twice, then a reset of the leaf to the root (``null``)."""
    records = [
        base._user("u1", None, "the prompt"),
        base._assistant("a1", "u1", "first answer"),
        base._user("u2", "u1", "retried prompt"),
        base._assistant("a2", "u2", "second answer"),
        base._entry("m1", "a2", "model_usage", activeLeafId=None),
    ]
    base._session(old, records=records)
    adapter = OmpAdapter()
    session = base._only(adapter)
    return session, adapter.load_transcript(session)


def test_an_empty_current_branch_has_every_branch_fork_from_its_start(old):
    """omp's empty leaf is its root: a parentless entry hangs off it, not off nothing."""
    _session, transcript = _reset(old)
    assert transcript.current == [] and transcript.hidden == 4
    assert [(b.label, _shown(b.messages)) for b in transcript.branches] == [
        ("forks from the start of the current branch", ["the prompt", "first answer"]),
        ("forks from other branch 1 after message 1", ["retried prompt", "second answer"]),
    ]


def test_an_empty_current_branch_reads_right_in_every_mode(old):
    """Nothing to divide from at the top; the counts leave the empty branch out."""
    session, transcript = _reset(old)

    assert transcript.messages(BranchMode.CURRENT) == []
    assert _shown(transcript.messages(BranchMode.APPENDIX)) == [
        "the prompt",
        "first answer",
        "--Other branch 2 of 2 — forks from other branch 1 after message 1",
        "retried prompt",
        "second answer",
    ]
    assert _shown(transcript.messages(BranchMode.TREE)) == [
        "the prompt",
        "first answer",
        "--branch — continues after message 1 (other branch)",
        "retried prompt",
        "second answer",
    ]
    assert transcript.describe(BranchMode.CURRENT) == (
        "current branch only — 4 message(s) on 2 other branch(es) not shown"
    )
    assert transcript.describe(BranchMode.APPENDIX) == (
        "current branch (empty), then 2 other branch(es)"
    )
    assert transcript.describe(BranchMode.TREE) == (
        "whole tree in the order it was written — 2 branches"
    )
    markdown = messages_to_markdown(
        session,
        transcript.messages(BranchMode.TREE),
        now=datetime(2026, 5, 30, 13, 0),
        branches=transcript.describe(BranchMode.TREE),
    )
    body = markdown.split("\n---\n", 1)[1]
    assert body.lstrip().startswith("### ")  # no second rule under the header


def test_a_second_root_after_a_reset_continues_from_the_start(old):
    """A conversation begun afresh after a reset joins the root, in the file-order view."""
    records = [
        base._user("u1", None, "first conversation"),
        base._assistant("a1", "u1", "first reply"),
        base._user("u3", None, "a fresh start"),
        base._assistant("a3", "u3", "fresh reply"),
        base._entry("m1", "a3", "model_usage", activeLeafId=None),
    ]
    base._session(old, records=records)
    adapter = OmpAdapter()
    transcript = adapter.load_transcript(base._only(adapter))
    assert _shown(transcript.messages(BranchMode.TREE)) == [
        "first conversation",
        "first reply",
        "--branch — continues from the start (other branch)",
        "a fresh start",
        "fresh reply",
    ]
    assert [b.label for b in transcript.branches] == [
        "forks from the start of the current branch",
        "forks from the start of the current branch",
    ]


# --- T11: titles as omp's session list shows them -------------------------------------------


def _local_stamp(hour: int, minute: int) -> str:
    """Return the UTC timestamp of a local time on 1 May 2026, as omp records it."""
    moment = datetime(2026, 5, 1, hour, minute).astimezone().astimezone(UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _untitled(old: Path, stamp: str) -> str:
    """Return the title of an empty session started at ``stamp``."""
    path = base._bucket(old) / base._name("S1")
    base._write(path, [base._header("S1", str(old), timestamp=stamp)], slot=base._slot())
    return base._title(path)


@pytest.mark.parametrize(
    ("variables", "expected"),
    [
        ({"LANG": "en_US.UTF-8"}, "Untitled · 09:05 PM"),
        ({"LANG": "en"}, "Untitled · 09:05 PM"),
        ({"LANG": "en_GB.UTF-8"}, "Untitled · 21:05"),
        ({"LANG": "de_DE.UTF-8"}, "Untitled · 21:05"),
        ({"LANG": "C"}, "Untitled · 21:05"),
        ({"LANG": "en_GB.UTF-8", "LC_TIME": "en_US.UTF-8"}, "Untitled · 09:05 PM"),
        ({"LANG": "en_US.UTF-8", "LC_TIME": "de_DE.UTF-8"}, "Untitled · 21:05"),
        ({"LC_TIME": "de_DE.UTF-8", "LC_ALL": "en_US.UTF-8"}, "Untitled · 09:05 PM"),
        ({"LC_ALL": "", "LANG": "en_US.UTF-8"}, "Untitled · 09:05 PM"),
    ],
)
def test_the_untitled_placeholder_follows_the_locales_clock(old, monkeypatch, variables, expected):
    """12 hours for US English, 24 otherwise, from ``LC_ALL``, else ``LC_TIME``, else ``LANG``."""
    monkeypatch.delenv("LANG", raising=False)
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    assert _untitled(old, _local_stamp(21, 5)) == expected


def test_the_morning_reads_am(old, monkeypatch):
    """Midnight is 12, the morning keeps its leading zero."""
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    assert _untitled(old, _local_stamp(9, 5)) == "Untitled · 09:05 AM"
    assert _untitled(old, _local_stamp(0, 30)) == "Untitled · 12:30 AM"


def test_only_the_first_4096_bytes_give_a_first_message(old):
    """omp's session list reads no further: a prompt past them leaves the session untitled."""
    path = base._bucket(old) / base._name("S1")
    filler = base._entry("c1", None, "custom", customType="state", data="x" * 5000)
    base._write(
        path,
        [base._header("S1", str(old)), filler, base._user("u1", "c1", "far down")],
        slot=base._slot(),
    )
    assert base._title(path).startswith("Untitled · ")


def test_a_first_prompt_cut_off_at_4096_bytes_still_titles_the_session(old):
    """omp scans the raw text for it, as it does when the line does not parse."""
    path = base._bucket(old) / base._name("S1")
    prompt = "Refactor the parser " + "x" * 6000
    base._write(path, [base._header("S1", str(old)), base._user("u1", None, prompt)], slot=base._slot())
    assert base._title(path).startswith("Refactor the parser xxx")


def test_a_compaction_summary_stands_in_for_a_missing_title(old):
    """The last ``shortSummary`` comes before the first message, as in omp's list."""
    path = base._bucket(old) / base._name("S1")
    records = [
        base._header("S1", str(old)),
        base._user("u1", None, "the first prompt"),
        base._entry("c1", "u1", "compaction", summary="long", shortSummary="Older summary"),
        base._entry("c2", "c1", "compaction", summary="long", shortSummary="Parser rewrite"),
    ]
    base._write(path, records, slot=base._slot("   "))
    assert base._title(path) == "Parser rewrite"
    base._write(path, records, slot=base._slot("Kept title"))
    assert base._title(path) == "Kept title"


def test_a_legacy_headers_empty_title_still_blocks_the_summary(old):
    """omp's ``title ?? summary``: an empty string is a title, which shows as none."""
    path = base._bucket(old) / base._name("S1")
    records = [
        base._header("S1", str(old), title=""),
        base._user("u1", None, "the first prompt"),
        base._entry("c1", "u1", "compaction", summary="long", shortSummary="Parser rewrite"),
    ]
    base._write(path, records, slot=None)
    assert base._title(path) == "the first prompt"
