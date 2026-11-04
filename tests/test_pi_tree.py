"""Tests for how Pi's session tree is arranged into branches, on synthetic entries.

Entries that show nothing — labels, thinking-level changes, usage — are
transparent: they never start or split a branch, and never mark a jump in the
file-order view. Building the tree touches each entry a bounded number of times.
"""

from __future__ import annotations

import itertools
import random

import pytest

from sx.adapters import pi as pi_module
from sx.adapters.pi import PiAdapter
from sx.model import BranchMode

TS = "2026-05-01T10:00:00.000Z"


def _entry(entry_id: str, parent: str | None, entry_type: str = "message", **fields) -> dict:
    """Build a tree entry."""
    return {"type": entry_type, "id": entry_id, "parentId": parent, "timestamp": TS, **fields}


def _user(entry_id: str, parent: str | None, text: str) -> dict:
    """Build a user message entry."""
    return _entry(entry_id, parent, message={"role": "user", "content": text, "timestamp": 1})


def _assistant(entry_id: str, parent: str | None, text: str) -> dict:
    """Build an assistant message entry holding one text block."""
    message = {"role": "assistant", "content": [{"type": "text", "text": text}],
               "stopReason": "stop", "timestamp": 2}
    return _entry(entry_id, parent, message=message)


def _quiet(entry_id: str, parent: str | None) -> dict:
    """Build an entry that shows nothing: a thinking-level change."""
    return _entry(entry_id, parent, "thinking_level_change", thinkingLevel="high")


def _view(entries: list[dict], mode: BranchMode) -> list[tuple[bool, str]]:
    """Return ``(is marker, text)`` for each message ``mode`` shows."""
    transcript = PiAdapter()._tree(entries)
    return [(message.marker, message.text) for message in transcript.messages(mode)]


# --- T7: entries that show nothing are transparent ---------------------------------------------


def test_bookkeeping_with_a_stale_parent_draws_no_divider():
    """Written between two turns of one branch, it is not a jump in the tree view."""
    entries = [
        _user("u1", None, "q1"),
        _assistant("a1", "u1", "answer 1"),
        _quiet("t1", "u1"),  # its parent is where the request started
        _user("u2", "a1", "q2"),
        _assistant("a2", "u2", "answer 2"),
        _user("u2b", "a1", "q2 retried"),
        _assistant("a2b", "u2b", "answer 2b"),
    ]

    assert _view(entries, BranchMode.TREE) == [
        (False, "q1"),
        (False, "answer 1"),
        (False, "q2"),
        (False, "answer 2"),
        (True, "branch — continues after message 2 (current branch)"),
        (False, "q2 retried"),
        (False, "answer 2b"),
    ]


def test_bookkeeping_written_before_a_branch_tip_does_not_split_the_branch():
    """One abandoned branch stays one branch in the appendix."""
    entries = [
        _user("u1", None, "q1"),
        _assistant("a1", "u1", "answer 1"),
        _user("u2", "a1", "q2"),
        _assistant("a2", "u2", "answer 2"),
        _entry("l1", "u2", "label", targetId="u2", label="checkpoint"),
        _user("u3", "a2", "q3"),
        _assistant("a3", "u3", "answer 3"),
        _user("u2b", "a1", "q2 retried"),
        _assistant("a2b", "u2b", "answer 2b"),
    ]
    transcript = PiAdapter()._tree(entries)

    assert [(b.label, [m.text for m in b.messages]) for b in transcript.branches] == [
        ("forks from the current branch after message 2", ["q2", "answer 2", "q3", "answer 3"])
    ]
    assert [text for marker, text in _view(entries, BranchMode.APPENDIX) if marker] == [
        "Other branch 1 of 1 — forks from the current branch after message 2"
    ]


def test_a_jump_through_bookkeeping_names_the_message_it_continues_from():
    """A message whose parent shows nothing continues from the nearest one that does."""
    entries = [
        _user("u1", None, "q1"),
        _assistant("a1", "u1", "answer 1"),
        _user("u2", "a1", "q2"),
        _quiet("t1", "a1"),
        _user("u3", "t1", "q2, again"),
    ]

    assert _view(entries, BranchMode.TREE) == [
        (False, "q1"),
        (False, "answer 1"),
        (False, "q2"),
        (True, "branch — continues after message 2 (current branch)"),
        (False, "q2, again"),
    ]


def test_a_branch_from_bookkeeping_before_any_message_continues_from_the_start():
    """Pi's setup entries come first; editing the first prompt branches from them."""
    entries = [
        _quiet("t0", None),
        _user("u1", "t0", "first try"),
        _assistant("a1", "u1", "answer"),
        _user("u1b", "t0", "second try"),
    ]
    transcript = PiAdapter()._tree(entries)

    assert [b.label for b in transcript.branches] == ["forks from the start of the current branch"]
    assert _view(entries, BranchMode.TREE)[2] == (
        True, "branch — continues from the start (current branch)"
    )


# --- T5: the work grows with the file ------------------------------------------------------------


def test_a_long_run_of_bookkeeping_tips_is_not_walked_again_and_again(monkeypatch):
    """Each tip of an empty side chain used to re-walk the whole chain: quadratic."""
    side = []
    for number in range(2000):
        parent = "u1" if number == 0 else f"c{number - 1}"
        side += [_quiet(f"c{number}", parent), _quiet(f"t{number}", f"c{number}")]
    entries = [_user("u1", None, "q"), *side, _user("u2", "u1", "current tail")]
    walked: list[int] = []
    real = pi_module._walk_up

    def counting(start, by_id, stop):
        """Record how many entries each walk covers."""
        segment, fork = real(start, by_id, stop)
        walked.append(len(segment))
        return segment, fork

    monkeypatch.setattr(pi_module, "_walk_up", counting)
    transcript = PiAdapter()._tree(entries)

    assert [m.text for m in transcript.current] == ["q", "current tail"]
    assert not transcript.branched
    assert sum(walked) <= 2 * len(entries)


def test_many_visible_tips_on_a_shared_chain_are_walked_once(monkeypatch):
    """Visible tips hanging off one long invisible chain: each link is walked once."""
    entries = [_user("u1", None, "q")]
    for number in range(2000):
        parent = "u1" if number == 0 else f"c{number - 1}"
        tip = _user(f"t{number}", f"c{number}", f"tip {number}")
        entries += [_quiet(f"c{number}", parent), tip]
    entries.append(_user("u2", "u1", "current tail"))
    walked: list[int] = []
    real = pi_module._walk_up

    def counting(start, by_id, stop):
        """Record how many entries each walk covers."""
        segment, fork = real(start, by_id, stop)
        walked.append(len(segment))
        return segment, fork

    monkeypatch.setattr(pi_module, "_walk_up", counting)
    transcript = PiAdapter()._tree(entries)

    assert len(transcript.branches) == 2000 and transcript.hidden == 2000
    assert sum(walked) <= 2 * len(entries)


# --- the invariants, on random trees ----------------------------------------------------------


def _random_entries(rng: random.Random, count: int) -> list[dict]:
    """Build ``count`` entries with arbitrary parents: forks, cycles, gaps, repeated ids."""
    ids = [f"e{index}" for index in range(count)]
    entries: list[dict] = []
    for index, entry_id in enumerate(ids):
        roll = rng.random()
        if index == 0 or roll < 0.05:
            parent = None
        elif roll < 0.10:
            parent = f"missing{index}"
        elif roll < 0.15:
            parent = rng.choice(ids)  # anything: forward references and cycles included
        elif roll < 0.60:
            parent = ids[index - 1]
        else:
            parent = ids[rng.randrange(0, index)]
        if index and rng.random() < 0.05:
            entry_id = rng.choice(ids[:index])  # a repeated id
        kind = rng.choice(["message", "message", "message", "label", "quiet"])
        if kind == "message":
            entries.append(_user(entry_id, parent, f"text-{index}"))
        elif kind == "label":
            entries.append(_entry(entry_id, parent, "label", targetId="e0", label="x"))
        else:
            entries.append(_quiet(entry_id, parent))
    return entries


@pytest.mark.parametrize("seed", range(400))
def test_every_message_is_shown_exactly_once_in_every_view(seed):
    """Tree view in file order, appendix and hidden count agree, whatever the shape."""
    rng = random.Random(seed)
    entries = _random_entries(rng, rng.randrange(1, 40))
    adapter = PiAdapter()
    last = {entry["id"]: entry for entry in entries}
    ordered = [entry for entry in entries if last[entry["id"]] is entry]
    expected = [m.text for m in (adapter.entry_message(e) for e in ordered) if m is not None]

    transcript = adapter._tree(entries)

    if not transcript.branched:
        assert sorted(m.text for m in transcript.current) == sorted(expected)
        return
    tree = transcript.messages(BranchMode.TREE)
    appendix = [m.text for m in transcript.messages(BranchMode.APPENDIX) if not m.marker]
    assert [m.text for m in tree if not m.marker] == expected
    assert sorted(appendix) == sorted(expected)
    assert transcript.hidden == len(appendix) - len(transcript.current)
    assert not tree[0].marker and not tree[-1].marker
    assert not any(a.marker and b.marker for a, b in itertools.pairwise(tree))
    assert all(branch.messages for branch in transcript.branches)
