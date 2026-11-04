"""Pi fork links: on the way out of a move cut short, and on a re-run.

A fork's header names the transcript it was forked from (``parentSession``).
A move cut short re-points the links of the units that finished before the
interruption goes on, and hands what it did to the move service on the
exception (``partial_result``), so the op-log records it. A re-run repairs a
link an earlier run left dangling, even when it has nothing left to relocate,
under the same guards as any re-pointing — and never "repairs" a link that
still resolves, or one outside the store's folders, or a fork reached through
a symbolic link.

Every store is synthetic, under ``tmp_path`` (the builders and the autouse
``home`` fixture come from :mod:`tests.test_pi_fixes`); ``pi`` is never run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import sx.adapters.base as base_module
from sx.adapters import pi as pi_module
from sx.adapters.pi import PiAdapter, encode_session_folder
from sx.model import MoveResult
from tests.test_pi_fixes import (
    _age_tree,
    _folder,
    _forked,
    _header,
    _link,
    _name,
    _service,
    _session,
    _user,
    _write,
    home,
)

__all__ = ["home"]  # the autouse sandbox

OLD, NEW = Path("/x/old"), Path("/x/new")


def _stop_rename(
    monkeypatch: pytest.MonkeyPatch, stop: type[BaseException], nth: int = 2
) -> None:
    """Make the ``nth`` rename of a move raise ``stop``: Ctrl-C, or a failure."""
    real = base_module.rename_no_clobber
    calls = {"n": 0}

    def rename(src: Path, dst: Path) -> str | None:
        calls["n"] += 1
        if calls["n"] == nth:
            raise stop(f"on unit {nth}")
        return real(src, dst)

    monkeypatch.setattr(base_module, "rename_no_clobber", rename)


def _set_header(path: Path, **fields) -> None:
    """Rewrite fields of a transcript's header line."""
    lines = path.read_bytes().splitlines(keepends=True)
    header = json.dumps({**json.loads(lines[0]), **fields}, separators=(",", ":"))
    path.write_bytes(header.encode() + b"\n" + b"".join(lines[1:]))


def _plan(adapter: PiAdapter):
    """Plan the move of /x/old to /x/new as the move service does."""
    return _service_for(adapter).plan(OLD, NEW)["pi"]


def _service_for(adapter: PiAdapter):
    """Return a move service around ``adapter``, logging into the sandbox."""
    from sx.service import MoveService

    return MoveService({"pi": adapter}, log_path=pi_module.home() / "ops.log")


# --- 3: what a move cut short re-points goes with the exception ------------------------


@pytest.mark.parametrize("stop", [KeyboardInterrupt, RuntimeError], ids=["ctrl-c", "failure"])
def test_a_move_cut_short_hands_the_fork_links_it_repointed_to_the_op_log(home, monkeypatch, stop):
    """F1 is re-pointed on the way out; the exception says so, and says nothing else."""
    s1, f1 = _forked(home)
    _stop_rename(monkeypatch, stop)
    adapter = PiAdapter()
    plan = _plan(adapter)

    with pytest.raises(stop) as caught:
        adapter.move(plan)

    handed = caught.value.partial_result
    assert _link(f1) == str(_folder(NEW) / s1.name)
    assert isinstance(handed, MoveResult) and not handed.dry_run
    assert handed.rewritten == [f1] and handed.fields_updated == 1
    assert not handed.moved and not handed.relocated  # the units' own records hold those


def test_a_second_ctrl_c_while_repointing_still_hands_over_what_was_done(home, monkeypatch):
    """F1 was re-pointed, F2 was not: the second interruption carries F1."""
    s1, f1 = _forked(home)
    f2 = _session("/elsewhere/two", "F2", parentSession=str(s1))
    _age_tree(PiAdapter().sessions_dir(), 3600)
    _stop_rename(monkeypatch, KeyboardInterrupt)
    real = pi_module.rewrite_jsonl
    calls = {"n": 0}

    def rewrite(path, transform, *, dry_run=False):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt("again")
        return real(path, transform, dry_run=dry_run)

    monkeypatch.setattr(pi_module, "rewrite_jsonl", rewrite)
    adapter = PiAdapter()
    plan = _plan(adapter)

    with pytest.raises(KeyboardInterrupt, match="again") as caught:
        adapter.move(plan)

    done = [path for path in (f1, f2) if _link(path) == str(_folder(NEW) / s1.name)]
    assert len(done) == 1
    assert caught.value.partial_result.rewritten == done


def test_a_fork_a_finished_unit_moved_is_re_pointed_on_the_way_out_and_listed_once(
    home, monkeypatch
):
    """F1 moved with a unit that finished, so it was written just now: it is re-pointed anyway.

    Ctrl-C lands on the third unit, after S1's and F1's. F1 is recent only
    because its unit rewrote it; taken for written by someone else, it would
    be refused on the way out, its link left naming S1's old path. Its rewrite
    is listed once in the final record.
    """
    s1 = _session(OLD, "S1")
    f1 = _session(OLD / "sub", "F1", parentSession=str(s1))
    _session(OLD / "sub2", "S2")
    _age_tree(PiAdapter().sessions_dir(), 3600)
    _stop_rename(monkeypatch, KeyboardInterrupt, nth=3)
    service = _service(home)

    with pytest.raises(KeyboardInterrupt):
        service.move(service.plan(OLD, NEW))

    moved = _folder(NEW / "sub") / f1.name
    assert moved.exists() and _link(moved) == str(_folder(NEW) / s1.name)
    final = json.loads((home / "ops.log").read_text().splitlines()[-1])["harnesses"]["pi"]
    assert str(f1) in final["rewritten"]
    assert len(final["rewritten"]) == len(set(final["rewritten"]))


# --- 5: a re-run repairs dangling links even with nothing to relocate ------------------------


def _left_dangling(home: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    """Move S1 with a run killed before it re-pointed F1; S2 stays, in a version sx does not know.

    Returns:
        S1 where it is now, F1, and S2.
    """
    s1, f1 = _forked(home)
    s2 = _session(OLD, "S2", version=4)
    _age_tree(PiAdapter().sessions_dir(), 3600)
    with monkeypatch.context() as killed:  # as an older sx, or kill -9, would leave it
        killed.setattr(PiAdapter, "_repoint_fork_links", lambda *a, **k: None)
        service = _service(home)
        service.move(service.plan(OLD, NEW))
    moved = _folder(NEW) / s1.name
    assert moved.exists() and s2.exists() and _link(f1) == str(s1)  # the premise: dangling
    return moved, f1, s2


def test_a_rerun_with_nothing_left_to_relocate_still_repairs_a_dangling_link(home, monkeypatch):
    """S2 is refused again and nothing moves, but F1 is pointed at where S1 went."""
    moved, f1, s2 = _left_dangling(home, monkeypatch)
    service = _service(home)

    dry = service.move(service.plan(OLD, NEW), dry_run=True)["pi"]
    assert f1 in dry.rewritten and _link(f1) != str(moved)
    result = service.move(service.plan(OLD, NEW))["pi"]

    assert not result.moved and "version 4" in result.refused[s2]
    assert f1 in result.rewritten and _link(f1) == str(moved)


@pytest.mark.parametrize("why", ["recent", "unknown-version"])
def test_a_rerun_repairs_a_dangling_link_only_under_the_usual_guards(home, monkeypatch, why):
    """A fork written in the last 90 seconds, or one sx may not rewrite, keeps its link."""
    _, f1, _ = _left_dangling(home, monkeypatch)
    if why == "unknown-version":
        _set_header(f1, version=4)
        _age_tree(f1.parent, 3600)
    else:
        os.utime(f1)
    before = f1.read_bytes()
    service = _service(home)

    result = service.move(service.plan(OLD, NEW))["pi"]

    expected = "written in the last 90 seconds" if why == "recent" else "not safe to rewrite"
    assert expected in result.refused[f1] and f1.read_bytes() == before


def test_a_rerun_leaves_a_dangling_link_that_changed_after_it_was_read_as_it_now_is(
    home, monkeypatch
):
    """The user re-forks F1 between sx reading its header and rewriting it: F1 keeps the new link."""
    _, f1, _ = _left_dangling(home, monkeypatch)
    dangling = _link(f1)
    real_read = PiAdapter.read_header
    real_repoint = PiAdapter._repoint_fork_links

    def stale_then_changed(self, path):
        """Return the header as read, then change the link on disk."""
        header = real_read(self, path)
        if path == f1 and header and header.get("parentSession") == dangling:
            _set_header(f1, parentSession="/somewhere/else.jsonl")
            _age_tree(f1.parent, 3600)
        return header

    def repoint(self, *args, **kwargs):
        """Re-point links while every header read is stale."""
        monkeypatch.setattr(PiAdapter, "read_header", stale_then_changed)
        try:
            return real_repoint(self, *args, **kwargs)
        finally:
            monkeypatch.setattr(PiAdapter, "read_header", real_read)

    monkeypatch.setattr(PiAdapter, "_repoint_fork_links", repoint)
    service = _service(home)

    result = service.move(service.plan(OLD, NEW))["pi"]

    assert _link(f1) == "/somewhere/else.jsonl" and f1 not in result.rewritten


def test_a_link_that_still_resolves_is_never_taken_for_a_dangling_one(home):
    """S1 is still where F1 says, though a copy of it already sits in the new folder."""
    s1, f1 = _forked(home)
    copy = _write(_folder(NEW) / s1.name, [_header("S1", str(NEW)), _user("u1", None, "hello")])
    _age_tree(PiAdapter().sessions_dir(), 3600)
    service = _service(home)

    result = service.move(service.plan(OLD, NEW))["pi"]

    assert result.moved and "already exists" in result.refused[s1]  # S2 moved; S1 could not
    assert s1.exists() and copy.exists() and _link(f1) == str(s1)
    assert f1 not in result.rewritten


def test_a_dangling_link_outside_the_stores_folders_is_left_alone(home):
    """F1 names a transcript in a backup of the store: same folder and file names, another place."""
    s1, f1 = _forked(home)
    elsewhere = home / "backup" / "sessions" / s1.parent.name / s1.name  # never existed
    _set_header(f1, parentSession=str(elsewhere))
    s1.write_bytes(s1.read_bytes().replace(b'"cwd":"/x/old"', b'"cwd":"/x/new"', 1))
    _folder(NEW).mkdir(parents=True)
    s1.rename(_folder(NEW) / s1.name)  # where a killed run left it
    _age_tree(PiAdapter().sessions_dir(), 3600)
    service = _service(home)

    result = service.move(service.plan(OLD, NEW))["pi"]

    assert result.moved  # S2
    assert _link(f1) == str(elsewhere) and f1 not in result.rewritten


def test_a_fork_reached_through_a_linked_folder_keeps_its_link(home, tmp_path):
    """F1 lives in a directory outside the store that a project folder links to: never rewritten."""
    s1 = _session(OLD, "S1")
    outside = tmp_path / "outside"
    records = [_header("F1", "/elsewhere", parentSession=str(s1)), _user("u1", None, "hi")]
    f1 = _write(outside / _name("F1"), records)
    linked = PiAdapter().sessions_dir() / encode_session_folder("/elsewhere")
    linked.symlink_to(outside, target_is_directory=True)
    _age_tree(PiAdapter().sessions_dir(), 3600)
    _age_tree(outside, 3600)
    before = f1.read_bytes()
    service = _service(home)

    result = service.move(service.plan(OLD, NEW))["pi"]

    assert result.moved and f1.read_bytes() == before
    assert "symbolic link" in result.refused[linked / f1.name]
