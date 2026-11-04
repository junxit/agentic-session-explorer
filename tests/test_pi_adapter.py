"""Tests for sx.adapters.pi on SYNTHETIC stores under tmp_path.

The real ``~/.pi`` is never read. Every test redirects the module's ``home``
(adapters import it into their own namespace), and the suite's autouse fixtures
scrub the ``PI_*`` variables. Importing ``sx.adapters.omp`` is made to fail in
every test: whether Pi's store is Oh My Pi's is judged from the files in it,
never by asking the Oh My Pi adapter.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sx.adapters import pi as pi_module
from sx.adapters.base import JsonlFolderAdapter
from sx.adapters.pi import (
    UNRECOGNIZED_KEY,
    UNRECOGNIZED_TITLE,
    PiAdapter,
    TranscriptKind,
    agent_dir_override,
    encode_session_folder,
)
from sx.model import (
    BranchMode,
    Capability,
    Message,
    MovePlan,
    MoveResult,
    OrphanKind,
    ProjectLeftovers,
    Role,
    Session,
)
from sx.service import MoveService
from sx.util import file_mtime, is_within
from tests.conftest import write_compact_jsonl

OLD = Path("/old/proj")
NEW = Path("/new/home/proj")
TS = "2026-05-01T10:00:00.000Z"
DAY = 24 * 3600


# --- fixtures and builders ---------------------------------------------------


@pytest.fixture(autouse=True)
def _sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point Pi's home at ``tmp_path``, and make importing Oh My Pi's adapter fail."""
    monkeypatch.setattr(pi_module, "home", lambda: tmp_path)
    monkeypatch.setitem(sys.modules, "sx.adapters.omp", None)


def _sessions(tmp_path: Path) -> Path:
    """Return the default Pi sessions directory under the sandboxed home."""
    return tmp_path / ".pi" / "agent" / "sessions"


def _header(sid: str = "S1", cwd: str | None = str(OLD), **extra) -> dict:
    """Build a version 3 session header."""
    header = {"type": "session", "version": 3, "id": sid, "timestamp": TS}
    if cwd is not None:
        header["cwd"] = cwd
    return {**header, **extra}


def _entry(entry_id: str, parent: str | None, entry_type: str = "message", **fields) -> dict:
    """Build a tree entry."""
    return {"type": entry_type, "id": entry_id, "parentId": parent, "timestamp": TS, **fields}


def _user(entry_id: str, parent: str | None, content) -> dict:
    """Build a user message entry."""
    return _entry(entry_id, parent, message={"role": "user", "content": content, "timestamp": 1})


def _assistant(entry_id: str, parent: str | None, text: str) -> dict:
    """Build an assistant message entry holding one text block."""
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "stopReason": "stop",
        "timestamp": 2,
    }
    return _entry(entry_id, parent, message=message)


def _transcript(
    tmp_path: Path, records: list[dict], *, cwd: str = str(OLD), name: str = "s.jsonl"
) -> Path:
    """Write a transcript into the folder Pi uses for ``cwd``."""
    path = _sessions(tmp_path) / encode_session_folder(cwd) / name
    return write_compact_jsonl(path, records)


def _age(path: Path, seconds: float = 3600) -> None:
    """Backdate a file so it no longer counts as being written right now."""
    past = time.time() - seconds
    os.utime(path, (past, past))


def _age_tree(root: Path, seconds: float = 2 * DAY) -> None:
    """Backdate a folder and everything in it, deepest entries first.

    Pi creates a project's folder when it starts there, so a folder is offered
    for removal only once it has been idle for a day.
    """
    for top, dirs, files in os.walk(root, topdown=False):
        for name in (*files, *dirs):
            _age(Path(top, name), seconds)
    _age(root, seconds)


def _session_file(tmp_path: Path, cwd: str = str(OLD), sid: str = "S1", **header_extra) -> Path:
    """Write a realistic transcript: free text naming the project, and a torn last line."""
    name = f"2026-05-01T10-00-00-000Z_{sid}.jsonl"
    path = _sessions(tmp_path) / encode_session_folder(cwd) / name
    write_compact_jsonl(
        path,
        [
            _header(sid, cwd, **header_extra),
            _user("u1", None, f"edit {cwd}/src/app.py ✓"),
            _assistant("a1", "u1", f"done in {cwd}"),
        ],
    )
    with path.open("ab") as fh:
        fh.write(b'{"type":"message","id":"torn"')  # what a crash mid-append leaves
    _age(path)
    return path


def _only(adapter: PiAdapter) -> Session:
    """Return the single session an adapter discovers."""
    sessions = list(adapter.discover())
    assert len(sessions) == 1, sessions
    return sessions[0]


def _first(path: Path) -> dict:
    """Return a transcript's first line, parsed."""
    return json.loads(path.read_bytes().split(b"\n", 1)[0])


def _texts(messages: list[Message]) -> list[str]:
    """Return the text of each message."""
    return [message.text for message in messages]


def _move(adapter: PiAdapter, old: Path = OLD, new: Path = NEW, *, dry_run: bool = False):
    """Plan and perform a move for every session the adapter finds under ``old``."""
    plan = adapter.plan_move(adapter.sessions_for_project(old), old, new)
    return plan, adapter.move(plan, dry_run=dry_run)


# --- the folder encoding -----------------------------------------------------


@pytest.mark.parametrize(
    ("directory", "folder"),
    [
        ("/Users/Jade/source/junxit/agentic-session-explorer",
         "--Users-Jade-source-junxit-agentic-session-explorer--"),
        ("/Users/Jade/.extra/source/git_clones/usgpo/api",
         "--Users-Jade-.extra-source-git_clones-usgpo-api--"),
        ("/Users/Jade/Documents/Professional and Social Memberships/IEEE",
         "--Users-Jade-Documents-Professional and Social Memberships-IEEE--"),
        ("/Volumes/Archive/us-congress-pipeline", "--Volumes-Archive-us-congress-pipeline--"),
        ("/Users/Jade/source/junxit/forks--", "--Users-Jade-source-junxit-forks----"),
        ("/private/tmp/scratch/", "--private-tmp-scratch--"),
        ("/a/./b/../c", "--a-c--"),
        ("//srv//share", "--srv-share--"),
        ("/", "----"),
        ("C:\\Users\\me\\proj", "--C--Users-me-proj--"),
    ],
)
def test_folder_names_match_pis_encoding(directory, folder):
    """Separators and colons become hyphens; dots, spaces and underscores stay."""
    assert encode_session_folder(directory) == folder
    assert PiAdapter().folder_for(directory) == folder


def test_the_encoding_collides_so_folders_are_never_decoded():
    """``/a-b`` and ``/a/b`` share a folder, which is why the header decides."""
    assert encode_session_folder("/a-b") == encode_session_folder("/a/b") == "--a-b--"


# --- discovery ---------------------------------------------------------------


def test_a_session_is_built_from_its_header(tmp_path):
    """Id, project, start time, size and paths all come from the file itself."""
    project = tmp_path / "work" / "app"
    project.mkdir(parents=True)
    path = _transcript(
        tmp_path, [_header("S1", str(project)), _user("u1", None, "hello there")], cwd=str(project)
    )

    session = _only(PiAdapter())

    assert session.harness == "pi"
    assert session.session_id == "S1"
    assert session.project_path == str(project)
    assert session.created == datetime(2026, 5, 1, 10, tzinfo=UTC)
    assert session.modified == file_mtime(path)
    assert session.size_bytes == path.stat().st_size
    assert session.paths == [path]
    assert session.is_orphan is False
    assert session.title == "hello there"
    assert session.extra == {}


def test_a_session_whose_project_is_gone_is_an_orphan(tmp_path):
    """A recorded directory that no longer exists marks the session."""
    _transcript(tmp_path, [_header("S1", "/sx-test/gone/proj")], cwd="/sx-test/gone/proj")
    assert _only(PiAdapter()).is_orphan is True


def test_a_project_on_an_unmounted_volume_is_not_an_orphan(tmp_path):
    """An unplugged disk is unavailable, not deleted."""
    cwd = "/Volumes/sx-test-unplugged-disk/proj"
    _transcript(tmp_path, [_header("S1", cwd)], cwd=cwd)
    assert _only(PiAdapter()).is_orphan is False


def test_a_header_without_a_cwd_has_no_project(tmp_path):
    """No ``cwd`` means unknown — never a project derived from the folder name."""
    _transcript(tmp_path, [_header("S1", None)])
    session = _only(PiAdapter())
    assert session.project_path is None
    assert session.is_orphan is False


def test_only_top_level_transcripts_with_a_header_are_sessions(tmp_path):
    """Fragments, nested files, flat files and dot-files are not sessions."""
    sessions = _sessions(tmp_path)
    folder = sessions / encode_session_folder(str(OLD))
    write_compact_jsonl(folder / "good.jsonl", [_header("good")])
    write_compact_jsonl(folder / "fragment.jsonl", [_user("u9", "u8", "appended after a delete")])
    write_compact_jsonl(folder / "nested" / "deep.jsonl", [_header("nested")])
    write_compact_jsonl(folder / ".hidden.jsonl", [_header("hidden")])
    write_compact_jsonl(sessions / "flat.jsonl", [_header("flat")])
    write_compact_jsonl(sessions / ".dot-folder" / "x.jsonl", [_header("dotted")])
    (folder / "notes.txt").write_text("not a transcript", encoding="utf-8")

    assert [s.session_id for s in PiAdapter().discover()] == ["good"]


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (["", "not json", '{"type":"session","id":"S1","cwd":"/p"}'], "S1"),
        (["null", "0", '""', "false", '{"type":"session","id":"S1"}'], "S1"),
        (["[]", '{"type":"session","id":"S1"}'], None),
        (["{}", '{"type":"session","id":"S1"}'], None),
        (['{"type":"message","id":"a"}', '{"type":"session","id":"S1"}'], None),
        (['{"type":"session","id":7}'], None),
        (['{"type":"session"}'], None),
        ([], None),
    ],
)
def test_the_header_is_the_first_parsed_value_as_in_pi(tmp_path, lines, expected):
    """Blank, malformed and falsy lines are skipped; any other first value decides."""
    path = tmp_path / "t.jsonl"
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    header = PiAdapter().read_header(path)
    assert (header["id"] if header else None) == expected


_HEADER_LINE = '{"type":"session","version":3,"id":"S1","cwd":"/old/proj"}'


@pytest.mark.parametrize(
    ("lines", "kind"),
    [
        ([_HEADER_LINE], TranscriptKind.SESSION),
        ([], TranscriptKind.FRAGMENT),
        (['{"type":"message","id":"u9","parentId":"u8","message":{}}'], TranscriptKind.FRAGMENT),
        (['{"type":"label","id":"l1","parentId":null,"targetId":"x"}'], TranscriptKind.FRAGMENT),
        # Only the first line itself makes debris: an entry after a blank, a
        # null or a damaged line is a session that lost its header.
        (["", "null", "{not json", '{"type":"usage","id":"x","parentId":"y"}'],
         TranscriptKind.UNRECOGNIZED),
        (["{not json", '{"type":"usage","id":"x","parentId":"y"}'], TranscriptKind.UNRECOGNIZED),
        (["null", '{"type":"usage","id":"x","parentId":"y"}'], TranscriptKind.UNRECOGNIZED),
        (['{"type":"message","id":"u1"}'], TranscriptKind.UNRECOGNIZED),
        (['{"type":"message","id":"u1","parentId":7}'], TranscriptKind.UNRECOGNIZED),
        (['{"type":"message","id":"","parentId":null}'], TranscriptKind.UNRECOGNIZED),
        (['{"type":"model_usage","id":"m","parentId":null}'], TranscriptKind.UNRECOGNIZED),
        (['{"type":"preamble","format":4}', _HEADER_LINE], TranscriptKind.UNRECOGNIZED),
        (["[1, 2]"], TranscriptKind.UNRECOGNIZED),
        (["", "not json at all"], TranscriptKind.UNRECOGNIZED),
    ],
)
def test_every_file_is_a_session_a_fragment_or_unrecognized(tmp_path, lines, kind):
    """Only an empty file or a Pi tree entry first is debris; anything else is kept."""
    path = tmp_path / "t.jsonl"
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    assert PiAdapter().transcript_kind(path) is kind


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no named pipes here")
def test_a_pipe_named_like_a_transcript_is_never_opened(tmp_path):
    """Opening a pipe with no writer blocks forever; it is unrecognized without a read."""
    path = _session_file(tmp_path)
    pipe = path.parent / "2026-05-01T10-00-00-000Z_PIPE.jsonl"
    os.mkfifo(pipe)
    adapter = PiAdapter()
    found: dict = {}

    def probe() -> None:
        """Classify the pipe and sample the store for Oh My Pi's marks."""
        found["kind"] = adapter.transcript_kind(pipe)
        found["busy"] = adapter.busy_transcript(pipe)
        found["evidence"] = adapter.omp_evidence()
        found["sessions"] = [s.session_id for s in adapter.discover()]

    worker = threading.Thread(target=probe, daemon=True)
    worker.start()
    worker.join(10)

    assert not worker.is_alive(), "a pipe was opened"
    assert found == {
        "kind": TranscriptKind.UNRECOGNIZED, "busy": True, "evidence": [], "sessions": ["S1"]
    }


def test_a_missing_file_is_missing_not_unrecognized(tmp_path):
    """Nothing there: neither a session to keep nor a file to protect."""
    adapter = PiAdapter()
    assert adapter.transcript_kind(tmp_path / "gone.jsonl") is TranscriptKind.MISSING
    assert adapter.busy_transcript(tmp_path / "gone.jsonl") is False


def test_an_unrecognized_file_is_listed_flagged_and_protected(tmp_path):
    """No project, so no move selects it; every delete and move refuses it."""
    path = _transcript(
        tmp_path,
        [{"type": "preamble", "format": 4}, _header(), _user("u1", None, "kept as it is")],
        name="2026-05-01T10-00-00-000Z_X.jsonl",
    )
    _age(path)
    adapter = PiAdapter()

    session = _only(adapter)

    assert session.title == f"{UNRECOGNIZED_TITLE} {path.name}"
    assert session.extra == {UNRECOGNIZED_KEY: "does not start with a session header"}
    assert (session.project_path, session.is_orphan, session.paths) == (None, False, [path])
    assert adapter.sessions_for_project(OLD) == []
    assert "does not recognize" in adapter.delete(session, dry_run=True).refused[path]
    assert "does not recognize" in adapter.plan_move([session], OLD, NEW).blocked[path]
    assert adapter.busy_transcript(path) is True
    assert _texts(adapter.load(session)) == ["kept as it is"]
    assert adapter.find_orphans() == []


def test_a_fork_keeps_its_parent_and_says_so(tmp_path):
    """``parentSession`` is kept for the UI, and the title marks the fork."""
    parent = "/x/.pi/agent/sessions/--old-proj--/a.jsonl"
    _transcript(tmp_path, [_header("S2", parentSession=parent), _user("u1", None, "forked")])
    session = _only(PiAdapter())
    assert session.extra == {"parent_session": parent}
    assert session.title == "forked · fork"


def test_the_project_comes_from_the_header_not_the_folder(tmp_path):
    """Two projects that encode alike share one folder and keep their own paths."""
    folder = _sessions(tmp_path) / "--a-b--"
    write_compact_jsonl(folder / "one.jsonl", [_header("one", "/a-b")])
    write_compact_jsonl(folder / "two.jsonl", [_header("two", "/a/b")])

    found = {s.session_id: s.project_path for s in PiAdapter().discover()}

    assert found == {"one": "/a-b", "two": "/a/b"}


def test_availability_needs_the_sessions_directory_not_the_agent_dir(tmp_path):
    """Another tool's ``extensions/`` in the agent dir does not mean Pi is used."""
    (tmp_path / ".pi" / "agent" / "extensions").mkdir(parents=True)
    adapter = PiAdapter()
    assert adapter.available() is False
    assert list(adapter.discover()) == []

    _sessions(tmp_path).mkdir()
    assert adapter.available() is True


def test_capabilities_branches_and_store_roots():
    """Pi browses, cleans, deletes and moves; its only store root is the sessions dir."""
    adapter = PiAdapter()
    assert adapter.capabilities == (
        Capability.BROWSE | Capability.ORPHANS | Capability.DELETE | Capability.MOVE
    )
    assert adapter.supports_branches is True
    assert adapter.store_roots() == [adapter.sessions_dir()]
    assert adapter.sessions_root() == adapter.sessions_dir()


def test_liveness_is_the_transcripts_mtime(tmp_path):
    """Pi holds no lock, so the inherited mtime check is the liveness signal."""
    path = _session_file(tmp_path)
    adapter = PiAdapter()
    assert adapter.last_activity(_only(adapter)) == file_mtime(path)


def test_the_confirmations_warn_that_pi_marks_no_open_session():
    """Delete and move dialogs say what removing a session Pi still has open does."""
    from sx.tui.app import _open_session_caveat

    caveat = PiAdapter.open_session_caveat
    assert caveat and "close" in caveat
    assert "folder is gone" in caveat and "never lists" in caveat
    assert _open_session_caveat(PiAdapter()) == caveat[:1].upper() + caveat[1:]


def test_busy_transcript_judges_the_format_not_whether_pi_is_writing(tmp_path):
    """Pi leaves nothing to check: a transcript written this instant is not "busy"."""
    fresh = _transcript(tmp_path, [_header("S1"), _user("u1", None, "now")], name="a.jsonl")
    newer = _transcript(tmp_path, [_header("S2", version=4)], name="b.jsonl")
    fragment = _transcript(tmp_path, [_user("u9", "u8", "appended")], name="c.jsonl")
    unknown = _transcript(tmp_path, [{"type": "preamble"}, _header("S3")], name="d.jsonl")
    adapter = PiAdapter()

    assert adapter.busy_transcript(fresh) is False
    assert adapter.busy_transcript(fragment) is False
    assert adapter.busy_transcript(newer) is True
    assert adapter.busy_transcript(unknown) is True


# --- titles --------------------------------------------------------------------


def test_the_latest_session_name_wins(tmp_path):
    """``/name`` can be used repeatedly; the last ``session_info`` is the name."""
    _transcript(
        tmp_path,
        [
            _header(),
            _user("u1", None, "a prompt"),
            _entry("n1", "u1", "session_info", name="First name"),
            _entry("n2", "n1", "session_info", name="  Refactor auth  "),
        ],
    )
    assert _only(PiAdapter()).title == "Refactor auth"


def test_an_empty_session_name_clears_it(tmp_path):
    """An explicit empty name falls back to the first user message, as in Pi."""
    _transcript(
        tmp_path,
        [
            _header(),
            _entry("n1", None, "session_info", name="Old name"),
            _user("u1", "n1", "the real prompt"),
            _entry("n2", "u1", "session_info", name="   "),
        ],
    )
    assert _only(PiAdapter()).title == "the real prompt"


def test_the_title_is_the_whole_first_user_text_as_pis_list_shows_it(tmp_path):
    """Image-only prompts are passed over; control characters become spaces; cut at 80."""
    long_line = "Fix   the\tflaky " + "x" * 80
    image = {"type": "image", "data": "QUJD", "mimeType": "image/png"}
    _transcript(
        tmp_path,
        [
            _header(),
            _user("u1", None, [image]),
            _user("u2", "u1", [{"type": "text", "text": f"\n\n{long_line}\nsecond line"}]),
        ],
    )
    # Pi does not collapse runs of spaces: "Fix   the flaky" keeps its three.
    assert _only(PiAdapter()).title == "Fix   the flaky " + "x" * 64 + "…"


def test_a_multi_line_prompt_titles_the_session_whole(tmp_path):
    """Pi lists the whole first message, its newlines turned into spaces."""
    _transcript(
        tmp_path, [_header(), _user("u1", None, "Refactor the parser\nthen add tests\x07")]
    )
    assert _only(PiAdapter()).title == "Refactor the parser then add tests"


def test_a_session_name_is_cleaned_like_a_prompt(tmp_path):
    """Control characters in a name become spaces too, and a long one is cut."""
    _transcript(
        tmp_path,
        [
            _header(),
            _user("u1", None, "a prompt"),
            _entry("n1", "u1", "session_info", name="Auth\tredo " + "y" * 100),
        ],
    )
    assert _only(PiAdapter()).title == "Auth redo " + "y" * 70 + "…"


def test_a_prompt_that_is_blank_once_cleaned_is_passed_over(tmp_path):
    """Pi would show an empty title; sx titles the session by its next prompt."""
    _transcript(
        tmp_path, [_header(), _user("u1", None, "\n\t\x01 "), _user("u2", "u1", "the real one")]
    )
    assert _only(PiAdapter()).title == "the real one"


def test_text_blocks_join_with_a_space_as_in_pis_session_list(tmp_path):
    """Images are left out of a title; the text around them reads as one line."""
    image = {"type": "image", "data": "QUJD", "mimeType": "image/png"}
    content = [{"type": "text", "text": "Explain"}, image, {"type": "text", "text": "this diagram"}]
    _transcript(tmp_path, [_header(), _user("u1", None, content)])
    assert _only(PiAdapter()).title == "Explain this diagram"


def test_a_session_without_user_text_is_untitled(tmp_path):
    """No name and no user text: Pi's own placeholder."""
    _transcript(tmp_path, [_header(), _assistant("a1", None, "unprompted")])
    assert _only(PiAdapter()).title == "(no messages)"


def test_the_title_scan_parses_only_marked_lines(tmp_path, monkeypatch):
    """A multi-megabyte line that cannot hold a title is never parsed."""
    blob = "A" * 2_000_000
    path = _transcript(
        tmp_path,
        [
            _header(),
            _assistant("a0", None, blob),
            _user("u1", "a0", "short prompt"),
            _assistant("a1", "u1", blob),
        ],
    )
    parsed: list[int] = []
    real = pi_module._parse_line

    def recording(raw: bytes):
        """Record the size of every line handed to the parser."""
        parsed.append(len(raw))
        return real(raw)

    monkeypatch.setattr(pi_module, "_parse_line", recording)

    assert PiAdapter().title_for_file(path, {}) == "short prompt"
    assert parsed and all(size < 1000 for size in parsed)


# --- entries to messages ---------------------------------------------------------


def _message_entry(message) -> dict:
    """Wrap a ``message`` value in a message entry."""
    return _entry("e1", None, message=message)


def _render(entry: dict) -> Message | None:
    """Render one entry with a fresh adapter."""
    return PiAdapter().entry_message(entry)


def test_user_messages_render_text_and_image_placeholders():
    """A string or text blocks; images appear as ``[image]``."""
    plain = _render(_message_entry({"role": "user", "content": "hello"}))
    assert (plain.role, plain.text) == (Role.USER, "hello")
    assert plain.timestamp == datetime(2026, 5, 1, 10, tzinfo=UTC)

    image = {"type": "image", "data": "QUJD", "mimeType": "image/png"}
    blocks = _render(
        _message_entry({"role": "user", "content": [{"type": "text", "text": "look"}, image]})
    )
    assert blocks.text == "look\n[image]"


def test_assistant_messages_split_text_thinking_and_tool_calls():
    """Thinking is kept apart; tool calls become a summary of their names."""
    message = _render(
        _message_entry(
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "pondering", "thinkingSignature": "x"},
                    {"type": "text", "text": "Running it."},
                    {"type": "toolCall", "id": "c1", "name": "bash", "arguments": {}},
                    {"type": "toolCall", "id": "c2", "name": "read", "arguments": {}},
                ],
                "stopReason": "toolUse",
            }
        )
    )
    assert (message.role, message.text) == (Role.ASSISTANT, "Running it.")
    assert (message.thinking, message.tool_summary) == ("pondering", "bash; read")


def test_a_failed_assistant_turn_shows_its_error():
    """``stopReason: error`` adds the error message, so a failed turn is visible."""
    bare = _render(
        _message_entry(
            {"role": "assistant", "content": [], "stopReason": "error",
             "errorMessage": "rate limited"}
        )
    )
    assert bare.text == "Error: rate limited"

    partial = _render(
        _message_entry(
            {"role": "assistant", "content": [{"type": "text", "text": "partial"}],
             "stopReason": "error"}
        )
    )
    assert partial.text == "partial\n\nError: Unknown error"  # Pi's words for a missing message


@pytest.mark.parametrize(
    ("error", "shown"),
    [
        (None, "Operation aborted"),
        ("Request was aborted", "Operation aborted"),
        ("Operation aborted", "Operation aborted"),
        ("Aborted after 2 retry attempts", "Aborted after 2 retry attempts"),
    ],
)
def test_an_aborted_turn_shows_what_pi_shows(error, shown):
    """An empty aborted turn is no longer dropped; Pi's notice follows any partial text."""
    message = {"role": "assistant", "content": [], "stopReason": "aborted"}
    if error is not None:
        message["errorMessage"] = error
    assert _render(_message_entry(message)).text == shown

    partial = {**message, "content": [{"type": "text", "text": "half an answer"}]}
    assert _render(_message_entry(partial)).text == f"half an answer\n\n{shown}"


def test_an_aborted_turn_with_tool_calls_says_the_operation_was_aborted():
    """Pi marks each of its tool calls "Operation aborted" when the session is reopened."""
    message = {
        "role": "assistant",
        "content": [{"type": "toolCall", "id": "c1", "name": "bash", "arguments": {}}],
        "stopReason": "aborted",
        "errorMessage": "Aborted after 1 retry attempt",
    }
    rendered = _render(_message_entry(message))
    assert (rendered.text, rendered.tool_summary) == ("Operation aborted", "bash")


def test_a_truncated_turn_says_so():
    """``stopReason: length``: Pi adds its truncation notice after the partial text."""
    message = {"role": "assistant", "content": [{"type": "text", "text": "The answer is"}],
               "stopReason": "length"}
    assert _render(_message_entry(message)).text == (
        "The answer is\n\nResponse was truncated before completion."
    )


def test_an_empty_assistant_turn_is_skipped():
    """Nothing to show — including redacted thinking — renders nothing."""
    assert _render(_message_entry({"role": "assistant", "content": []})) is None
    redacted = {"type": "thinking", "thinking": "", "redacted": True}
    assert _render(_message_entry({"role": "assistant", "content": [redacted]})) is None


def test_tool_results_summarize_name_and_error_and_truncate():
    """The summary names the tool; the output is cut at 200 characters."""
    result = _render(
        _message_entry(
            {"role": "toolResult", "toolCallId": "c1", "toolName": "bash",
             "content": [{"type": "text", "text": "x" * 300}], "isError": False}
        )
    )
    assert (result.role, result.tool_summary) == (Role.TOOL, "bash result")
    assert result.text == "x" * 200 + "…"

    failed = _render(
        _message_entry(
            {"role": "toolResult", "toolName": "edit", "content": [], "isError": True}
        )
    )
    assert (failed.tool_summary, failed.text) == ("edit result (error)", "")

    anonymous = _render(_message_entry({"role": "toolResult", "content": "done"}))
    assert anonymous.tool_summary == "tool result"


def test_bash_executions_show_command_output_and_exit_code():
    """A ``!`` command: the command on one line, its output, and how it exited."""
    run = _render(
        _message_entry(
            {"role": "bashExecution", "command": "git status\n  --short",
             "output": "M file.py\n", "exitCode": 1, "cancelled": False, "truncated": False}
        )
    )
    assert (run.role, run.tool_summary) == (Role.TOOL, "bash: git status --short")
    assert run.text == "M file.py\n(exit code 1)"

    long = _render(
        _message_entry({"role": "bashExecution", "command": "y" * 200, "output": "o" * 300})
    )
    assert long.tool_summary == "bash: " + "y" * 120 + "…"
    assert long.text == "o" * 200 + "…"

    cancelled = _render(
        _message_entry({"role": "bashExecution", "command": "sleep 9", "output": "",
                        "cancelled": True})
    )
    assert cancelled.text == ""


def test_extension_messages_show_only_when_displayed():
    """``custom`` (and v2's ``hookMessage``) appear only when Pi shows them."""
    shown = _render(_message_entry({"role": "custom", "customType": "x", "content": "note",
                                    "display": True}))
    assert (shown.role, shown.text) == (Role.OTHER, "note")
    assert _render(_message_entry({"role": "custom", "content": "quiet", "display": False})) is None
    legacy = _render(_message_entry({"role": "hookMessage", "content": "old", "display": True}))
    assert legacy.text == "old"


@pytest.mark.parametrize("role", ["system", "developer", "branchSummary", "somethingNew"])
def test_roles_pi_does_not_show_are_skipped(role):
    """System prompts, developer messages and unknown roles are not transcript."""
    assert _render(_message_entry({"role": role, "content": "hidden"})) is None


def test_compactions_summaries_and_model_changes_render():
    """The non-message entries a reader needs to follow the session."""
    compaction = _render(_entry("c", None, "compaction", summary="User discussed X",
                                firstKeptEntryId="c", tokensBefore=5))
    assert compaction.role is Role.SYSTEM
    assert compaction.text == "Context compacted\n\nUser discussed X"
    assert _render(_entry("c", None, "compaction")).text == "Context compacted"

    summary = _render(_entry("b", None, "branch_summary", fromId="f", summary="Tried approach A"))
    assert (summary.role, summary.text) == (Role.SYSTEM, "Branch summary: Tried approach A")

    model = _render(_entry("m", None, "model_change", provider="anthropic",
                           modelId="claude-sonnet-4-5"))
    assert (model.role, model.text) == (Role.SYSTEM, "model → anthropic/claude-sonnet-4-5")

    injected = _render(_entry("i", None, "custom_message", customType="x", content="Injected",
                              display=True))
    assert (injected.role, injected.text) == (Role.OTHER, "Injected")
    hidden = _entry("i", None, "custom_message", customType="x", content="Injected", display=False)
    assert _render(hidden) is None


@pytest.mark.parametrize(
    "entry",
    [
        _entry("l", "x", "label", targetId="x", label="checkpoint"),
        _entry("n", "x", "session_info", name="A name"),
        _entry("u", "x", "usage", kind="cache_warm", usage={}),
        _entry("c", "x", "custom", customType="ext", data={"n": 1}),
        _entry("e", "x", "context_edit", targetId="x", replacement=None),
        _entry("t", "x", "thinking_level_change", thinkingLevel="high"),
        _entry("b", "x", "branch_summary", fromId="f", summary=""),
        _entry("m", "x", "model_change"),
        _entry("z", "x", "a_future_entry_type", payload=1),
    ],
)
def test_bookkeeping_entries_are_skipped(entry):
    """Labels, names, usage, extension state and edits are not conversation."""
    assert _render(entry) is None


@pytest.mark.parametrize(
    "entry",
    [
        {"type": "message"},
        {"type": "message", "message": "a string"},
        {"type": "message", "message": {"role": ["user"], "content": "x"}},
        {"type": "message", "message": {"role": "user", "content": 42}},
        {"type": "message",
         "message": {"role": "user", "content": [None, 3, "s", {"type": "text", "text": 5}]}},
        {"type": "message",
         "message": {"role": "assistant", "content": [{"type": "toolCall", "name": {"x": 1}}],
                     "stopReason": "error", "errorMessage": 3}},
        {"type": "message",
         "message": {"role": "toolResult", "content": {"type": "text"}, "toolName": ["x"]}},
        {"type": "message",
         "message": {"role": "bashExecution", "command": 5, "output": None, "exitCode": True}},
        {"type": "compaction", "summary": {"nested": True}},
        {"type": "model_change", "provider": ["a"], "modelId": None},
        {"type": "custom_message", "display": True, "content": [{"type": "image"}]},
        {"type": ["message"], "timestamp": 12345},
        {"type": "message", "timestamp": "not a date",
         "message": {"role": "user", "content": "ok"}},
    ],
)
def test_malformed_entries_never_raise(entry):
    """Any shape at all renders as a message or as nothing."""
    result = _render(entry)
    assert result is None or isinstance(result, Message)


# --- loading: linear files ----------------------------------------------------------


def test_a_version_1_file_reads_linearly(tmp_path):
    """Entries without ids cannot form a tree; the file is shown in order."""
    records = [
        {"type": "session", "id": "V1", "timestamp": TS, "cwd": str(OLD)},
        {"type": "message", "timestamp": TS, "message": {"role": "user", "content": "one"}},
        {"type": "message", "timestamp": TS,
         "message": {"role": "assistant", "content": [{"type": "text", "text": "two"}]}},
        {"type": "model_change", "timestamp": TS, "provider": "openai", "modelId": "gpt-4o"},
        {"type": "message", "timestamp": TS, "message": {"role": "user", "content": "three"}},
    ]
    _transcript(tmp_path, records)
    adapter = PiAdapter()
    session = _only(adapter)

    transcript = adapter.load_transcript(session)

    assert _texts(transcript.current) == ["one", "two", "model → openai/gpt-4o", "three"]
    assert (transcript.branches, transcript.tree, transcript.hidden) == ([], [], 0)
    assert adapter.load(session) == transcript.current


def test_entries_without_ids_under_a_v3_header_read_linearly(tmp_path):
    """One entry missing its id is enough to give up on the tree."""
    copied = {"type": "message", "timestamp": TS,
              "message": {"role": "user", "content": "copied without an id"}}
    _transcript(tmp_path, [_header(), _user("u1", None, "first"), copied, _user("u2", "u1", "x")])
    adapter = PiAdapter()

    transcript = adapter.load_transcript(_only(adapter))

    assert _texts(transcript.current) == ["first", "copied without an id", "x"]
    assert not transcript.branched


# --- loading: the tree ------------------------------------------------------------


def _branched(tmp_path: Path) -> tuple[PiAdapter, Session]:
    """Write a session with a retried prompt, two follow-ups and a label.

    ::

        u1 ─ a1 ─┬─ u2 ─ a2 ─┬─ u3x              (other branch 1 …)
                 │           └─ u3y              (… and other branch 2)
                 └─ u2b ─ a2b ─ lbl ─ u4         (current: u4 is written last)
    """
    _transcript(
        tmp_path,
        [
            _header(),
            _user("u1", None, "first question"),
            _assistant("a1", "u1", "first answer"),
            _user("u2", "a1", "second question"),
            _assistant("a2", "u2", "second answer"),
            _user("u2b", "a1", "second question, retried"),
            _assistant("a2b", "u2b", "retried answer"),
            _user("u3x", "a2", "follow-up A"),
            _user("u3y", "a2", "follow-up B"),
            _entry("lbl", "a2b", "label", targetId="u1", label="start"),
            _user("u4", "lbl", "final question"),
        ],
    )
    adapter = PiAdapter()
    return adapter, _only(adapter)


def test_the_current_branch_ends_at_the_last_entry(tmp_path):
    """Pi stores no leaf pointer; it resumes from whatever was written last."""
    adapter, session = _branched(tmp_path)
    transcript = adapter.load_transcript(session)
    assert _texts(transcript.current) == [
        "first question",
        "first answer",
        "second question, retried",
        "retried answer",
        "final question",
    ]
    assert adapter.load(session) == transcript.current


def test_other_branches_are_labeled_by_where_they_fork(tmp_path):
    """Each branch holds only its own messages and says where it leaves off."""
    adapter, session = _branched(tmp_path)
    transcript = adapter.load_transcript(session)

    assert [(b.label, _texts(b.messages)) for b in transcript.branches] == [
        ("forks from the current branch after message 2",
         ["second question", "second answer", "follow-up A"]),
        ("forks from other branch 1 after message 2", ["follow-up B"]),
    ]
    assert transcript.hidden == 4
    assert transcript.describe(BranchMode.CURRENT) == (
        "current branch only — 4 message(s) on 2 other branch(es) not shown"
    )


def test_the_tree_view_marks_every_jump_between_branches(tmp_path):
    """File order, with a marker wherever the next entry is not the previous one's child."""
    adapter, session = _branched(tmp_path)
    tree = adapter.load_transcript(session).tree

    assert [(m.marker, m.text) for m in tree] == [
        (False, "first question"),
        (False, "first answer"),
        (False, "second question"),
        (False, "second answer"),
        (True, "branch — continues after message 2 (current branch)"),
        (False, "second question, retried"),
        (False, "retried answer"),
        (True, "branch — continues after message 4 (other branch)"),
        (False, "follow-up A"),
        (True, "branch — continues after message 4 (other branch)"),
        (False, "follow-up B"),
        (True, "branch — continues after message 6 (current branch)"),
        (False, "final question"),
    ]
    assert all(m.role is Role.OTHER for m in tree if m.marker)


def test_appendix_mode_lists_each_branch_after_the_current_one(tmp_path):
    """The model's appendix headings carry the fork labels."""
    adapter, session = _branched(tmp_path)
    messages = adapter.load_transcript(session).messages(BranchMode.APPENDIX)
    assert [m.text for m in messages if m.marker] == [
        "Other branch 1 of 2 — forks from the current branch after message 2",
        "Other branch 2 of 2 — forks from other branch 1 after message 2",
    ]


def test_the_leaf_can_sit_on_an_older_branch(tmp_path):
    """Going back to an earlier answer makes the newer one the side branch."""
    _transcript(
        tmp_path,
        [
            _header(),
            _user("u1", None, "q"),
            _assistant("a1", "u1", "answer one"),
            _assistant("a2", "u1", "answer two"),
            _user("u3", "a1", "back on the first answer"),
        ],
    )
    adapter = PiAdapter()
    transcript = adapter.load_transcript(_only(adapter))

    assert _texts(transcript.current) == ["q", "answer one", "back on the first answer"]
    assert [(b.label, _texts(b.messages)) for b in transcript.branches] == [
        ("forks from the current branch after message 1", ["answer two"])
    ]


def test_an_unbranched_tree_has_no_branches(tmp_path):
    """A linear id-linked session sets no branches, tree or hidden count."""
    _transcript(
        tmp_path,
        [
            _header(),
            _user("u1", None, "q"),
            _assistant("a1", "u1", "a"),
            _entry("l1", "a1", "label", targetId="u1", label="x"),
        ],
    )
    adapter = PiAdapter()
    transcript = adapter.load_transcript(_only(adapter))
    assert _texts(transcript.current) == ["q", "a"]
    assert (transcript.branches, transcript.tree, transcript.hidden) == ([], [], 0)


def test_a_side_branch_that_shows_nothing_is_not_a_branch(tmp_path):
    """Bookkeeping left off the current branch is not worth a heading."""
    _transcript(
        tmp_path,
        [
            _header(),
            _user("u1", None, "q"),
            _assistant("a1", "u1", "a"),
            _entry("t1", "u1", "thinking_level_change", thinkingLevel="high"),
            _user("u2", "a1", "next"),
        ],
    )
    adapter = PiAdapter()
    transcript = adapter.load_transcript(_only(adapter))
    assert _texts(transcript.current) == ["q", "a", "next"]
    assert not transcript.branched


def test_a_repeated_id_keeps_its_last_occurrence(tmp_path):
    """As in Pi's index, the later entry with an id replaces the earlier one."""
    _transcript(
        tmp_path,
        [
            _header(),
            _user("u1", None, "q"),
            _assistant("a1", "u1", "first draft"),
            _assistant("a1", "u1", "final draft"),
        ],
    )
    adapter = PiAdapter()
    transcript = adapter.load_transcript(_only(adapter))
    assert _texts(transcript.current) == ["q", "final draft"]
    assert not transcript.branched


def test_cycles_and_missing_parents_do_not_hang(tmp_path):
    """Corrupt links end a walk; every entry still lands in exactly one place."""
    _transcript(
        tmp_path,
        [
            _header(),
            _user("x", "y", "x points at y"),
            _user("y", "x", "y points at x"),
            _user("s", "s", "its own parent"),
            _user("o", "nowhere", "parent missing"),
            _user("r", None, "new root"),
            _assistant("leaf", "r", "the end"),
        ],
    )
    adapter = PiAdapter()
    transcript = adapter.load_transcript(_only(adapter))

    assert _texts(transcript.current) == ["new root", "the end"]
    assert [b.label for b in transcript.branches] == ["a separate root"] * 3
    assert sorted(text for b in transcript.branches for text in _texts(b.messages)) == [
        "its own parent",
        "parent missing",
        "x points at y",
        "y points at x",
    ]
    assert transcript.hidden == 4
    assert sum(not m.marker for m in transcript.tree) == 6


def test_a_leaf_inside_a_cycle_still_loads(tmp_path):
    """Two entries that are each other's parent: the walk stops, nothing hangs."""
    _transcript(tmp_path, [_header(), _user("a", "b", "a"), _user("b", "a", "b")])
    adapter = PiAdapter()
    transcript = adapter.load_transcript(_only(adapter))
    assert _texts(transcript.current) == ["a", "b"]
    assert not transcript.branched


# --- PI_CODING_AGENT_DIR ----------------------------------------------------------


def test_an_absolute_agent_dir_override_is_followed(tmp_path, monkeypatch):
    """A decoy under the default proves the override was followed, not stumbled on."""
    _transcript(tmp_path, [_header("decoy")])
    custom = tmp_path / "elsewhere" / "pi-agent"
    folder = custom / "sessions" / encode_session_folder(str(OLD))
    write_compact_jsonl(folder / "s.jsonl", [_header("real")])
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(custom))

    adapter = PiAdapter()

    assert adapter.sessions_dir() == custom / "sessions"
    assert adapter.store_roots() == [custom / "sessions"]
    assert [s.session_id for s in adapter.discover()] == ["real"]
    notes = adapter.store_notes()
    assert len(notes) == 1 and "PI_CODING_AGENT_DIR" in notes[0]
    assert adapter.store_warnings() == []


def test_a_tilde_override_expands_against_home(tmp_path, monkeypatch):
    """``~/…`` is expanded the way Pi's ``expandTildePath`` does it."""
    monkeypatch.setenv("PI_CODING_AGENT_DIR", "~/alt/agent")
    assert PiAdapter().sessions_dir() == tmp_path / "alt" / "agent" / "sessions"


def test_the_override_is_not_trimmed(tmp_path, monkeypatch):
    """Pi uses the value as given: a trailing space is part of the directory's name."""
    monkeypatch.setenv("PI_CODING_AGENT_DIR", f"{tmp_path}/agent ")
    assert PiAdapter().sessions_dir() == tmp_path / "agent " / "sessions"


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        ("relative/agent", "a relative path"),
        ("~other/agent", "a relative path"),
        ("~", "the home directory"),
        ("~/", "the home directory"),
        ("/", "a filesystem root"),
        ("   ", "a relative path"),
        ("  /data/pi-agent", "a relative path"),
        ("file://server/share/agent", "a file: URL"),
        ("file:///data/a%2Fb", "a file: URL"),
        ("file:///data/agent?x=1", "a file: URL"),
        ("file:///data/%zz", "a file: URL"),
        ("file:///data/%ff", "a file: URL"),
    ],
)
def test_an_unusable_override_is_ignored_with_a_warning(tmp_path, monkeypatch, value, reason):
    """The default store is read, and the user is told why."""
    monkeypatch.setenv("PI_CODING_AGENT_DIR", value)
    adapter = PiAdapter()

    assert adapter.sessions_dir() == _sessions(tmp_path)
    warnings = adapter.store_warnings()
    assert len(warnings) == 1
    assert "PI_CODING_AGENT_DIR" in warnings[0] and reason in warnings[0]
    assert adapter.store_notes() == []


def test_the_home_directory_is_refused_however_it_is_spelled(tmp_path, monkeypatch):
    """``~/sessions`` would otherwise join the delete allowlist."""
    monkeypatch.setenv("PI_CODING_AGENT_DIR", f"{tmp_path}/sub/..")
    assert agent_dir_override(tmp_path) == (None, "the home directory")


# --- custom session directories --------------------------------------------------------


def test_a_session_dir_variable_is_warned_about(tmp_path, monkeypatch):
    """Sessions written to a custom directory are not listed, and sx says so."""
    monkeypatch.setenv("PI_CODING_AGENT_SESSION_DIR", "~/flat-sessions")
    warnings = PiAdapter().store_warnings()
    assert len(warnings) == 1
    assert "PI_CODING_AGENT_SESSION_DIR" in warnings[0] and "not listed" in warnings[0]


def test_a_session_dir_setting_is_warned_about(tmp_path):
    """``sessionDir`` in Pi's settings (BOM and all) redirects new sessions too."""
    settings = tmp_path / ".pi" / "agent" / "settings.json"
    settings.parent.mkdir(parents=True)
    payload = json.dumps({"theme": "dark", "sessionDir": "/data/pi-sessions"})
    settings.write_text("\ufeff" + payload, encoding="utf-8")

    warnings = PiAdapter().store_warnings()

    assert len(warnings) == 1
    assert "sessionDir" in warnings[0] and "/data/pi-sessions" in warnings[0]


@pytest.mark.parametrize("content", ['{"theme":"dark"}', '{"sessionDir":""}', "{not json", "[]"])
def test_settings_without_a_session_dir_raise_no_warning(tmp_path, content):
    """Unset, empty or unreadable settings are not a custom session directory."""
    settings = tmp_path / ".pi" / "agent" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(content, encoding="utf-8")
    assert PiAdapter().store_warnings() == []


# --- a store Oh My Pi has written to --------------------------------------------------------


def _slot(title: str = "") -> bytes:
    """Build Oh My Pi's title slot: one line padded to 256 bytes, newline included."""
    fields = {"type": "title", "v": 1, "title": title, "updatedAt": TS}
    bare = json.dumps({**fields, "pad": ""}, separators=(",", ":")).encode("utf-8") + b"\n"
    padded = {**fields, "pad": " " * (256 - len(bare))}
    return json.dumps(padded, separators=(",", ":")).encode("utf-8") + b"\n"


def _leave_omp_mark(tmp_path: Path, marker: str) -> str:
    """Leave one thing only Oh My Pi writes in Pi's store.

    Args:
        tmp_path: The sandboxed home.
        marker: Which mark to leave.

    Returns:
        A piece of the evidence :func:`omp_markers` reports for it.
    """
    agent = tmp_path / ".pi" / "agent"
    folder = _sessions(tmp_path) / "-work-app"  # an Oh My Pi bucket name
    transcript = folder / "2026-05-01T10-00-00-000Z_O1.jsonl"
    if marker in ("agent.db", "history.db", "models.db", "omp-2026-05-01.log"):
        agent.mkdir(parents=True, exist_ok=True)
        (agent / marker).write_bytes(b"")
        return marker
    if marker in ("terminal-sessions", "blobs"):
        (agent / marker).mkdir(parents=True)
        return marker
    if marker == "title slot":
        folder.mkdir(parents=True, exist_ok=True)
        transcript.write_bytes(_slot("x") + json.dumps(_header("O1")).encode("utf-8") + b"\n")
        return "title slot"
    write_compact_jsonl(transcript, [_header("O1")])
    if marker == "backup":
        # Oh My Pi's backup is a copy of a transcript it wrote: slot first.
        backup = folder / f"{transcript.name}.0123abcd.bak"
        backup.write_bytes(_slot("x") + json.dumps(_header("O1")).encode("utf-8") + b"\n")
        return "0123abcd.bak (starts with Oh My Pi's title slot)"
    assert marker == "artifacts", marker
    (folder / transcript.stem).mkdir()
    return "artifacts directory"


_OMP_MARKS = [
    "agent.db",
    "history.db",
    "models.db",
    "terminal-sessions",
    "blobs",
    "omp-2026-05-01.log",
    "title slot",
    "backup",
    "artifacts",
]


@pytest.mark.parametrize("marker", _OMP_MARKS)
def test_pi_yields_a_store_carrying_oh_my_pis_marks(tmp_path, marker):
    """Pi lists, cleans, moves and deletes nothing there, and says why."""
    path = _session_file(tmp_path)
    write_compact_jsonl(_sessions(tmp_path) / "--gone--" / "f.jsonl", [_user("x", "y", "frag")])
    evidence = _leave_omp_mark(tmp_path, marker)
    adapter = PiAdapter()
    session = Session(harness="pi", session_id="S1", project_path=str(OLD), paths=[path])
    before = path.read_bytes()

    assert any(evidence in item for item in adapter.omp_evidence())
    assert adapter.available() is False
    assert list(adapter.discover()) == []
    assert adapter.find_orphans() == []
    assert adapter.project_leftovers(str(OLD)) is None
    deleted = adapter.delete(session)
    assert path.exists() and "Oh My Pi" in deleted.refused[path]
    plan = adapter.plan_move([session], OLD, NEW)
    assert not plan.rewrites and not plan.relocations and path in plan.blocked
    assert adapter.move(plan).failed and path.read_bytes() == before
    warning = next(w for w in adapter.store_warnings() if "Oh My Pi" in w)
    assert evidence in warning and "listed under Oh My Pi" in warning


@pytest.mark.parametrize(
    "name",
    ["S.jsonl.1730000000.bak", "S.jsonl.0123abcd.bak", "S.jsonl.lock", ".S.jsonl.lock",
     ".S.jsonl.lock.os"],
)
def test_a_name_alone_is_not_oh_my_pis_mark(tmp_path, name):
    """A user's ``cp S.jsonl S.jsonl.$(date +%s).bak`` must not hide a whole Pi store.

    Oh My Pi names its crash backups and lock files that way, but only what a
    file holds — a title slot first — says Oh My Pi wrote it.
    """
    path = _session_file(tmp_path)
    other = _session_file(tmp_path, cwd="/work/b", sid="S2")
    copy = path.parent / name.replace("S.jsonl", path.name)
    copy.write_bytes(path.read_bytes() if name.endswith(".bak") else b"")
    adapter = PiAdapter()

    assert adapter.omp_evidence() == []
    assert adapter.available() is True
    assert sorted(s.session_id for s in adapter.discover()) == ["S1", "S2"]
    assert not any("Oh My Pi" in warning for warning in adapter.store_warnings())
    assert other.exists()


def test_pi_store_files_are_not_oh_my_pi_marks(tmp_path):
    """Pi's own configuration, logs and folders say nothing about Oh My Pi."""
    agent = tmp_path / ".pi" / "agent"
    for name in ("auth.json", "models.json", "settings.json", "pi-debug.log"):
        (agent / name).parent.mkdir(parents=True, exist_ok=True)
        (agent / name).write_text("{}", encoding="utf-8")
    for name in ("extensions", "themes", "tools", "bin", "prompts"):
        (agent / name).mkdir()
    path = _session_file(tmp_path)
    (path.parent / "nested").mkdir()
    (path.parent / f"{path.name}.orig").write_bytes(path.read_bytes())
    (path.parent / "notes.txt").write_text("mine", encoding="utf-8")

    assert pi_module.omp_markers(agent, _sessions(tmp_path)) == []
    adapter = PiAdapter()
    assert adapter.available() is True
    assert [s.session_id for s in adapter.discover()] == ["S1"]
    assert adapter.store_warnings() == []


def test_agent_dir_marks_are_all_reported_and_spare_the_sample(tmp_path, monkeypatch):
    """Every agent-directory mark is listed; the sessions directory is then not read."""
    agent = tmp_path / ".pi" / "agent"
    agent.mkdir(parents=True)
    for name in ("agent.db", "history.db", "omp-a.log", "omp-b.log"):
        (agent / name).write_bytes(b"")
    monkeypatch.setattr(pi_module, "_omp_session_evidence", lambda path: pytest.fail("sampled"))

    found = pi_module.omp_markers(agent, _sessions(tmp_path))

    assert found == ["agent.db", "history.db", "omp-a.log and 1 more log(s)"]


def test_the_sessions_sample_is_bounded(tmp_path, monkeypatch):
    """However many folders there are, at most 32 first lines are read."""
    for number in range(60):
        write_compact_jsonl(
            _sessions(tmp_path) / f"--p{number}--" / f"2026-05-01T10-00-00-000Z_S{number}.jsonl",
            [_header(f"S{number}")],
        )
    probes: list[Path] = []
    real = pi_module._starts_with_title_slot
    monkeypatch.setattr(
        pi_module, "_starts_with_title_slot", lambda path: probes.append(path) or real(path)
    )

    assert PiAdapter().omp_evidence() == []
    assert 0 < len(probes) <= 32


def test_an_orphan_is_refused_once_the_store_carries_oh_my_pis_marks(tmp_path):
    """The scan found a dead folder; by the delete, Oh My Pi had written to the store."""
    folder = _sessions(tmp_path) / "--sx-gone--"
    write_compact_jsonl(folder / "d.jsonl", [_header("d", "/sx-gone")])
    _age_tree(folder)
    adapter = PiAdapter()
    (orphan,) = adapter.find_orphans()
    _leave_omp_mark(tmp_path, "agent.db")

    result = adapter.delete_orphan(orphan)

    assert folder.is_dir() and "Oh My Pi" in result.refused[folder]


def test_pi_never_asks_the_oh_my_pi_adapter(tmp_path):
    """The autouse fixture makes importing it fail; Pi works all the same."""
    _transcript(tmp_path, [_header()])
    adapter = PiAdapter()
    assert adapter.available() is True
    assert len(list(adapter.discover())) == 1
    assert adapter.store_warnings() == []


# --- delete -------------------------------------------------------------------------------


def test_delete_removes_only_the_transcript(tmp_path):
    """Pi keeps nothing beside the ``.jsonl``; nothing else in the folder goes."""
    keep = _transcript(tmp_path, [_header("other"), _user("u1", None, "keep me")], name="b.jsonl")
    path = _transcript(tmp_path, [_header("S1"), _user("u1", None, "bye")], name="a.jsonl")
    (path.parent / "notes.txt").write_text("unrelated", encoding="utf-8")
    adapter = PiAdapter()
    session = next(s for s in adapter.discover() if s.session_id == "S1")
    assert adapter.correlated_paths(session) == []

    preview = adapter.delete(session, dry_run=True)
    assert preview.removed == [path] and path.exists()

    result = adapter.delete(session)
    assert result.removed == [path] and not path.exists()
    assert keep.exists() and (path.parent / "notes.txt").exists()


def test_delete_never_widens_to_files_named_like_the_transcript(tmp_path):
    """``…_S1.jsonl`` goes; ``…_S10.jsonl`` and ``…_S1.jsonl.orig`` are someone else's."""
    path = _session_file(tmp_path, sid="S1")
    sibling = _session_file(tmp_path, sid="S10")
    copy = path.parent / f"{path.name}.orig"
    copy.write_bytes(path.read_bytes())
    adapter = PiAdapter()
    session = next(s for s in adapter.discover() if s.session_id == "S1")

    assert adapter.delete(session, dry_run=True).removed == [path]
    result = adapter.delete(session)

    assert result.removed == [path] and not path.exists()
    assert sibling.exists() and copy.exists()
    assert [s.session_id for s in adapter.discover()] == ["S10"]


def test_delete_refuses_when_the_header_is_no_longer_valid(tmp_path):
    """A format change under sx's feet is refused, not guessed at."""
    path = _transcript(tmp_path, [_header(), _user("u1", None, "hi")])
    adapter = PiAdapter()
    session = _only(adapter)
    path.write_bytes(b'{"type":"preamble","format":4}\n' + path.read_bytes())

    result = adapter.delete(session)

    assert path.exists()
    assert result.failed and "session header" in result.refused[path]


def test_delete_refuses_when_the_file_now_holds_another_session(tmp_path):
    """The file the user picked is not the file that is there now."""
    path = _transcript(tmp_path, [_header("S1")])
    adapter = PiAdapter()
    session = _only(adapter)
    write_compact_jsonl(path, [_header("S2")])

    result = adapter.delete(session)

    assert path.exists() and "different session" in result.refused[path]


def test_deleting_a_vanished_transcript_reports_it_missing(tmp_path):
    """Already gone is not a refusal."""
    path = _transcript(tmp_path, [_header()])
    adapter = PiAdapter()
    session = _only(adapter)
    path.unlink()

    result = adapter.delete(session)

    assert result.skipped == {path: "does not exist"} and not result.failed


# --- orphans --------------------------------------------------------------------------------


def _orphans(adapter: PiAdapter) -> dict[str, tuple[OrphanKind, list[str], str]]:
    """Index orphans by the name of their first path."""
    return {
        o.paths[0].name: (o.kind, [p.name for p in o.paths], o.reason)
        for o in adapter.find_orphans()
    }


def test_an_empty_folder_is_reported_with_what_it_holds(tmp_path):
    """A bare "no sessions" would understate a folder that still holds files."""
    sessions = _sessions(tmp_path)
    (sessions / "--empty--").mkdir(parents=True)
    (sessions / "--leftover--").mkdir()
    (sessions / "--leftover--" / "notes.txt").write_text("x", encoding="utf-8")
    write_compact_jsonl(sessions / "--hidden--" / ".tmp.jsonl", [_header()])
    for folder in sessions.iterdir():
        _age_tree(folder)

    found = _orphans(PiAdapter())

    assert found["--empty--"] == (
        OrphanKind.EMPTY,
        ["--empty--"],
        f"empty project folder; {PiAdapter.empty_folder_note}",
    )
    assert found["--leftover--"][0] is OrphanKind.EMPTY
    assert "contains 1 file(s)" in found["--leftover--"][2]
    assert "1 nested or hidden transcript(s)" in found["--hidden--"][2]


def test_a_header_less_file_is_a_fragment(tmp_path):
    """What Pi recreates when it appends after a delete: a stray, not a session."""
    _transcript(tmp_path, [_header()], name="ok.jsonl")
    _transcript(tmp_path, [_user("u9", "u8", "appended")], name="frag.jsonl")

    found = _orphans(PiAdapter())

    assert found == {
        "frag.jsonl": (
            OrphanKind.STRAY_TEMP,
            ["frag.jsonl"],
            "no session header — a fragment, not a session",
        )
    }


def test_a_folder_whose_projects_are_all_gone_is_dead(tmp_path):
    """Both colliding projects gone: the folder they share is dead."""
    folder = _sessions(tmp_path) / "--sx-gone-a-b--"
    write_compact_jsonl(folder / "one.jsonl", [_header("one", "/sx-gone/a-b")])
    write_compact_jsonl(folder / "two.jsonl", [_header("two", "/sx-gone/a/b")])

    orphans = PiAdapter().find_orphans()

    assert [(o.kind, o.paths) for o in orphans] == [(OrphanKind.DEAD_PROJECT, [folder])]
    assert orphans[0].reason == (
        "project directories no longer exist: /sx-gone/a-b, /sx-gone/a/b"
    )
    assert orphans[0].size_bytes > 0


@pytest.mark.parametrize("other", ["no cwd", "fragment", "live", "unmounted"])
def test_a_folder_is_never_called_dead_on_doubtful_evidence(tmp_path, other):
    """An unknown, live or offline project in the folder keeps it off the dead list."""
    folder = _sessions(tmp_path) / "--sx-gone-proj--"
    write_compact_jsonl(folder / "dead.jsonl", [_header("dead", "/sx-gone/proj")])
    if other == "no cwd":
        write_compact_jsonl(folder / "x.jsonl", [_header("x", None)])
    elif other == "fragment":
        write_compact_jsonl(folder / "x.jsonl", [_user("u", "p", "appended")])
    elif other == "live":
        write_compact_jsonl(folder / "x.jsonl", [_header("x", str(tmp_path))])
    else:
        write_compact_jsonl(folder / "x.jsonl", [_header("x", "/Volumes/sx-test-unplugged/p")])

    kinds = [o.kind for o in PiAdapter().find_orphans()]

    assert OrphanKind.DEAD_PROJECT not in kinds


def test_every_orphan_lies_inside_the_store(tmp_path):
    """The delete guard's allowlist holds every path the cleanup offers."""
    sessions = _sessions(tmp_path)
    (sessions / "--empty--").mkdir(parents=True)
    _age_tree(sessions / "--empty--")
    write_compact_jsonl(sessions / "--frag--" / "f.jsonl", [_user("u", "p", "appended")])
    write_compact_jsonl(sessions / "--sx-gone--" / "d.jsonl", [_header("d", "/sx-gone")])
    outside = tmp_path / "outside"
    outside.mkdir()
    (sessions / "--linked--").symlink_to(outside, target_is_directory=True)
    adapter = PiAdapter()

    orphans = adapter.find_orphans()

    assert {o.kind for o in orphans} == {
        OrphanKind.EMPTY, OrphanKind.STRAY_TEMP, OrphanKind.DEAD_PROJECT
    }
    for orphan in orphans:
        for path in orphan.paths:
            assert is_within(path, adapter.store_roots())


# --- move --------------------------------------------------------------------------------


def test_move_repoints_the_header_and_relocates_the_transcript(tmp_path):
    """Only ``cwd`` changes; every other line — torn last line included — is untouched."""
    path = _session_file(tmp_path)
    original = path.read_bytes().split(b"\n")
    adapter = PiAdapter()

    _, result = _move(adapter)

    moved = _sessions(tmp_path) / encode_session_folder(str(NEW)) / path.name
    assert not path.parent.exists()
    lines = moved.read_bytes().split(b"\n")
    assert lines[0] == original[0].replace(b'"cwd":"/old/proj"', b'"cwd":"/new/home/proj"')
    assert lines[1:] == original[1:]
    assert result.rewritten == [path] and result.fields_updated == 1
    assert result.moved == [(path.parent, moved.parent)]
    assert not result.refused
    session = _only(adapter)
    assert (session.project_path, session.paths) == (str(NEW), [moved])


def test_a_dry_run_changes_nothing(tmp_path):
    """The preview counts the work and leaves every byte where it was."""
    path = _session_file(tmp_path)
    before = path.read_bytes()

    _, result = _move(PiAdapter(), dry_run=True)

    assert path.read_bytes() == before
    assert result.dry_run and result.fields_updated == 1 and result.moved
    assert not (_sessions(tmp_path) / encode_session_folder(str(NEW))).exists()


def test_move_merges_into_an_existing_folder(tmp_path):
    """Pi already ran at the new path: the transcripts end up side by side."""
    path = _session_file(tmp_path)
    existing = _session_file(tmp_path, cwd=str(NEW), sid="S0")

    _, result = _move(PiAdapter())

    assert sorted(p.name for p in existing.parent.iterdir()) == sorted([existing.name, path.name])
    assert _first(existing.parent / path.name)["cwd"] == str(NEW)
    assert not path.parent.exists()
    assert not result.refused


def test_move_refuses_to_overwrite_a_transcript_at_the_destination(tmp_path):
    """A name already taken in the new folder is refused, never overwritten."""
    path = _session_file(tmp_path)
    taken = _sessions(tmp_path) / encode_session_folder(str(NEW)) / path.name
    write_compact_jsonl(taken, [_header("squatter", str(NEW))])
    before = taken.read_bytes()

    _, result = _move(PiAdapter())

    assert taken.read_bytes() == before
    assert path.exists()
    assert any("already exists" in reason for reason in result.refused.values())


def test_the_inverse_move_restores_the_original_bytes(tmp_path):
    """A move is reversible: there and back is byte-for-byte the same file."""
    path = _session_file(tmp_path)
    original = path.read_bytes()
    adapter = PiAdapter()

    _move(adapter)
    _move(adapter, old=NEW, new=OLD)

    assert path.read_bytes() == original
    assert not (_sessions(tmp_path) / encode_session_folder(str(NEW))).exists()


def test_a_colliding_projects_transcript_stays_put(tmp_path):
    """``/old/a/b`` moves out of the folder it shares with ``/old/a-b``, alone."""
    mine = _session_file(tmp_path, cwd="/old/a/b", sid="MINE")
    theirs = _session_file(tmp_path, cwd="/old/a-b", sid="THEIRS")
    assert mine.parent == theirs.parent
    before = theirs.read_bytes()

    _, result = _move(PiAdapter(), old=Path("/old/a/b"), new=Path("/new/c"))

    destination = _sessions(tmp_path) / encode_session_folder("/new/c")
    assert _first(destination / mine.name)["cwd"] == "/new/c"
    assert not mine.exists()
    assert theirs.read_bytes() == before
    assert "another project" in (result.note or "")


class _Recording(PiAdapter):
    """A Pi adapter that records what each header update is told."""

    def __init__(self) -> None:
        """Start with nothing recorded."""
        self.moved_from: list[Path | None] = []

    def header_update(self, header, old, new, *, moved_from):
        """Record ``moved_from``, then update the header as Pi does."""
        self.moved_from.append(moved_from)
        return super().header_update(header, old, new, moved_from=moved_from)


def test_header_update_learns_where_a_relocated_transcript_came_from(tmp_path):
    """Oh My Pi records the old path; the hook is given it."""
    path = _session_file(tmp_path)
    adapter = _Recording()
    _move(adapter)
    assert adapter.moved_from == [path]


def test_a_move_between_paths_sharing_a_folder_rewrites_in_place(tmp_path):
    """``/x/a/b`` to ``/x/a-b`` encodes to the same folder: nothing relocates."""
    path = _session_file(tmp_path, cwd="/x/a/b")
    adapter = _Recording()

    _, result = _move(adapter, old=Path("/x/a/b"), new=Path("/x/a-b"))

    assert _first(path)["cwd"] == "/x/a-b"
    assert result.moved == [] and adapter.moved_from == [None]


def test_fork_links_follow_a_relocated_parent(tmp_path):
    """A fork elsewhere names its parent by path; the path moves, so the link does."""
    parent = _session_file(tmp_path, sid="PARENT")
    child = _session_file(tmp_path, cwd="/elsewhere/proj", sid="CHILD", parentSession=str(parent))
    child_before = child.read_bytes().split(b"\n")
    adapter = PiAdapter()

    _, result = _move(adapter)

    new_parent = _sessions(tmp_path) / encode_session_folder(str(NEW)) / parent.name
    child_lines = child.read_bytes().split(b"\n")
    assert json.loads(child_lines[0])["parentSession"] == str(new_parent)
    assert child_lines[1:] == child_before[1:]
    assert child in result.rewritten and result.fields_updated == 2
    fork = next(s for s in adapter.discover() if s.session_id == "CHILD")
    assert fork.extra["parent_session"] == str(new_parent)


def test_a_fork_moving_with_its_parent_is_relinked_too(tmp_path):
    """Same project: the fork's ``cwd`` and its link both change, counted once per file."""
    parent = _session_file(tmp_path, sid="PARENT")
    child = _session_file(tmp_path, sid="CHILD", parentSession=str(parent))

    _, result = _move(PiAdapter())

    folder = _sessions(tmp_path) / encode_session_folder(str(NEW))
    header = _first(folder / child.name)
    assert header["cwd"] == str(NEW)
    assert header["parentSession"] == str(folder / parent.name)
    assert result.fields_updated == 3
    assert len(result.rewritten) == 2


def test_a_fork_written_in_the_last_90_seconds_keeps_its_link(tmp_path):
    """A transcript being written right now is not rewritten under the harness."""
    parent = _session_file(tmp_path, sid="PARENT")
    child = _session_file(tmp_path, cwd="/elsewhere/proj", sid="CHILD", parentSession=str(parent))
    os.utime(child, None)
    before = child.read_bytes()

    _, result = _move(PiAdapter())

    assert child.read_bytes() == before
    assert "90 seconds" in result.skipped[child]


def test_a_dry_run_counts_fork_links_without_writing(tmp_path):
    """The preview includes the links it would re-point."""
    parent = _session_file(tmp_path, sid="PARENT")
    child = _session_file(tmp_path, cwd="/elsewhere/proj", sid="CHILD", parentSession=str(parent))
    before = child.read_bytes()

    _, result = _move(PiAdapter(), dry_run=True)

    assert child.read_bytes() == before
    assert result.fields_updated == 2 and child in result.rewritten


def test_a_fork_sx_must_not_rewrite_keeps_its_link(tmp_path):
    """An old fork whose header records a format version sx does not know is left alone."""
    parent = _session_file(tmp_path, sid="PARENT")
    child = _session_file(
        tmp_path, cwd="/elsewhere/proj", sid="CHILD", parentSession=str(parent), version=4
    )
    before = child.read_bytes()

    _, result = _move(PiAdapter())

    assert (_sessions(tmp_path) / encode_session_folder(str(NEW)) / parent.name).is_file()
    assert child.read_bytes() == before
    assert "not safe to rewrite" in result.skipped[child] and child not in result.rewritten


def test_a_fork_follows_its_parent_into_a_folder_that_appeared_after_planning(tmp_path):
    """Pi started at the new path meanwhile: the folder is merged into, and the link follows."""
    parent = _session_file(tmp_path, sid="PARENT")
    child = _session_file(tmp_path, cwd="/elsewhere/proj", sid="CHILD", parentSession=str(parent))
    adapter = PiAdapter()
    plan = adapter.plan_move(adapter.sessions_for_project(OLD), OLD, NEW)
    newcomer = _session_file(tmp_path, cwd=str(NEW), sid="NEWCOMER")

    result = adapter.move(plan)

    moved = newcomer.parent / parent.name
    assert moved.is_file() and newcomer.is_file() and not parent.exists()
    assert result.relocated == {parent: moved}
    assert _first(child)["parentSession"] == str(moved)


def test_fork_links_follow_a_chained_move_to_where_each_parent_ends_up(tmp_path):
    """``/x/app`` -> ``/x/app-v1`` while ``/x/app/v1`` has sessions: one folder name, reused."""
    first = _session_file(tmp_path, cwd="/x/app", sid="S1")
    second = _session_file(tmp_path, cwd="/x/app/v1", sid="S2")
    fork_one = _session_file(tmp_path, cwd="/elsewhere", sid="F1", parentSession=str(first))
    fork_two = _session_file(tmp_path, cwd="/elsewhere", sid="F2", parentSession=str(second))

    _, result = _move(PiAdapter(), old=Path("/x/app"), new=Path("/x/app-v1"))

    one, two = Path(_first(fork_one)["parentSession"]), Path(_first(fork_two)["parentSession"])
    assert one.is_file() and two.is_file()
    assert (one.parent.name, two.parent.name) == ("--x-app-v1--", "--x-app-v1-v1--")
    assert _first(one)["id"] == "S1" and _first(two)["id"] == "S2"
    assert not result.refused


def test_final_locations_follow_each_transcript_through_the_units_that_moved_it():
    """Moved twice: its last place. A path vacated and reused: not the same transcript."""
    a, b, c = Path("/s/--a--/x.jsonl"), Path("/s/--b--/x.jsonl"), Path("/s/--c--/x.jsonl")
    assert pi_module._final_locations({a: b, b: c}) == {a: c}
    assert pi_module._final_locations({b: c, a: b}) == {b: c, a: b}
    assert pi_module._final_locations({}) == {}


def test_a_fork_link_is_never_pointed_at_nothing(tmp_path):
    """Should a move report a destination where nothing is, the link stays as it was."""
    parent = _session_file(tmp_path, sid="PARENT")
    child = _session_file(tmp_path, cwd="/elsewhere/proj", sid="CHILD", parentSession=str(parent))
    before = child.read_bytes()
    result = MoveResult()

    PiAdapter()._repoint_fork_links({parent: tmp_path / "nowhere" / parent.name}, {}, result,
                                    dry_run=False)

    assert child.read_bytes() == before and "nothing is at" in result.skipped[child]


def test_moves_rewrite_through_the_move_transform_hook():
    """Pi no longer overrides ``_rewrite_all``, so every unit can be undone exactly."""
    adapter = PiAdapter()
    assert type(adapter)._rewrite_all is JsonlFolderAdapter._rewrite_all
    assert adapter._legacy_rewrites() is False


def test_the_move_transform_rewrites_the_header_alone_into_a_new_mapping():
    """Transforms must never edit the record they are given: the undo relies on it."""
    plan = MovePlan(harness="pi", old=OLD, new=NEW)
    transform = PiAdapter().move_transform(Path("/s/--old-proj--/s.jsonl"), plan, moved_from=None)
    header = _header()
    snapshot = json.loads(json.dumps(header))

    updated = transform(header)

    assert updated["cwd"] == str(NEW) and updated is not header and header == snapshot
    assert transform(_user("u1", None, f"edit {OLD}/x.py")) is None
    assert transform(_header("S2")) is None  # only the first header is the session's


def test_a_session_whose_header_changed_is_refused_not_moved(tmp_path):
    """The tripwire runs at planning time; the refused session is still listed."""
    path = _session_file(tmp_path)
    adapter = PiAdapter()
    sessions = adapter.sessions_for_project(OLD)
    path.write_bytes(b'{"type":"preamble","format":4}\n' + path.read_bytes())
    before = path.read_bytes()

    plan = adapter.plan_move(sessions, OLD, NEW)
    result = adapter.move(plan)

    assert plan.sessions == sessions and path in plan.blocked
    assert not plan.rewrites and not plan.relocations
    assert path.read_bytes() == before and path in result.refused


class _Busy(PiAdapter):
    """A Pi adapter that reports one transcript as held open by the harness."""

    def busy_transcript(self, path: Path) -> bool:
        """Report the ``OPEN`` session's transcript as in use."""
        return path.name.endswith("_OPEN.jsonl")


def test_a_busy_transcript_never_travels_with_its_folder(tmp_path):
    """The folder stays; only the selected session moves out of it."""
    chosen = _session_file(tmp_path, sid="CHOSEN")
    busy = _session_file(tmp_path, sid="OPEN")
    before = busy.read_bytes()
    adapter = _Busy()
    selected = [s for s in adapter.sessions_for_project(OLD) if s.session_id == "CHOSEN"]

    result = adapter.move(adapter.plan_move(selected, OLD, NEW))

    assert busy.read_bytes() == before
    assert (_sessions(tmp_path) / encode_session_folder(str(NEW)) / chosen.name).is_file()
    assert not chosen.exists()
    assert "stay put" in (result.note or "")


@pytest.mark.parametrize("version", [4, "3", True])
def test_a_move_never_rewrites_a_format_version_sx_does_not_know(tmp_path, version):
    """A header from a newer Pi is left alone: neither moved nor deleted."""
    path = _session_file(tmp_path, version=version)
    before = path.read_bytes()
    adapter = PiAdapter()
    session = _only(adapter)

    plan = adapter.plan_move([session], OLD, NEW)
    result = adapter.move(plan)

    assert f"version {version!r}" in plan.blocked[path]
    assert not plan.rewrites and not plan.relocations
    assert path.read_bytes() == before and path in result.refused
    preview = adapter.delete(session, dry_run=True)
    deleted = adapter.delete(session)
    assert preview.removed == [] and f"version {version!r}" in preview.refused[path]
    assert deleted.failed and f"version {version!r}" in deleted.refused[path]
    assert path.read_bytes() == before


def test_an_unknown_version_never_travels_with_its_folder(tmp_path):
    """The known session moves out alone; the refused one stays put, unchanged."""
    chosen = _session_file(tmp_path, sid="CHOSEN")
    newer = _session_file(tmp_path, sid="NEWER", version=4)
    before = newer.read_bytes()
    adapter = PiAdapter()

    result = adapter.move(adapter.plan_move(adapter.sessions_for_project(OLD), OLD, NEW))

    assert newer.read_bytes() == before
    assert (_sessions(tmp_path) / encode_session_folder(str(NEW)) / chosen.name).is_file()
    assert not chosen.exists()
    assert "stay put" in (result.note or "")


def test_every_planned_path_stays_inside_the_store(tmp_path):
    """The delete guard's invariant, applied to rewrites and relocations."""
    _session_file(tmp_path)
    _session_file(tmp_path, cwd=str(OLD / "sub"), sid="SUB")
    adapter = PiAdapter()

    plan = adapter.plan_move(adapter.sessions_for_project(OLD), OLD, NEW)

    roots = adapter.store_roots()
    assert len(plan.relocations) == 2
    for source, destination in plan.relocations:
        assert is_within(source, roots) and is_within(destination, roots)
    for path in plan.rewrites:
        assert is_within(path, roots)


def test_the_move_service_moves_pi_sessions_end_to_end(tmp_path):
    """Through the service, as the TUI and CLI drive it."""
    path = _session_file(tmp_path)
    adapter = PiAdapter()
    service = MoveService({adapter.name: adapter}, log_path=tmp_path / "op.log")

    plans = service.plan(OLD, NEW)
    results = service.move(plans)

    assert list(plans) == ["pi"] and plans["pi"].live == []
    assert results["pi"].fields_updated == 1
    moved = _sessions(tmp_path) / encode_session_folder(str(NEW)) / path.name
    assert _first(moved)["cwd"] == str(NEW)


# --- project leftovers ---------------------------------------------------------------------


def test_leftovers_offer_the_folder_once_its_last_session_is_deleted(tmp_path):
    """Delete the last session, then the folder it lived in — and nothing more."""
    path = _session_file(tmp_path)
    _age_tree(path.parent)
    adapter = PiAdapter()
    session = _only(adapter)

    leftovers = adapter.project_leftovers(str(OLD))

    assert leftovers is not None
    assert leftovers.paths == [path.parent] and leftovers.size_bytes > 0
    assert (leftovers.memory_files, leftovers.config_notes) == ([], [])
    adapter.delete(session)
    result = adapter.delete_project_leftovers(leftovers)
    assert result.removed == [path.parent] and not path.parent.exists()
    assert _sessions(tmp_path).is_dir()


def test_a_folder_shared_with_another_project_is_not_offered(tmp_path):
    """``/old/a-b``'s transcript would go with ``/old/a/b``'s folder."""
    _session_file(tmp_path, cwd="/old/a/b", sid="MINE")
    _session_file(tmp_path, cwd="/old/a-b", sid="THEIRS")
    assert PiAdapter().project_leftovers("/old/a/b") is None


def test_a_folder_holding_a_fragment_is_not_offered(tmp_path):
    """A file whose project cannot be read cannot be shown to be this project's."""
    path = _session_file(tmp_path)
    write_compact_jsonl(path.parent / "frag.jsonl", [_user("x", "y", "orphaned turn")])
    assert PiAdapter().project_leftovers(str(OLD)) is None


def test_no_folder_means_no_leftovers(tmp_path):
    """Nothing on disk, nothing to offer."""
    assert PiAdapter().project_leftovers(str(OLD)) is None


def test_leftovers_stay_if_a_session_appeared_since_the_offer(tmp_path):
    """The offer is made before the delete; a session started meanwhile survives."""
    path = _session_file(tmp_path)
    _age_tree(path.parent)
    adapter = PiAdapter()
    session = _only(adapter)
    leftovers = adapter.project_leftovers(str(OLD))
    adapter.delete(session)
    newcomer = _session_file(tmp_path, sid="LATER")

    result = adapter.delete_project_leftovers(leftovers)

    assert newcomer.exists()
    assert result.failed and "still holds a session" in result.refused[path.parent]


def test_leftovers_stay_if_pi_just_appended_a_fragment(tmp_path):
    """Pi still running the deleted session recreates a fragment; leave it be."""
    path = _session_file(tmp_path)
    _age_tree(path.parent)
    adapter = PiAdapter()
    session = _only(adapter)
    leftovers = adapter.project_leftovers(str(OLD))
    adapter.delete(session)
    write_compact_jsonl(path, [_user("u2", "a1", "one more turn")])

    result = adapter.delete_project_leftovers(leftovers)

    assert path.exists() and "90 seconds" in result.refused[path.parent]


def test_leftovers_refuse_a_folder_that_is_not_the_projects(tmp_path):
    """A doctored plan cannot point the removal at another project's folder."""
    _session_file(tmp_path)
    other = _session_file(tmp_path, cwd="/unrelated", sid="X")
    leftovers = ProjectLeftovers(project_path=str(OLD), paths=[other.parent])

    result = PiAdapter().delete_project_leftovers(leftovers)

    assert other.exists() and "not this project's" in result.refused[other.parent]
