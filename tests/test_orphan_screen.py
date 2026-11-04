"""Regression tests for the orphan-cleanup screen's delete bindings.

These guard against the "action calls an async helper but never awaits it" bug,
where pressing ``d``/``D`` silently did nothing. The screen's delete helpers
must run as Textual workers so their confirmation/await chain executes. The
confirmation also carries each harness's open-session caveat, as a session
delete does (D1). One orphan that removes two paths or more is a bulk delete:
it needs a typed phrase counting the paths the delete will actually remove.
Every path the confirmation shows as refused stays on disk, and is still
reported as it would be had the delete refused it: in the toast, in the
op-log, and by the row, which stays while anything of its orphan does.

All file operations use synthetic orphans under ``tmp_path``; the real
``~/.claude`` / ``~/.codex`` / ``~/.gemini`` stores are never touched.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime

import pytest
from textual.widgets import Button, DataTable, Input, Label

from sx.adapters import base as base_mod
from sx.adapters.base import HarnessAdapter
from sx.model import Capability, Message, Orphan, OrphanKind, Role
from sx.service import DeleteService
from sx.tui.app import SxApp
from sx.tui.screens import ConfirmDeleteScreen, OrphanScreen


class _FakeAdapter(HarnessAdapter):
    """Synthetic adapter whose store root is a temp dir."""

    name = "fake"
    display = "Fake"
    capabilities = Capability.BROWSE | Capability.ORPHANS | Capability.DELETE

    def __init__(self, root, orphan_files):
        self._root = root
        self._orphan_files = orphan_files

    def store_roots(self):
        return [self._root]

    def discover(self):
        return iter(())

    def load(self, session):
        return [Message(Role.USER, text="hi", timestamp=datetime(2026, 5, 30, 12, 0))]

    def find_orphans(self):
        return [
            Orphan(
                harness=self.name,
                kind=OrphanKind.STRAY_TEMP,
                paths=[p],
                reason=f"stray {p.name}",
                size_bytes=p.stat().st_size,
            )
            for p in self._orphan_files
            if p.exists()
        ]


def _wire(app, store, orphan_files, log_path, adapter=None):
    """Inject a synthetic adapter into the app, replacing the real registry."""
    adapter = adapter or _FakeAdapter(store, orphan_files)
    app._adapters_by_name = {adapter.name: adapter}
    app._delete_service = DeleteService({adapter.name: adapter}, log_path=log_path)
    app._sessions_by_harness = {}
    app._load_errors = []
    app._live_cache = {}


@pytest.fixture(autouse=True)
def _no_real_registry(monkeypatch):
    """Keep ``on_mount`` away from the developer's real harness stores.

    ``SxApp()`` runs a full registry scan on mount, before a test can inject its
    fake adapter, which made these tests read the real ``~/.claude`` and friends.
    """
    monkeypatch.setattr("sx.tui.app.build_registry", lambda: ([], []))


@pytest.mark.asyncio
async def test_orphan_delete_selected_removes_one(tmp_path):
    """Pressing ``d`` confirms and deletes exactly the selected orphan."""
    store = tmp_path / "store"
    store.mkdir()
    a = store / "a.tmp"
    a.write_text("a")
    b = store / "b.tmp"
    b.write_text("b")

    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        _wire(app, store, [a, b], tmp_path / "ops.log")
        app.action_show_orphans()
        await pilot.pause()
        await pilot.pause()
        assert isinstance(app.screen, OrphanScreen)
        assert len(app.screen._orphans) == 2

        app.screen.query_one("#orphan-table", DataTable).focus()
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        # The bug: nothing happened. The fix: a confirm modal appears.
        assert isinstance(app.screen, ConfirmDeleteScreen)
        await pilot.click("#confirm")
        await pilot.pause()
        await pilot.pause()

        assert not a.exists()
        assert b.exists()
        assert isinstance(app.screen, OrphanScreen)
        assert len(app.screen._orphans) == 1


@pytest.mark.asyncio
async def test_orphan_delete_all_requires_phrase_then_removes_all(tmp_path):
    """Pressing ``D`` requires the typed phrase, then deletes every orphan."""
    store = tmp_path / "store"
    store.mkdir()
    files = []
    for name in ("a.tmp", "b.tmp", "c.tmp"):
        p = store / name
        p.write_text(name)
        files.append(p)

    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        _wire(app, store, files, tmp_path / "ops.log")
        app.action_show_orphans()
        await pilot.pause()
        await pilot.pause()
        app.screen.query_one("#orphan-table", DataTable).focus()
        await pilot.pause()

        await pilot.press("D")
        await pilot.pause()
        await pilot.pause()
        assert isinstance(app.screen, ConfirmDeleteScreen)
        confirm = app.screen.query_one("#confirm", Button)
        assert confirm.disabled is True  # gated on the typed phrase

        app.screen.query_one("#confirm-input", Input).focus()
        await pilot.pause()
        for ch in "DELETE 3":
            await pilot.press(ch if ch != " " else "space")
        await pilot.pause()
        assert confirm.disabled is False

        await pilot.click("#confirm")
        await pilot.pause()
        await pilot.pause()
        assert all(not p.exists() for p in files)


class _GroupingAdapter(_FakeAdapter):
    """Reports all its stray files as one orphan, as stale artifacts are grouped."""

    def find_orphans(self):
        """One orphan holding every file."""
        files = [p for p in self._orphan_files if p.exists()]
        return [
            Orphan(
                harness=self.name,
                kind=OrphanKind.STALE_SESSION,
                paths=files,
                reason=f"{len(files)} entry(s) belong to sessions that no longer exist",
                size_bytes=sum(p.stat().st_size for p in files),
            )
        ] if files else []


@pytest.mark.asyncio
async def test_one_orphan_that_removes_many_paths_needs_the_typed_phrase(tmp_path):
    """A grouped orphan is a bulk delete: a burst of Enters never confirms it.

    This is how 24 real scratchpads were once deleted: one row grouped them,
    and a plain Enter confirmed the row.
    """
    store = tmp_path / "store"
    store.mkdir()
    files = []
    for name in ("a.tmp", "b.tmp", "c.tmp"):
        p = store / name
        p.write_text(name)
        files.append(p)

    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        _wire(app, store, files, tmp_path / "ops.log", adapter=_GroupingAdapter(store, files))
        app.action_show_orphans()
        await pilot.pause()
        await pilot.pause()
        app.screen.query_one("#orphan-table", DataTable).focus()
        await pilot.pause()

        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        assert isinstance(app.screen, ConfirmDeleteScreen)
        confirm = app.screen.query_one("#confirm", Button)
        assert confirm.disabled is True, "three paths need the typed phrase"
        await pilot.press("enter", "enter", "enter")
        await pilot.pause()
        assert all(p.exists() for p in files), "Enter alone must never delete them"

        app.screen.query_one("#confirm-input", Input).focus()
        await pilot.pause()
        for ch in "DELETE 3":
            await pilot.press(ch if ch != " " else "space")
        await pilot.pause()
        assert confirm.disabled is False
        await pilot.click("#confirm")
        await pilot.pause()
        await pilot.pause()
        assert all(not p.exists() for p in files)


class _ShieldingAdapter(_GroupingAdapter):
    """Groups its files into one orphan, but its guard refuses the last of them."""

    def protected_paths(self):
        """Refuse the last file outright, as a store's own bulk is refused."""
        return self._orphan_files[-1:]


def _files(store, count: int) -> list:
    """Make ``count`` small stray files in ``store``."""
    store.mkdir(exist_ok=True)
    made = []
    for index in range(count):
        path = store / f"e{index}.tmp"
        path.write_text(path.name)
        made.append(path)
    return made


async def _confirmation(app, pilot, store, files, log_path, adapter) -> ConfirmDeleteScreen:
    """Open the orphan screen on ``adapter`` and press ``d`` on its first row."""
    _wire(app, store, files, log_path, adapter=adapter)
    await _open_orphans(app, pilot)
    await pilot.press("d")
    await pilot.pause()
    await pilot.pause()
    assert isinstance(app.screen, ConfirmDeleteScreen)
    return app.screen


async def _type(pilot, text: str) -> None:
    """Type ``text`` into the focused widget, a key at a time."""
    for ch in text:
        await pilot.press(ch if ch != " " else "space")
    await pilot.pause()


def _phrase_label(screen: ConfirmDeleteScreen) -> str | None:
    """Return the typed-phrase prompt the dialog shows, if it shows one."""
    found = [
        str(label.content)
        for label in screen.query(Label)
        if str(label.content).startswith("Type ")
    ]
    return found[0] if found else None


@pytest.mark.asyncio
async def test_an_orphan_removing_exactly_two_paths_needs_the_typed_phrase(tmp_path):
    """Two paths are already a bulk delete: the boundary is two, not three."""
    store = tmp_path / "store"
    files = _files(store, 2)

    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        screen = await _confirmation(
            app, pilot, store, files, tmp_path / "ops.log", _GroupingAdapter(store, files)
        )
        confirm = screen.query_one("#confirm", Button)
        assert _phrase_label(screen) == 'Type "DELETE 2" to confirm:'
        assert confirm.disabled is True
        await pilot.press("enter", "enter", "enter")
        await pilot.pause()
        assert all(p.exists() for p in files), "Enter alone must never delete them"

        await _type(pilot, "DELETE 2")
        assert confirm.disabled is False
        await pilot.click("#confirm")
        await pilot.pause()
        await pilot.pause()
        assert not any(p.exists() for p in files)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("adapter_class", "listed"),
    [(_GroupingAdapter, 1), (_ShieldingAdapter, 2)],
    ids=["one-path", "two-paths-one-refused"],
)
async def test_an_orphan_removing_one_path_needs_no_typed_phrase(tmp_path, adapter_class, listed):
    """One path to remove is confirmed as any single delete is, however many are listed."""
    store = tmp_path / "store"
    files = _files(store, listed)

    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        screen = await _confirmation(
            app, pilot, store, files, tmp_path / "ops.log", adapter_class(store, files)
        )
        assert _phrase_label(screen) is None and not screen.query("#confirm-input")
        assert screen.query_one("#confirm", Button).disabled is False

        await pilot.click("#confirm")
        await pilot.pause()
        await pilot.pause()
        assert not files[0].exists()
        assert all(p.exists() for p in files[1:]), "a refused path stays"


@pytest.mark.asyncio
async def test_the_typed_phrase_counts_the_paths_the_delete_will_remove(tmp_path):
    """Three paths listed, one refused by the guard: the phrase and the delete say two."""
    store = tmp_path / "store"
    files = _files(store, 3)
    log = tmp_path / "ops.log"

    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        screen = await _confirmation(app, pilot, store, files, log, _ShieldingAdapter(store, files))
        confirm = screen.query_one("#confirm", Button)
        assert _phrase_label(screen) == 'Type "DELETE 2" to confirm:'
        assert screen.query_one("#confirm-input", Input).placeholder == "DELETE 2"

        await _type(pilot, "DELETE 3")
        assert confirm.disabled is True, "the listed count is not the phrase"
        screen.query_one("#confirm-input", Input).value = ""
        await pilot.pause()
        await _type(pilot, "DELETE 2")
        assert confirm.disabled is False
        await pilot.click("#confirm")
        await pilot.pause()
        await pilot.pause()

    assert [p.exists() for p in files] == [False, False, True]
    entry = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert entry["removed"] == [str(files[0]), str(files[1])]


class _FingerprintingAdapter(_GroupingAdapter):
    """Groups its files into one orphan, fingerprinted as real store data is."""

    fingerprints_orphans = True


@pytest.mark.asyncio
async def test_a_path_the_confirmation_shows_refused_is_not_deleted_after_all(
    tmp_path, monkeypatch
):
    """The dialog said "DELETE 2" and showed the third path refused: it must stay.

    The third path was written moments ago, so the preview refuses it for the
    quiet window. Typing the phrase takes long enough for the window to pass;
    the delete must still remove only what the user was shown and confirmed.
    """
    store = tmp_path / "store"
    files = _files(store, 3)
    past = time.time() - 3600
    for path in files[:2]:
        os.utime(path, (past, past))

    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        adapter = _FingerprintingAdapter(store, files)
        screen = await _confirmation(app, pilot, store, files, tmp_path / "ops.log", adapter)
        assert _phrase_label(screen) == 'Type "DELETE 2" to confirm:'
        await _type(pilot, "DELETE 2")
        monkeypatch.setattr(base_mod, "ORPHAN_QUIET_SECONDS", 0)  # the window has passed
        await pilot.click("#confirm")
        await pilot.pause()
        await pilot.pause()

    assert [p.exists() for p in files] == [False, False, True]


class _RefusingAdapter(_FakeAdapter):
    """Reports an orphan that its own store roots do not permit deleting.

    Mirrors the shipped Gemini defect: ``find_orphans`` emitted paths above the
    declared roots, so the guard refused every one of them.
    """

    name = "refusing"

    def store_roots(self):
        """Return a root that does not contain the reported orphan."""
        return [self._root / "narrower"]


@pytest.mark.asyncio
async def test_refused_orphan_keeps_its_row_and_reports_failure(tmp_path):
    """A refused delete must not be reported as success.

    The shipped behavior dropped the row and toasted "Deleted orphan · freed
    0 B" while the file remained on disk — the failure only appeared in the
    op-log.
    """
    store = tmp_path / "store"
    store.mkdir()
    # The declared root exists (so the adapter is "available"), but the reported
    # orphan sits outside it — exactly the shipped Gemini geometry.
    (store / "narrower").mkdir()
    stray = store / "stray.tmp"
    stray.write_text("x")

    notices: list[tuple[str, str]] = []
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        _wire(
            app,
            store,
            [stray],
            tmp_path / "ops.log",
            adapter=_RefusingAdapter(store, [stray]),
        )
        original_notify = app.notify
        app.notify = lambda msg, **kw: (  # type: ignore[assignment]
            notices.append((str(msg), kw.get("severity", "information"))),
            original_notify(msg, **kw),
        )[1]

        app.action_show_orphans()
        await pilot.pause()
        await pilot.pause()
        assert isinstance(app.screen, OrphanScreen)
        assert len(app.screen._orphans) == 1

        app.screen.query_one("#orphan-table", DataTable).focus()
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        assert isinstance(app.screen, ConfirmDeleteScreen)
        await pilot.click("#confirm")
        await pilot.pause()
        await pilot.pause()

        assert stray.exists(), "refused file must remain on disk"
        assert len(app.screen._orphans) == 1, "the row must survive a refusal"
        assert any(sev == "error" for _, sev in notices), notices
        assert not any("Deleted orphan" in msg for msg, _ in notices), notices


@pytest.mark.asyncio
async def test_orphan_delete_empty_is_graceful(tmp_path):
    """``d``/``D`` with no orphans must not crash and stays on the screen."""
    store = tmp_path / "store"
    store.mkdir()
    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        _wire(app, store, [], tmp_path / "ops.log")
        app.action_show_orphans()
        await pilot.pause()
        await pilot.pause()
        assert isinstance(app.screen, OrphanScreen)
        await pilot.press("d")
        await pilot.pause()
        await pilot.press("D")
        await pilot.pause()
        assert isinstance(app.screen, OrphanScreen)


# --- the open-session caveat (D1) -------------------------------------------------


class _CarefulAdapter(_FakeAdapter):
    """Warns about deleting what its harness has open, as Pi does.

    An empty Pi folder can still be in use by a Pi that just started in that
    project, so the warning belongs in the orphan confirmation too.
    """

    name = "careful"
    display = "Careful"
    open_session_caveat = "a Careful that just started in a project may be using its folder"


class _WaryAdapter(_FakeAdapter):
    """A second harness with a warning of its own, as Oh My Pi has."""

    name = "wary"
    display = "Wary"
    open_session_caveat = "Wary may write to it again"


def _wire_all(app, adapters, log_path):
    """Inject several synthetic adapters into the app, replacing the real registry."""
    by_name = {adapter.name: adapter for adapter in adapters}
    app._adapters_by_name = by_name
    app._delete_service = DeleteService(by_name, log_path=log_path)
    app._sessions_by_harness = {}
    app._load_errors = []
    app._live_cache = {}


async def _open_orphans(app, pilot) -> OrphanScreen:
    """Open the orphan screen with its table focused."""
    app.action_show_orphans()
    await pilot.pause()
    await pilot.pause()
    assert isinstance(app.screen, OrphanScreen)
    app.screen.query_one("#orphan-table", DataTable).focus()
    await pilot.pause()
    return app.screen


@pytest.mark.asyncio
async def test_the_orphan_confirmation_shows_the_open_session_caveat(tmp_path):
    """Deleting one orphan of a harness that warns shows its warning (D1)."""
    store = tmp_path / "store"
    store.mkdir()
    stray = store / "a.tmp"
    stray.write_text("a")

    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        _wire(app, store, [stray], tmp_path / "ops.log", adapter=_CarefulAdapter(store, [stray]))
        await _open_orphans(app, pilot)

        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        assert isinstance(app.screen, ConfirmDeleteScreen)
        caveat = app.screen.query_one("#caveat", Label)
        assert str(caveat.content) == (
            "⚠ A Careful that just started in a project may be using its folder"
        )
        await pilot.press("escape")
        await pilot.pause()
        assert stray.exists()


@pytest.mark.asyncio
async def test_deleting_every_orphan_shows_each_harness_caveat_once(tmp_path):
    """Every harness involved that warns is heard, once each; one that does not adds nothing."""
    stores = {name: tmp_path / name for name in ("careful", "wary", "fake")}
    files = {}
    for name, store in stores.items():
        store.mkdir()
        files[name] = [store / f"{n}.tmp" for n in ("a", "b")]
        for path in files[name]:
            path.write_text(path.name)

    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        _wire_all(
            app,
            [
                _CarefulAdapter(stores["careful"], files["careful"]),
                _FakeAdapter(stores["fake"], files["fake"]),
                _WaryAdapter(stores["wary"], files["wary"]),
            ],
            tmp_path / "ops.log",
        )
        await _open_orphans(app, pilot)

        await pilot.press("D")
        await pilot.pause()
        await pilot.pause()
        assert isinstance(app.screen, ConfirmDeleteScreen)
        caveat = app.screen.query_one("#caveat", Label)
        assert str(caveat.content) == (
            "⚠ A Careful that just started in a project may be using its folder\n"
            "⚠ Wary may write to it again"
        )
        await pilot.press("escape")
        await pilot.pause()
        assert all(path.exists() for paths in files.values() for path in paths)


@pytest.mark.asyncio
async def test_no_orphan_caveat_without_one_from_the_adapter(tmp_path):
    """A harness that sets no warning adds none to the orphan confirmation."""
    store = tmp_path / "store"
    store.mkdir()
    stray = store / "a.tmp"
    stray.write_text("a")

    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        _wire(app, store, [stray], tmp_path / "ops.log")
        await _open_orphans(app, pilot)

        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        assert isinstance(app.screen, ConfirmDeleteScreen)
        assert not app.screen.query("#caveat")
        await pilot.press("escape")
        await pilot.pause()


# --- what stayed: the confirmation's refusals are reported --------------------------------

FRESH = "was modified in the last 90 seconds (refused)"


class _Stale(_FakeAdapter):
    """Lists each file but the first three on its own, then those three as one orphan.

    Stale artifacts are reported a class at a time, as the three are. Every
    orphan is fingerprinted, as real store data is.
    """

    fingerprints_orphans = True

    def find_orphans(self):
        """The lone files, then the group, of the files still there."""
        found = [
            Orphan(self.name, OrphanKind.STRAY_TEMP, [p], f"stray {p.name}", 1)
            for p in self._orphan_files[3:]
            if p.exists()
        ]
        group = [p for p in self._orphan_files[:3] if p.exists()]
        if group:
            reason = f"{len(group)} stale entries"
            found.append(Orphan(self.name, OrphanKind.STALE_SESSION, group, reason, 3))
        return found


def _quiet(store, count: int, *fresh: int) -> list:
    """Make ``count`` stray files, quiet for an hour but for those numbered in ``fresh``."""
    files = _files(store, count)
    past = time.time() - 3600
    for index, path in enumerate(files):
        if index not in fresh:
            os.utime(path, (past, past))
    return files


def _notices(app) -> list[tuple[str, str]]:
    """Record every toast the app shows from now on, with its severity."""
    shown: list[tuple[str, str]] = []
    notify = app.notify

    def record(message, **kwargs):
        """Note the toast, then show it."""
        shown.append((str(message), kwargs.get("severity", "information")))
        return notify(message, **kwargs)

    app.notify = record
    return shown


def _records(log) -> list[dict]:
    """Return every record in the op-log."""
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


async def _confirm_with(app, pilot, key: str, phrase: str | None = None) -> ConfirmDeleteScreen:
    """Press ``key`` on the orphan screen and return the confirmation it opens.

    With ``phrase``, it is typed and the deletion confirmed, and the delete
    awaited to its end.
    """
    await pilot.press(key)
    await pilot.pause()
    await pilot.pause()
    dialog = app.screen
    assert isinstance(dialog, ConfirmDeleteScreen)
    if phrase is not None:
        await _type(pilot, phrase)
        await pilot.click("#confirm")
        await app.workers.wait_for_complete()
        await pilot.pause()
    return dialog


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key", "done"),
    [("d", "Removed 2 path(s) · 12 B."), ("D", "Deleted 1 orphan(s) in part · freed 12 B.")],
    ids=["one", "all"],
)
async def test_a_path_the_confirmation_refused_is_reported_and_keeps_its_row(tmp_path, key, done):
    """Two of three paths go; the third, written moments ago, stays, and everything says so.

    Pressed on the only orphan, ``D`` asks as ``d`` does. The row stays,
    narrowed to what is left; the toast warns which path stayed and why; the
    op-log names it with its reason.
    """
    store = tmp_path / "store"
    files = _quiet(store, 3, 2)
    log = tmp_path / "ops.log"

    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        _wire(app, store, files, log, adapter=_Stale(store, files))
        screen = await _open_orphans(app, pilot)
        shown = _notices(app)
        await _confirm_with(app, pilot, key, "DELETE 2")

        assert app.screen is screen
        assert [orphan.paths for orphan in screen._orphans] == [[files[2]]]

    assert [p.exists() for p in files] == [False, False, True]
    message, severity = shown[-1]
    assert severity == "warning"
    assert message.startswith(f"Partly deleted — 1 path(s) stayed on disk: e2.tmp ({FRESH}).")
    assert message.endswith(done)
    (entry,) = _records(log)
    assert entry["removed"] == [str(files[0]), str(files[1])]
    assert entry["skipped"] == {str(files[2]): FRESH}


@pytest.mark.asyncio
async def test_an_orphan_the_confirmation_refused_whole_is_logged_and_keeps_its_row(tmp_path):
    """Nothing is left to delete, and the delete is made all the same: the op-log says why."""
    store = tmp_path / "store"
    files = _quiet(store, 1, 0)
    log = tmp_path / "ops.log"

    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        _wire(app, store, files, log, adapter=_FingerprintingAdapter(store, files))
        screen = await _open_orphans(app, pilot)
        shown = _notices(app)
        await _confirm_with(app, pilot, "d", "")

        assert [orphan.paths for orphan in screen._orphans] == [files]

    assert files[0].exists()
    assert shown[-1] == (f"Not deleted — {FRESH}", "error")
    (entry,) = _records(log)
    assert (entry["removed"], entry["skipped"]) == ([], {str(files[0]): FRESH})


@pytest.mark.asyncio
async def test_delete_all_keeps_and_reports_every_path_it_showed_refused(tmp_path, monkeypatch):
    """A fresh file alone and a fresh file in a group stay, even once their refusals lapse.

    The dialog lists both refusals: the group's path below its paths, and the
    lone file's orphan among those refused whole. The quiet window runs out
    while the phrase is typed; what was shown refused stays all the same. Both
    rows stay, the group narrowed to what is left; the toast counts both paths;
    and the op-log has a record for each orphan, naming its refusal.
    """
    store = tmp_path / "store"
    files = _quiet(store, 4, 2, 3)
    log = tmp_path / "ops.log"

    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        _wire(app, store, files, log, adapter=_Stale(store, files))
        screen = await _open_orphans(app, pilot)
        shown = _notices(app)
        dialog = await _confirm_with(app, pilot, "D")
        text = dialog._preview.plain
        assert (
            "1 path(s) of the orphans listed will be REFUSED and left on disk:\n"
            f"  ✗ {files[2]} — {FRESH}\n"
        ) in text
        assert (
            f"1 orphan(s) will be REFUSED and left on disk:\n  ✗ stray e3.tmp — {FRESH}\n"
        ) in text
        await _type(pilot, "DELETE 2")
        monkeypatch.setattr(base_mod, "ORPHAN_QUIET_SECONDS", 0)  # the refusals lapse
        await pilot.click("#confirm")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app.screen is screen
        assert [orphan.paths for orphan in screen._orphans] == [[files[3]], [files[2]]]

    assert [p.exists() for p in files] == [False, False, True, True]
    message, severity = shown[-1]
    assert severity == "warning"
    assert message.startswith(
        f"Partly deleted — 2 path(s) stayed on disk: e3.tmp, e2.tmp ({FRESH})."
    )
    assert message.endswith("Deleted 1 orphan(s) in part · 1 refused · freed 12 B.")
    lone, group = _records(log)
    assert (lone["removed"], lone["skipped"]) == ([], {str(files[3]): FRESH})
    assert group["removed"] == [str(files[0]), str(files[1])]
    assert group["skipped"] == {str(files[2]): FRESH}


@pytest.mark.asyncio
async def test_delete_all_counts_the_orphans_and_paths_it_does_not_list(tmp_path):
    """Past the first 20 deletable orphans, one line counts the rest: orphans, paths, refusals."""
    store = tmp_path / "store"
    files = _quiet(store, 25, 2)  # 22 lone files, then a group of three with one fresh

    app = SxApp()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        _wire(app, store, files, tmp_path / "ops.log", adapter=_Stale(store, files))
        await _open_orphans(app, pilot)
        text = (await _confirm_with(app, pilot, "D"))._preview.plain
        await pilot.press("escape")
        await pilot.pause()

    assert [p for p in files if f"  • {p}\n" in text] == files[3:23]
    assert not any(str(p) in text for p in (*files[:3], *files[23:]))
    assert "  … and 3 more deletable orphan(s) not shown: 4 path(s) to delete, 1 refused\n" in text
    assert all(p.exists() for p in files)


@pytest.mark.asyncio
async def test_an_orphan_delete_the_op_log_could_not_record_says_so(tmp_path):
    """A logging failure is surfaced for orphan deletes, as it is for sessions."""
    store = tmp_path / "store"
    store.mkdir()
    stray = store / "a.tmp"
    stray.write_text("a")
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x")

    app = SxApp()
    seen: list[str] = []
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        _wire(app, store, [stray], blocker / "ops.log")
        original = app.notify

        def spy(message, **kwargs):
            """Record every notification, then show it as usual."""
            seen.append(str(message))
            return original(message, **kwargs)

        app.notify = spy  # type: ignore[method-assign]
        app.action_show_orphans()
        await pilot.pause()
        await pilot.pause()
        app.screen.query_one("#orphan-table", DataTable).focus()
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        await pilot.click("#confirm")
        await pilot.pause()
        await pilot.pause()

    assert not stray.exists()
    assert any("NOT logged" in message for message in seen), seen
