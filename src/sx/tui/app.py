"""The ``sx`` Textual application: a two-pane session browser with cleanup.

Left pane: a tree of sessions, grouped by project / date / recency (cycle with
``b``) and filterable (``/``). Installed-but-unsupported harnesses appear grayed.
Right pane: the selected session's transcript, scrollable, parsed lazily on a
background thread. Some harnesses (Pi, Oh My Pi) keep a session's abandoned
attempts in the same file; ``t`` cycles how those branches are shown — the
current branch alone, followed by the other branches, or the whole tree in the
order it was written — and the choice carries over to the next session opened.

Actions that change something on disk are deliberate and gated:

* ``e`` exports the highlighted session to Markdown (asking which view first,
  when the session branched);
* ``d`` permanently deletes it, after a preview modal (and, for live sessions,
  a typed confirmation); deletion can export first, and archives a session
  that branched whole unless told otherwise;
* ``m`` re-points the highlighted session's project at a new directory — for
  when the project has already been moved by hand;
* ``M`` moves the project directory itself and then does the same;
* ``o`` opens the orphan-cleanup screen.

Deletion has no undo — the guard rails are the preview, the typed confirmation,
the store-root allowlist (in the adapter), and the append-only op-log. A move is
reversible: the same op-log records both endpoints, so the inverse move restores
the previous state.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, RichLog, Tree
from textual.widgets.tree import TreeNode
from textual.worker import Worker, get_current_worker

from sx.export import SessionGoneError, export_session, session_gone
from sx.grouping import GroupMode, filter_sessions, group_sessions
from sx.memory import discover_memories, export_memory
from sx.model import (
    BranchMode,
    Capability,
    DeleteResult,
    Orphan,
    ProjectLeftovers,
    Session,
    Transcript,
)
from sx.registry import build_registry
from sx.render import transcript_to_text
from sx.service import DeleteService, MoveService, move_summary
from sx.tui.screens import (
    BranchModeScreen,
    ConfirmDeleteScreen,
    MemoryScreen,
    MoveScreen,
    OrphanScreen,
    toast_safe,
)
from sx.util import claude_config_warning, human_size, is_under, sanitize_text

_READY_STYLE = "bold green"
_DORMANT_STYLE = "dim"
#: Appended to the transcript's branch header, naming the key that changes it.
_BRANCH_HINT = "t: change view"
#: The worker group transcript loads run in. Each load cancels the one before
#: it, and only it: in the default group it also canceled an export or a delete
#: still in flight.
_LOAD_GROUP = "load-transcript"
#: Follows the live-session count in a move plan; ``sx move`` says the same.
_LIVE_MOVE_WARNING = (
    "moving a session its harness still has open breaks it: the next write "
    "fails, or lands in a header-less fragment at the old path"
)
#: How many leftover paths a delete preview lists per harness.
_LEFTOVER_PATHS_SHOWN = 8
#: How many paths a toast names before it only counts the rest.
_PATHS_NAMED = 3
#: How many deletable orphans the delete-all confirmation lists the paths of.
_ORPHANS_LISTED = 20


def _safe(value: object) -> str:
    """Return ``value`` as text that cannot drive the terminal.

    For anything read off disk — titles, paths, notes and reasons that embed
    them — on its way into a Rich ``Text``. A ``Text`` is never parsed as
    markup, but its characters reach the terminal as they are.

    Args:
        value: The value, or anything ``str()`` turns into it.

    Returns:
        The text, control characters replaced by visible glyphs.
    """
    return sanitize_text(str(value))


def _open_session_caveat(adapter) -> str | None:
    """Return what an adapter says about deleting or moving a session it has open.

    ``open_session_caveat`` is optional on adapters, so it is read with
    ``getattr``.

    Args:
        adapter: A harness adapter, or ``None``.

    Returns:
        The sentence, sanitized and with an initial capital, or ``None``.
    """
    caveat = getattr(adapter, "open_session_caveat", None)
    if not isinstance(caveat, str) or not caveat.strip():
        return None
    caveat = _safe(caveat.strip())
    return caveat[:1].upper() + caveat[1:]


def _store_owner(adapter) -> str | None:
    """Return the harness an adapter has handed its store over to, if any.

    ``store_owner`` is optional on adapters, so it is read with ``getattr``;
    one that raises counts as no answer.

    Args:
        adapter: A harness adapter.

    Returns:
        The other harness's display name, sanitized, or ``None``.
    """
    method = getattr(adapter, "store_owner", None)
    if not callable(method):
        return None
    try:
        owner = method()
    except Exception:  # noqa: BLE001 - one adapter must not break the tree
        return None
    if not isinstance(owner, str) or not owner.strip():
        return None
    return _safe(owner.strip())


def _partial_delete_notice(session: Session, outcome: DeleteResult) -> str:
    """Say what a delete that refused something left on disk.

    The delete did something — removed a path, or did the work its ``note``
    describes — so it is not :attr:`~sx.model.DeleteResult.failed`, but it is
    no success either: Oh My Pi's delete, for one, stops at a backup it cannot
    remove and keeps the transcript. The first refusal is the one that
    stopped the rest.

    Args:
        session: The session deleted.
        outcome: What the delete did; ``refused`` is not empty.

    Returns:
        The notification, escaped for :meth:`~textual.app.App.notify`.
    """
    stayed = list(outcome.refused)
    names = ", ".join(toast_safe(path.name or path) for path in stayed[:_PATHS_NAMED])
    if len(stayed) > _PATHS_NAMED:
        names += f" and {len(stayed) - _PATHS_NAMED} more"
    reason = toast_safe(next(iter(outcome.refused.values())))
    done: list[str] = []
    if outcome.removed:
        done.append(f"{len(outcome.removed)} path(s) · {human_size(outcome.freed_bytes)}")
    if outcome.note:
        done.append(toast_safe(outcome.note))
    if session.primary_path in outcome.refused:
        return (
            f"Session NOT deleted — {len(stayed)} path(s) stayed on disk, its transcript "
            f"among them: {names} ({reason}). Removed only {' · '.join(done)}."
        )
    return (
        f"Partly deleted — {len(stayed)} path(s) stayed on disk: {names} ({reason}). "
        f"Removed {' · '.join(done)}."
    )


def _mode_descriptions(transcript: Transcript) -> dict[BranchMode, str]:
    """Describe what each :class:`~sx.model.BranchMode` shows of ``transcript``.

    The current branch can be empty: a session whose last step went back to
    the start shows nothing on it. The current-branch description says so,
    rather than offering an export that holds no messages.

    Args:
        transcript: A transcript that branched.

    Returns:
        A one-line description per mode, in :class:`~sx.model.BranchMode` order.
    """
    descriptions = {mode: transcript.describe(mode) or mode.label() for mode in BranchMode}
    if not any(not message.marker for message in transcript.current):
        descriptions[BranchMode.CURRENT] += (
            " · the current branch is EMPTY, so this exports no messages"
        )
    return descriptions


def _contents_summary(path: Path) -> str:
    """Describe what a directory target actually holds, for delete previews.

    Deleting an orphaned folder is recursive, so the confirmation must say what
    is inside it — transcripts and memory files in particular are easy to destroy
    unknowingly when a folder is labeled merely "orphan".

    Args:
        path: The target about to be deleted.

    Returns:
        An indented one-line description, or ``""`` for non-directories.
    """
    if not path.is_dir():
        return ""
    try:
        files = [p for p in path.rglob("*") if p.is_file()]
    except OSError:
        return ""
    if not files:
        return "      (empty directory)\n"
    transcripts = [p for p in files if p.suffix == ".jsonl"]
    memory = [p for p in files if p.suffix == ".md" and "memory" in p.parts]
    bits = [f"{len(files)} file(s)"]
    if transcripts:
        bits.append(f"{len(transcripts)} transcript(s)")
    if memory:
        bits.append(f"{len(memory)} MEMORY file(s)")
    return "      contains " + ", ".join(bits) + "\n"


def _leftover_summary(path: Path, plan: ProjectLeftovers, session: Session) -> str:
    """Describe what one leftover path holds, for the last-session preview.

    The memory count comes from the adapter's own list rather than from file
    names, and the session being deleted is called out when its files are
    inside: those go whether or not the box is ticked.

    Args:
        path: One of ``plan.paths``.
        plan: The harness's project-scoped state.
        session: The session being deleted.

    Returns:
        An indented line ending in a newline, or ``""`` for anything that is
        not a readable directory.
    """
    if not path.is_dir():
        return ""
    try:
        files = [p for p in path.rglob("*") if p.is_file()]
    except OSError:
        return ""
    if not files:
        return "        (empty directory)\n"
    bits = [f"{len(files)} file(s)"]
    transcripts = sum(1 for p in files if p.suffix == ".jsonl")
    if transcripts:
        bits.append(f"{transcripts} transcript(s)")
    memory = sum(1 for p in plan.memory_files if is_under(str(p), path))
    if memory:
        bits.append(f"{memory} memory file(s)")
    line = "        contains " + ", ".join(bits)
    if any(is_under(str(p), path) for p in session.paths):
        line += " — this session's files among them"
    return line + "\n"


def _leftovers_label(leftovers: dict[str, ProjectLeftovers], adapters: dict) -> str:
    """Label the opt-in box with what it would remove, as the preview lists it.

    Named after what is actually there: a Pi session folder is neither memory
    nor settings.

    Args:
        leftovers: Project-scoped state, per harness.
        adapters: Adapters by name, for display names.

    Returns:
        The checkbox label.
    """
    names = [adapters[name].display if name in adapters else name for name in leftovers]
    owners = names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"
    paths = sum(len(plan.paths) for plan in leftovers.values())
    memory = sum(len(plan.memory_files) for plan in leftovers.values())
    notes = sum(len(plan.config_notes) for plan in leftovers.values())
    bits: list[str] = []
    if paths:
        size = sum(plan.size_bytes for plan in leftovers.values())
        bits.append(f"{paths} path(s), {human_size(size)}")
    if memory:
        bits.append(f"{memory} memory file(s)")
    if notes:
        bits.append(f"{notes} settings or history entry(s)")
    return f"Also remove the {owners} project state listed above ({', '.join(bits)})"


class SxApp(App):
    """Two-pane terminal browser for AI coding-harness sessions."""

    CSS = """
    #left { width: 52; border-right: solid $panel-darken-2; }
    #tree { padding: 0 1; height: 1fr; }
    #filter { display: none; margin: 0 1; }
    #filter.visible { display: block; }
    #transcript { padding: 0 1; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh_tree", "Refresh"),
        Binding("b", "cycle_group", "Group"),
        Binding("t", "cycle_branches", "Branches"),
        Binding("slash", "start_filter", "Filter"),
        Binding("o", "show_orphans", "Orphans"),
        Binding("n", "show_memory", "Memory"),
        Binding("e", "export_session", "Export"),
        Binding("d", "delete_session", "Delete"),
        Binding("m", "move_sessions", "Move"),
        # Shown, unlike the vim-style aliases below: this is a primary action,
        # and the more consequential of the two — it moves a real directory on
        # disk, not just session records. Hiding it would leave the riskier key
        # as the undiscoverable one.
        Binding("M", "relocate_project", "Move dir"),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("g", "scroll_top", "Top", show=False),
        Binding("G", "scroll_bottom", "Bottom", show=False),
        Binding("tab", "focus_next", "Switch pane", show=False),
    ]

    def __init__(self, check_updates: bool = False) -> None:
        """Initialize empty state; data is loaded on mount.

        Args:
            check_updates: If True, check GitHub for a newer release in the
                background after mount and show a toast if one is available.
        """
        super().__init__()
        self._adapters_by_name: dict = {}
        self._sessions_by_harness: dict[str, list[Session]] = {}
        self._delete_service: DeleteService | None = None
        self._move_service: MoveService | None = None
        self._group_mode = GroupMode.PROJECT
        self._filter = ""
        #: Short-lived cache of liveness badges, keyed by (harness, session id).
        #: The tree is rebuilt on every filter keystroke and each label otherwise
        #: costs a fresh stat/DB query per visible session. The badge is
        #: advisory; the authoritative check runs again, uncached, at delete time.
        self._live_cache: dict[tuple[str, str], tuple[float, bool]] = {}
        self._check_updates = check_updates
        #: How a session that branched is shown. It carries over from session
        #: to session, and is the default offered when one is exported with
        #: ``e`` (an export on the way to a delete starts on the whole tree).
        self._branch_mode = BranchMode.CURRENT
        #: The session in the transcript pane and its parsed transcript, kept
        #: so ``t`` can re-render it without parsing the session again. Both
        #: are ``None`` while nothing is loaded. Exports never use them: the
        #: session may have been written to since it was opened.
        self._open_session: Session | None = None
        self._open_transcript: Transcript | None = None
        #: Counts the loads :meth:`open_session` has started. A load hands its
        #: result to the pane only if it is still the latest one.
        self._load_generation = 0

    def compose(self) -> ComposeResult:
        """Build the static widget layout."""
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="left"):
                yield Input(placeholder="filter title/project…", id="filter")
                yield Tree("Harnesses", id="tree")
            yield RichLog(id="transcript", wrap=True, highlight=False, markup=False)
        yield Footer()

    def on_mount(self) -> None:
        """Load adapters, build the tree, and show the welcome text."""
        self.title = "sx"
        self._reload_data()
        self._populate_tree()
        log = self.query_one("#transcript", RichLog)
        log.write(
            Text.from_markup(
                "[bold]sx[/bold] — select a session to view its transcript.\n\n"
                "[dim]↑/↓ move · enter open · t branches · b group · / filter · "
                "o orphans · n memory · e export · m move · M move dir · d delete · "
                "q quit[/dim]"
            )
        )
        # Focus the tree explicitly. Textual otherwise focuses the first
        # focusable widget, which is the filter Input — and it keeps receiving
        # keys even though its CSS hides it, so every keybinding silently typed
        # into an invisible box instead of firing.
        self.query_one("#tree", Tree).focus()
        if self._check_updates:
            self._run_update_check()

    @work(thread=True)
    def _run_update_check(self) -> None:
        """Check GitHub for a newer release off-thread; toast if one exists."""
        from sx.update import check_for_update

        try:
            info = check_for_update()
        except Exception:  # noqa: BLE001 - update check must never break the app
            return
        if info is not None:
            # `latest` is a remote-supplied tag name and notify() parses markup.
            latest = sanitize_text(str(info.latest)).replace("[", r"\[")
            self.call_from_thread(
                self.notify,
                f"sx {latest} is available (you have {info.current}). "
                f"Upgrade: {info.command}",
                title="Update available",
                severity="information",
                timeout=10,
            )

    # --- data ------------------------------------------------------------

    def _reload_data(self) -> None:
        """Re-scan every harness and rebuild the in-memory session lists."""
        adapters, errors = build_registry()
        self._adapters_by_name = {a.name: a for a in adapters}
        self._delete_service = DeleteService(self._adapters_by_name)
        self._move_service = MoveService(self._adapters_by_name)
        self._load_errors = errors
        # The CLI writes this to stderr, which the alt-screen swallows — so the
        # TUI reuses the load-error row instead of dropping the warning.
        warning = claude_config_warning()
        if warning:
            self._load_errors.append(("Claude Code", warning))
        # The same goes for what each adapter has to say about its store.
        for adapter in adapters:
            try:
                warnings = [str(text) for text in adapter.store_warnings()]
            except Exception as exc:  # noqa: BLE001 - one adapter must not break the tree
                warnings = [f"store check failed: {exc!r}"]
            for text in warnings:
                row = (adapter.display, sanitize_text(text))
                if row not in self._load_errors:
                    self._load_errors.append(row)
        self._sessions_by_harness = {}
        for adapter in adapters:
            if Capability.BROWSE not in adapter.capabilities or not adapter.available():
                continue
            try:
                self._sessions_by_harness[adapter.name] = list(adapter.discover())
            except Exception as exc:  # noqa: BLE001
                self._sessions_by_harness[adapter.name] = []
                self._load_errors.append((adapter.display, repr(exc)))

    # --- tree construction ----------------------------------------------

    def _subtitle(self) -> str:
        """Return the header subtitle reflecting group mode and filter."""
        bits = [f"group: {self._group_mode.label()}"]
        if self._filter:
            bits.append(f"filter: {self._filter!r}")
        return " · ".join(bits)

    def _populate_tree(self) -> None:
        """Rebuild the tree from current data, group mode, and filter."""
        tree = self.query_one("#tree", Tree)
        tree.clear()
        tree.root.expand()
        self.sub_title = self._subtitle()

        for adapter_name, err in getattr(self, "_load_errors", []):
            tree.root.add_leaf(Text(f"⚠ {adapter_name}: {err}", style="red"))

        ready_names = list(self._sessions_by_harness)
        for name in ready_names:
            adapter = self._adapters_by_name[name]
            sessions = filter_sessions(self._sessions_by_harness[name], self._filter)
            self._add_ready_harness(tree.root, adapter, sessions)

        # Dormant / unavailable harnesses, grayed. One that handed its store to
        # another harness says so: "not installed" would send the user looking
        # for an install that is there.
        for adapter in self._adapters_by_name.values():
            if adapter.name in self._sessions_by_harness:
                continue
            label = Text()
            label.append("· ", style=_DORMANT_STYLE)
            label.append(adapter.display, style=_DORMANT_STYLE)
            if adapter.available():
                state = "dormant"
            elif owner := _store_owner(adapter):
                state = f"store belongs to {owner}"
            else:
                state = "not installed"
            label.append(f"  ({state})", style="dim italic")
            tree.root.add(label, data=None, allow_expand=False)

    def _add_ready_harness(self, root: TreeNode, adapter, sessions: list[Session]) -> None:
        """Add one browsable harness and its grouped sessions to the tree."""
        header = Text()
        header.append("● ", style=_READY_STYLE)
        header.append(adapter.display, style=_READY_STYLE)
        header.append(f"  ({len(sessions)})", style="dim")
        harness_node = root.add(header, data=None, expand=True)

        if not sessions:
            harness_node.add_leaf(Text("(no matching sessions)", style="dim italic"))
            return

        for group_label, group_sessions_list in group_sessions(sessions, self._group_mode):
            orphan = any(s.is_orphan for s in group_sessions_list)
            glabel = Text()
            # Grouped by project, the label is a path read from a session file.
            glabel.append(_safe(group_label), style="yellow" if orphan else "white")
            glabel.append(f"  ({len(group_sessions_list)})", style="dim")
            if orphan:
                glabel.append("  ⚠ orphan", style="bold yellow")
            group_node = harness_node.add(glabel, data=None, expand=True)
            for session in group_sessions_list:
                group_node.add_leaf(self._session_label(session), data=session)

    def _is_live_cached(self, session: Session, ttl: float = 2.0) -> bool:
        """Return the liveness badge for a session, cached briefly.

        Args:
            session: The session to check.
            ttl: Seconds a cached answer stays valid.

        Returns:
            True if the session appears to be in active use.
        """
        if self._delete_service is None:
            return False
        key = (session.harness, session.session_id)
        now = time.monotonic()
        cached = self._live_cache.get(key)
        if cached is not None and (now - cached[0]) < ttl:
            return cached[1]
        live = self._delete_service.is_active(session)
        self._live_cache[key] = (now, live)
        return live

    def _session_label(self, session: Session) -> Text:
        """Render a session leaf label (live badge, date, size, title)."""
        live = self._is_live_cached(session)
        date = session.modified.strftime("%Y-%m-%d") if session.modified else "          "
        title = sanitize_text(
            (session.title or session.session_id).replace("\n", " ").strip()
        )
        label = Text()
        if live:
            label.append("● ", style="bold red")
        label.append(date + "  ", style="dim")
        label.append(f"{human_size(session.size_bytes):>9}  ", style="cyan")
        label.append(title)
        return label

    # --- selection + lazy load ------------------------------------------

    def _current_session(self) -> Session | None:
        """Return the session under the tree cursor, if any."""
        tree = self.query_one("#tree", Tree)
        node = tree.cursor_node
        if node is not None and isinstance(node.data, Session):
            return node.data
        return None

    @on(Tree.NodeSelected)
    def _on_node_selected(self, event: Tree.NodeSelected) -> None:
        """Load a session's transcript when its leaf is selected."""
        if isinstance(event.node.data, Session):
            # Nothing is open until the new transcript arrives, so ``t`` cannot
            # re-render the previous session over "Loading…".
            self._open_session = None
            self._open_transcript = None
            log = self.query_one("#transcript", RichLog)
            log.clear()
            log.write(Text("Loading…", style="dim italic"))
            self.open_session(event.node.data)

    def open_session(self, session: Session) -> Worker[None]:
        """Parse and display a session transcript off the UI thread.

        Each call supersedes the one before it. Loads run in their own worker
        group, so a new one cancels the previous load and nothing else — an
        export or a delete in flight keeps running. A load that finishes after
        it was superseded is dropped, never shown, so the pane and ``t`` always
        belong to the session opened last.

        Args:
            session: The session to show.

        Returns:
            The worker doing the load.
        """
        self._load_generation += 1
        return self._load_into_pane(session, self._load_generation)

    @work(exclusive=True, thread=True, group=_LOAD_GROUP)
    def _load_into_pane(self, session: Session, generation: int) -> None:
        """Parse ``session`` and hand the result to the UI thread.

        A session that branched is shown in the current :class:`BranchMode`,
        headed by a line saying what that mode leaves out.

        Args:
            session: The session to show.
            generation: Which load this is, as counted by :meth:`open_session`.
        """
        worker = get_current_worker()
        transcript: Transcript | None = None
        adapter = self._adapters_by_name.get(session.harness)
        if adapter is None:
            text = Text(f"No adapter for {session.harness!r}", style="red")
        else:
            try:
                transcript = adapter.load_transcript(session)
            except Exception as exc:  # noqa: BLE001
                text = Text(f"Failed to load: {exc!r}", style="red")
            else:
                if worker.is_cancelled:
                    return
                text = transcript_to_text(transcript, self._branch_mode, hint=_BRANCH_HINT)
        if worker.is_cancelled:
            return
        self.call_from_thread(
            self._set_transcript, session, text, transcript, generation=generation
        )

    def _set_transcript(
        self,
        session: Session,
        text: Text,
        transcript: Transcript | None = None,
        *,
        generation: int,
    ) -> None:
        """Replace the transcript pane contents (UI thread).

        Does nothing when a later load has started since this one: canceling
        a thread worker does not stop its thread, so a superseded load can
        still finish, and the session it parsed is no longer the one selected.

        Args:
            session: The session now in the pane.
            text: What to show.
            transcript: The parsed transcript behind ``text``, kept for ``t``;
                ``None`` when the session could not be loaded.
            generation: Which load produced ``text``.
        """
        if generation != self._load_generation:
            return
        self._open_session = session
        self._open_transcript = transcript
        self._show_text(text)

    def _show_text(self, text: Text) -> None:
        """Write ``text`` into the transcript pane, scrolled to the top."""
        log = self.query_one("#transcript", RichLog)
        log.clear()
        log.write(text)
        log.scroll_home(animate=False)

    def action_cycle_branches(self) -> None:
        """Cycle how the open session's branches are shown (``t``).

        Re-renders the transcript already loaded instead of parsing the session
        again. The new mode is kept for every session opened afterwards.
        """
        transcript = self._open_transcript
        if transcript is None:
            self.notify("No transcript is open.", severity="warning")
            return
        if not transcript.branched:
            self.notify("This session has no other branches.")
            return
        self._branch_mode = self._branch_mode.next()
        self._show_text(transcript_to_text(transcript, self._branch_mode, hint=_BRANCH_HINT))

    # --- filter ----------------------------------------------------------

    def action_start_filter(self) -> None:
        """Reveal and focus the filter input."""
        flt = self.query_one("#filter", Input)
        flt.add_class("visible")
        flt.value = self._filter
        flt.focus()

    @on(Input.Submitted, "#filter")
    def _on_filter_submitted(self, event: Input.Submitted) -> None:
        """Apply the filter and return focus to the tree."""
        self._filter = event.value.strip()
        self._populate_tree()
        self.query_one("#tree", Tree).focus()

    @on(Input.Changed, "#filter")
    def _on_filter_changed(self, event: Input.Changed) -> None:
        """Live-filter the tree as the query changes."""
        self._filter = event.value.strip()
        self._populate_tree()

    # --- actions ---------------------------------------------------------

    def action_refresh_tree(self) -> None:
        """Re-scan all harness stores and rebuild the tree."""
        self._reload_data()
        self._populate_tree()
        self.notify("Refreshed.")

    def action_cycle_group(self) -> None:
        """Cycle the grouping mode (project → date → recency)."""
        self._group_mode = self._group_mode.next()
        self._populate_tree()

    def action_cursor_down(self) -> None:
        """Move the tree cursor down (vim ``j``)."""
        self.query_one("#tree", Tree).action_cursor_down()

    def action_cursor_up(self) -> None:
        """Move the tree cursor up (vim ``k``)."""
        self.query_one("#tree", Tree).action_cursor_up()

    def action_scroll_top(self) -> None:
        """Scroll the transcript to the top (``g``)."""
        self.query_one("#transcript", RichLog).scroll_home(animate=False)

    def action_scroll_bottom(self) -> None:
        """Scroll the transcript to the bottom (``G``)."""
        self.query_one("#transcript", RichLog).scroll_end(animate=False)

    # --- export ----------------------------------------------------------

    @contextmanager
    def _toast_if_canceled(self, message: str) -> Iterator[None]:
        """Say so when the worker running the block is canceled.

        Textual cancels a worker by canceling its task, which raises
        :class:`asyncio.CancelledError` at whatever the worker is awaiting.
        Without this an export or a delete that was canceled simply never
        happens, and nothing tells the user.

        Args:
            message: The warning to show; it must hold true wherever in the
                block the cancellation lands.

        Yields:
            Nothing; the block runs as it is.
        """
        try:
            yield
        except asyncio.CancelledError:
            self.notify(message, severity="warning", timeout=10)
            raise

    @work
    async def action_export_session(self) -> None:
        """Export the highlighted session to Markdown.

        The session is read when the export runs, never taken from the pane:
        it may have been written to — or branched — since it was opened. For a
        session that branched, :class:`BranchModeScreen` first asks what the
        export should include, starting on the viewer's mode; canceling it
        cancels the export, and the session is read once more after the choice,
        so the file holds everything written until then. Any other session
        exports straight away; a harness without branches is not even read
        before the export. A session deleted since the tree was built is
        refused, and nothing is written.
        """
        session = self._current_session()
        if session is None:
            self.notify("No session selected.", severity="warning")
            return
        adapter = self._adapters_by_name.get(session.harness)
        if adapter is None:
            self.notify(f"No adapter for {session.harness!r}.", severity="error")
            return
        with self._toast_if_canceled("Export canceled before it finished."):
            mode = BranchMode.CURRENT
            transcript: Transcript | None = None
            if adapter.supports_branches:
                try:
                    transcript = await asyncio.to_thread(adapter.load_transcript, session)
                except Exception as exc:  # noqa: BLE001
                    self.notify(
                        f"Export failed: {self._toast_safe(repr(exc))}", severity="error"
                    )
                    return
                if transcript.branched:
                    choice = await self.push_screen_wait(
                        BranchModeScreen(
                            descriptions=_mode_descriptions(transcript),
                            default=self._branch_mode,
                        )
                    )
                    if choice is None:
                        return
                    mode = choice
                    # The dialog took time; read the session again as it exports.
                    transcript = None
            try:
                path = await asyncio.to_thread(
                    export_session, adapter, session, mode=mode, transcript=transcript
                )
            except SessionGoneError as exc:
                self.notify(
                    f"Nothing was exported — {self._toast_safe(exc)}. Press r to refresh "
                    "the list.",
                    severity="error",
                    timeout=10,
                )
                return
            except Exception as exc:  # noqa: BLE001
                self.notify(f"Export failed: {self._toast_safe(repr(exc))}", severity="error")
                return
            self.notify(f"Exported → {self._toast_safe(path)}")

    # --- delete ----------------------------------------------------------

    def _session_preview_text(
        self, session: Session, leftovers: dict, remaining: int
    ) -> Text:
        """Render the delete preview for a session.

        Shows the files that will go, any non-file work (database rows and
        prompt-history rows, via ``note``), and anything the guard will refuse —
        so the preview can never understate what a confirmation is about to
        destroy. It also states what is being **kept**: a project's memory
        outlives its conversations, and silently leaving it behind is how it ends
        up orphaned with nothing pointing at it.

        Everything shown that was read off disk — the title, paths, notes — is
        sanitized, so a session cannot drive the terminal through its preview.

        Args:
            session: The session about to be deleted.
            leftovers: Project-scoped state, per harness.
            remaining: How many sessions would still reference the project.
        """
        service = self._delete_service
        if service is None:
            return Text("Delete service unavailable.", style="red")
        result = service.preview(session)
        text = Text()
        title = (session.title or session.session_id).replace("\n", " ").strip()
        text.append(f"{_safe(title)}\n", style="bold")
        text.append(
            f"{_safe(session.harness)} · {_safe(session.project_path or '(unknown)')}\n\n",
            style="dim",
        )

        for path in result.removed:
            text.append(f"  • {_safe(path)}\n")
        if result.note:
            text.append(f"  • {_safe(result.note)}\n", style="bold yellow")
        if not result.removed and not result.note:
            text.append("Nothing to remove.\n", style="yellow")

        text.append(
            f"\nTotal: {len(result.removed)} path(s), {human_size(result.freed_bytes)}",
            style="bold",
        )
        cascade = len(result.removed) - 1
        if cascade > 0:
            text.append(f"  (includes {cascade} correlated file(s))", style="dim")

        if result.refused:
            text.append(f"\n\n{len(result.refused)} target(s) will be refused:\n", style="red")
            for path, reason in list(result.refused.items())[:5]:
                text.append(f"  ✗ {_safe(path)} — {_safe(reason)}\n", style="red")

        if leftovers:
            text.append("\n\nThis project also has:\n", style="bold")
            for name, plan in leftovers.items():
                self._append_leftovers(text, name, plan, session)
            if remaining > 0:
                text.append(
                    f"  kept — {remaining} other session(s) still use this project\n",
                    style="green",
                )
            else:
                text.append(
                    "  this is the LAST session for this project; tick the box "
                    "below to remove the above as well\n",
                    style="yellow",
                )
        return text

    def _append_leftovers(
        self, text: Text, name: str, plan: ProjectLeftovers, session: Session
    ) -> None:
        """List one harness's project-scoped state in a delete preview.

        Each path is named, with what it holds; then the memory documents and
        the settings the cleanup would edit.

        Args:
            text: The preview being built (appended to in place).
            name: The harness's name.
            plan: Its project-scoped state.
            session: The session being deleted.
        """
        adapter = self._adapters_by_name.get(name)
        text.append(f"  {adapter.display if adapter else _safe(name)}\n", style="bold")
        for path in plan.paths[:_LEFTOVER_PATHS_SHOWN]:
            kind = "folder" if path.is_dir() else "file"
            text.append(f"    • {kind} {_safe(path)}\n")
            text.append(_leftover_summary(path, plan, session), style="dim")
        hidden = len(plan.paths) - _LEFTOVER_PATHS_SHOWN
        if hidden > 0:
            text.append(f"    … and {hidden} more path(s)\n", style="dim")
        if plan.memory_files:
            names = ", ".join(_safe(path.name) for path in plan.memory_files[:5])
            more = " …" if len(plan.memory_files) > 5 else ""
            text.append(f"    • {len(plan.memory_files)} memory file(s): {names}{more}\n")
        for note in plan.config_notes:
            text.append(f"    • {_safe(note)}\n")

    @work
    async def action_delete_session(self) -> None:
        """Permanently delete the highlighted session, after confirmation.

        When the session branched, the dialog's optional export step also asks
        which :class:`BranchMode` to archive, starting on the whole tree
        whatever the viewer shows: that archive is the last copy of every
        branch. The export reads the session again when it runs; if no mode
        was chosen — the session had not branched when the dialog opened — it
        archives the whole tree, which keeps every branch it has by then.

        Liveness is checked again right before the delete. A session that
        became active while the dialog was open, and so was confirmed without
        the live-session phrase, is refused. One deleted elsewhere in the
        meantime is reported as gone, rather than taken for an active one
        because its file can no longer be read.
        """
        session = self._current_session()
        if session is None:
            self.notify("No session selected.", severity="warning")
            return
        service = self._delete_service
        if service is None:
            self.notify("Delete service unavailable.", severity="error")
            return
        # Every await in the flow comes before the delete itself, so wherever
        # a cancellation lands, nothing has been deleted yet.
        with self._toast_if_canceled("Delete canceled — nothing was deleted."):
            await self._delete_flow(service, session)

    async def _delete_flow(self, service: DeleteService, session: Session) -> None:
        """Run the delete dialog for ``session`` and carry out what it confirms.

        Args:
            service: The delete service.
            session: The session to delete.
        """
        adapter = self._adapters_by_name.get(session.harness)
        if self._gone(adapter, session):
            return
        is_live = service.is_active(session)
        leftovers = service.project_leftovers(session.project_path) if session.project_path else {}
        remaining = service.remaining_sessions(session, self._sessions_by_harness)
        preview = self._session_preview_text(session, leftovers, remaining)
        bundle_label = (
            _leftovers_label(leftovers, self._adapters_by_name)
            if remaining == 0 and leftovers
            else None
        )

        export_modes = None
        if adapter is not None and adapter.supports_branches:
            # Read fresh rather than reusing the pane's copy: the session may
            # have branched since it was opened, and this is the last chance to
            # archive what it holds. A transcript that will not load still
            # leaves the delete possible; the export step reports that failure.
            try:
                transcript = await asyncio.to_thread(adapter.load_transcript, session)
            except Exception:  # noqa: BLE001
                transcript = None
            if transcript is not None and transcript.branched:
                export_modes = _mode_descriptions(transcript)

        screen = ConfirmDeleteScreen(
            heading="Permanently delete this session?",
            preview=preview,
            can_export=True,
            is_live=is_live,
            typed_phrase="DELETE" if is_live else None,
            bundle_label=bundle_label,
            export_modes=export_modes,
            # Not the viewer's mode: what is not archived here is gone for good.
            export_mode=BranchMode.TREE,
            caveat=_open_session_caveat(adapter),
        )
        result = await self.push_screen_wait(screen)
        if result is None:
            return
        if adapter is None:
            self.notify(f"No adapter for {session.harness!r}.", severity="error")
            return
        if result.get("export"):
            # Read again at the moment of export, so the archive holds
            # everything written up to the delete. A mode was asked for only if
            # the session had branched when the dialog opened; if it branched
            # since, the whole tree is what keeps the branch it started on.
            # A session deleted elsewhere meanwhile has nothing left to archive.
            if self._gone(adapter, session):
                return
            mode = result.get("mode", BranchMode.TREE)
            try:
                path = await asyncio.to_thread(export_session, adapter, session, mode=mode)
            except SessionGoneError:
                self._report_gone(session)
                return
            except Exception as exc:  # noqa: BLE001
                self.notify(
                    f"Export failed, aborting delete: {self._toast_safe(repr(exc))}",
                    severity="error",
                )
                return
            self.notify(f"Exported → {self._toast_safe(path)}")
        # Deleted elsewhere while the dialog was open: the liveness check below
        # cannot read a file that is not there and takes that for activity, and
        # the delete would only report nothing to remove.
        if self._gone(adapter, session):
            return
        if not is_live and service.is_active(session):
            # Live sessions need the typed phrase, and this one was not live
            # when the dialog opened, so the phrase was never asked for.
            self.notify(
                "Nothing was deleted — the session became active while the dialog "
                "was open. Press d again to confirm it as a live session.",
                severity="error",
                timeout=10,
            )
            return
        outcome = service.delete(session)
        if outcome.failed:
            reason = next(iter(outcome.refused.values()), "unknown reason")
            self.notify(
                f"Nothing was deleted — {self._toast_safe(outcome.summary())} "
                f"({self._toast_safe(reason)})",
                severity="error",
                timeout=10,
            )
        elif outcome.refused:
            self.notify(
                _partial_delete_notice(session, outcome), severity="warning", timeout=10
            )
        else:
            self.notify(f"Deleted: {self._toast_safe(outcome.summary())}")
        if result.get("bundle") and leftovers:
            # A transcript that stayed is a session the project still has.
            kept = outcome.failed or session.primary_path in outcome.refused
            self._delete_leftovers(service, session, kept, leftovers)
        if service.last_log_error:
            self.notify(
                f"Deletion happened but was NOT logged — "
                f"{self._toast_safe(service.last_log_error)}",
                severity="warning",
                timeout=10,
            )
        self._reload_data()
        self._populate_tree()

    def _gone(self, adapter, session: Session) -> bool:
        """Tell the user if ``session`` no longer exists, and return whether it is gone.

        Args:
            adapter: The session's adapter, or ``None``.
            session: The session about to be exported or deleted.

        Returns:
            True if the session is gone (see :func:`~sx.export.session_gone`)
            and the user has been told so.
        """
        if not session_gone(adapter, session):
            return False
        self._report_gone(session)
        return True

    def _report_gone(self, session: Session) -> None:
        """Say that ``session`` no longer exists, and that nothing was deleted.

        Args:
            session: A session :func:`~sx.export.session_gone` found gone.
        """
        self.notify(
            f"Nothing was deleted — the session no longer exists: "
            f"{self._toast_safe(session.primary_path)} is gone. Press r to refresh the list.",
            severity="warning",
            timeout=10,
        )

    def _delete_leftovers(
        self,
        service: DeleteService,
        session: Session,
        session_failed: bool,
        leftovers: dict[str, ProjectLeftovers],
    ) -> None:
        """Remove the project state the user ticked, if it is still left over.

        The box was offered from the session list scanned when the tree was
        last built. A session started in the project since then would live in
        the same folders, so the count is taken again from the stores first;
        and if the session itself could not be deleted, the project still has
        it, and its state stays too.

        Args:
            service: The delete service.
            session: The session just deleted.
            session_failed: Whether the session is still there: its delete
                removed nothing, or its transcript was refused.
            leftovers: The project state offered, per harness.
        """
        if session_failed:
            self.notify(
                "Project state kept — the session itself was not deleted.",
                severity="warning",
                timeout=10,
            )
            return
        others = service.remaining_sessions(session)
        if others:
            self.notify(
                f"Project state kept — {others} other session(s) now use this project.",
                severity="warning",
                timeout=10,
            )
            return
        bundle = service.delete_project_leftovers(leftovers)
        freed = sum(r.freed_bytes for r in bundle.values())
        refused = sum(len(r.refused) for r in bundle.values())
        notes = [r.note for r in bundle.values() if r.note]
        detail = f"project state removed · {human_size(freed)}"
        if notes:
            detail += " · " + " · ".join(notes)
        if refused:
            detail += f" · {refused} refused"
        self.notify(
            self._toast_safe(detail),
            severity="warning" if refused else "information",
            timeout=10,
        )

    # --- move ------------------------------------------------------------

    @staticmethod
    def _toast_safe(text: object) -> str:
        """Make untrusted text safe to pass to :meth:`notify`.

        Notifications are rendered as markup and these strings embed paths read
        off disk, so a directory containing square brackets would otherwise be
        parsed as a style tag. See :func:`sx.tui.screens.toast_safe`.
        """
        return toast_safe(text)

    def _move_preview(
        self,
        old: Path,
        destination: str,
        *,
        relocate: bool,
        include_config: bool,
    ) -> tuple[Text, bool, bool]:
        """Render what moving ``old`` to ``destination`` would do.

        Every harness is dry-run, so the preview states the real extent of the
        work — including the parts that will be refused — rather than a guess.
        Each harness discovers its sessions afresh for the plan rather than
        reusing the tree's list, which can predate a transcript written since.

        Args:
            old: The project directory the sessions currently point at.
            destination: The candidate new directory, as typed.
            relocate: Whether the project directory itself is being moved.
            include_config: Whether to also re-point Claude's project settings.

        Returns:
            A ``(text, can_move, needs_phrase)`` triple for :class:`MoveScreen`.
        """
        service = self._move_service
        if service is None:
            return Text("Move service unavailable.", style="red"), False, False

        new = Path(destination).expanduser()
        if not new.is_absolute():
            return Text("Destination must be an absolute path.", style="red"), False, False
        if new == old:
            return Text("That is the current location.", style="yellow"), False, False

        text = Text()
        if relocate:
            reason = service.check_relocation(old, new)
            if reason is not None:
                text.append("The project directory cannot be moved:\n", style="bold red")
                text.append(f"  {_safe(reason)}\n", style="red")
                return text, False, False
            text.append("Move the project directory\n", style="bold")
            text.append(f"  {_safe(old)}\n   → {_safe(new)}\n")
            if service.crosses_devices(old, new):
                text.append(
                    "  crosses filesystems — copies, then deletes the original\n",
                    style="yellow",
                )
            text.append("\n")
        elif not new.exists():
            text.append(f"note: {_safe(new)} does not exist yet\n\n", style="yellow")

        plans = service.plan(old, new, include_config=include_config)
        if not plans:
            text.append("No harness has sessions at this project path.", style="yellow")
            return text, relocate, relocate

        results = service.move(plans, dry_run=True)
        live_total = 0
        for name, plan in plans.items():
            result = results[name]
            adapter = self._adapters_by_name.get(name)
            text.append(adapter.display if adapter else _safe(name), style="bold")
            text.append(f"  ({len(plan.sessions)} session(s))\n", style="dim")
            caveat = _open_session_caveat(adapter)
            if caveat and plan.sessions:
                text.append(f"  ⚠ {caveat}\n", style="bold yellow")
            for path in result.rewritten:
                text.append(f"  • re-point {_safe(path)}\n")
            for src, dst in result.moved:
                text.append(f"  • move  {_safe(src)}\n          → {_safe(dst)}\n")
            if result.note:
                text.append(f"  • {_safe(result.note)}\n", style="bold yellow")
            if plan.live:
                live_total += len(plan.live)
                text.append(
                    f"  ● {len(plan.live)} live session(s) — {_LIVE_MOVE_WARNING}\n",
                    style="bold red",
                )
            for path, reason in list(result.refused.items())[:5]:
                text.append(f"  ✗ {_safe(path)} — {_safe(reason)}\n", style="red")
            text.append("\n")

        files = sum(len(r.rewritten) for r in results.values())
        fields = sum(r.fields_updated for r in results.values())
        moves = sum(len(r.moved) for r in results.values())
        text.append(
            f"Total: {files} file(s) re-pointed, {fields} recorded path(s) updated, "
            f"{moves} relocation(s)",
            style="bold",
        )
        has_work = any(r.rewritten or r.moved or r.note for r in results.values())
        return text, (relocate or has_work), bool(live_total) or relocate

    @work
    async def action_move_sessions(self) -> None:
        """Re-point the highlighted session's project at a directory you moved."""
        await self._move_flow(relocate=False)

    @work
    async def action_relocate_project(self) -> None:
        """Move the project directory itself, then re-point its sessions."""
        await self._move_flow(relocate=True)

    async def _move_flow(self, *, relocate: bool) -> None:
        """Run the move dialog and carry out the confirmed operation.

        For a relocation the directory is moved first: if that fails nothing else
        is touched, whereas re-pointing first would leave every session aimed at
        a directory that was never created.

        The move is planned again right before it runs, from sessions each
        harness discovers then — as ``sx move`` does — so a transcript written
        since the preview is re-pointed too. A re-point confirmed without the
        typed phrase is refused if that plan finds a live session the preview
        did not.

        Args:
            relocate: True to move the project directory as well.
        """
        session = self._current_session()
        if session is None:
            self.notify("No session selected.", severity="warning")
            return
        service = self._move_service
        if service is None:
            self.notify("Move service unavailable.", severity="error")
            return
        if not session.project_path:
            self.notify(
                "This session has no known project directory.", severity="warning"
            )
            return

        old = Path(session.project_path)

        def preview(destination: str, include_config: bool):
            return self._move_preview(
                old, destination, relocate=relocate, include_config=include_config
            )

        screen = MoveScreen(
            heading=(
                "Move this project directory and re-point its sessions?"
                if relocate
                else "Re-point this project's sessions at a new directory?"
            ),
            old=str(old),
            preview=preview,
            can_config="claude" in self._adapters_by_name,
        )
        with self._toast_if_canceled("Move canceled — nothing was changed."):
            choice = await self.push_screen_wait(screen)
        if choice is None:
            return

        new = Path(str(choice["new"])).expanduser()
        include_config = bool(choice.get("include_config"))

        if relocate:
            with self._toast_if_canceled(
                "Move canceled while the project directory was being moved — its "
                "sessions were not re-pointed. Check both locations before retrying."
            ):
                reason = await asyncio.to_thread(service.relocate_project, old, new)
            if reason is not None:
                self.notify(
                    f"Nothing was moved — {self._toast_safe(reason)}",
                    severity="error",
                    timeout=10,
                )
                return
            self.notify(f"Moved {self._toast_safe(old)} → {self._toast_safe(new)}")

        plans = service.plan(old, new, include_config=include_config)
        live = sum(len(plan.live) for plan in plans.values())
        if live and not choice.get("typed_phrase"):
            # Only a re-point can get here: moving the directory always takes
            # the phrase, so nothing has been changed yet.
            self.notify(
                f"Nothing was moved — {live} session(s) became active after the "
                "preview. Preview again and confirm with the typed phrase.",
                severity="error",
                timeout=10,
            )
            return
        results = service.move(plans)
        self._report_move(results)
        if service.last_log_error:
            self.notify(
                f"The move happened but was NOT logged — "
                f"{self._toast_safe(service.last_log_error)}",
                severity="warning",
                timeout=10,
            )
        self._live_cache.clear()
        self._reload_data()
        self._populate_tree()

    def _report_move(self, results: dict) -> None:
        """Notify the outcome of a move, never overstating what happened."""
        if not results:
            self.notify("Nothing to re-point.", severity="warning")
            return
        if all(r.failed for r in results.values()):
            reason = next(
                (value for r in results.values() for value in r.refused.values()),
                "unknown reason",
            )
            self.notify(
                f"Nothing was re-pointed — {self._toast_safe(reason)}",
                severity="error",
                timeout=10,
            )
            return
        refused = sum(len(r.refused) for r in results.values())
        self.notify(
            f"Moved · {self._toast_safe(move_summary(results))}",
            severity="warning" if refused else "information",
            timeout=10 if refused else 6,
        )

    # --- memory ----------------------------------------------------------

    async def _confirm_memory(self, memories: list) -> bool:
        """Confirm removing one or more memory documents.

        Memory is the one thing ``sx`` deletes that was written to outlive the
        conversations that produced it, so the confirmation names each document
        and, for a whole project's worth, asks for a typed phrase.

        Args:
            memories: The memories about to be deleted.

        Returns:
            True if the user confirmed.
        """
        if self._delete_service is None or not memories:
            return False
        preview = Text()
        total = sum(m.size_bytes for m in memories)
        if len(memories) == 1:
            memory = memories[0]
            preview.append(f"{_safe(memory.name)}\n", style="bold")
            preview.append(
                f"{_safe(memory.project_path or '(unknown project)')}\n\n", style="dim"
            )
            if memory.description:
                preview.append(f"{_safe(memory.description)}\n\n", style="italic")
            preview.append(f"  • {_safe(memory.path)}\n")
        else:
            project = memories[0].project_path or "(unknown project)"
            preview.append(f"Every memory for {_safe(project)}\n", style="bold")
            preview.append(f"{len(memories)} document(s)\n\n", style="dim")
            for memory in memories[:20]:
                preview.append(f"  • {_safe(memory.name)}  ", style="")
                preview.append(f"{_safe(memory.path.name)}\n", style="dim")
            if len(memories) > 20:
                preview.append(f"  … and {len(memories) - 20} more\n", style="dim")
        preview.append(f"\nTotal: {human_size(total)}", style="bold")
        preview.append(
            "\n\nMemory is kept between sessions on purpose — deleting it is "
            "permanent and the sessions that wrote it will not restore it.",
            style="yellow",
        )
        screen = ConfirmDeleteScreen(
            heading=(
                "Permanently delete this memory?"
                if len(memories) == 1
                else "Permanently delete every memory for this project?"
            ),
            preview=preview,
            typed_phrase=f"DELETE {len(memories)}" if len(memories) > 1 else None,
        )
        return await self.push_screen_wait(screen) is not None

    def action_show_memory(self) -> None:
        """Open the project-memory browser."""
        if self._delete_service is None:
            self.notify("Delete service unavailable.", severity="error")
            return
        memories = discover_memories()
        if not memories:
            self.notify("No project memory found.", severity="warning")
            return
        screen = MemoryScreen(
            memories=memories,
            on_export=export_memory,
            on_delete=self._delete_service.delete_memory,
            confirm=self._confirm_memory,
        )
        self.push_screen(screen, lambda _result: self._after_orphans())

    # --- orphans ---------------------------------------------------------

    def _collect_orphans(self) -> list[Orphan]:
        """Gather orphans from every capable, available adapter."""
        orphans: list[Orphan] = []
        for adapter in self._adapters_by_name.values():
            if Capability.ORPHANS not in adapter.capabilities or not adapter.available():
                continue
            try:
                orphans.extend(adapter.find_orphans())
            except Exception:  # noqa: BLE001, S112
                continue
        return orphans

    def _orphan_caveat(self, orphans: list[Orphan]) -> str | None:
        """Return what the harnesses behind ``orphans`` say about what they have open.

        An orphan can still be in use: a Pi that just started in a project has
        made its folder and written nothing to it yet.

        Args:
            orphans: The orphans about to be deleted.

        Returns:
            Each distinct caveat on its own line, or ``None`` for none.
        """
        caveats: list[str] = []
        for orphan in orphans:
            caveat = _open_session_caveat(self._adapters_by_name.get(orphan.harness))
            if caveat and caveat not in caveats:
                caveats.append(caveat)
        return "\n".join(caveats) or None

    async def _confirm_orphan(
        self, orphans: list[Orphan]
    ) -> list[tuple[Orphan, dict[Path, str]]] | None:
        """Show a confirm modal for one or many orphans.

        The count shown and the typed-confirmation phrase are both derived from
        ``orphans`` (the exact list the caller will delete), so what the user
        sees always matches what they must type — no re-scan drift. Each
        harness involved that warns about deleting what it has open says so
        above the preview, as it does when a session is deleted. What the
        delete will refuse is shown with why, a path inside an orphan that is
        otherwise deletable included; past the first :data:`_ORPHANS_LISTED`
        deletable orphans, one line counts the orphans and paths not listed.

        Args:
            orphans: The orphan(s) about to be deleted.

        Returns:
            If the user confirmed, each orphan in order, narrowed to what the
            dialog showed as removable — its own paths, less those shown as
            refused or found gone, so it may be left with none — paired with
            the paths shown as refused and why. ``None`` if they canceled.
        """
        if self._delete_service is None or not orphans:
            return None

        if len(orphans) == 1:
            orphan = orphans[0]
            result = self._delete_service.preview_orphan(orphan)
            previews = [(orphan, result)]
            preview = Text()
            preview.append(f"{_safe(orphan.reason)}\n", style="bold")
            preview.append(f"{_safe(orphan.harness)} · {orphan.kind.value}\n\n", style="dim")
            for path in result.removed:
                preview.append(f"  • {_safe(path)}\n")
                preview.append(_contents_summary(path), style="dim")
            preview.append(f"\nTotal: {human_size(result.freed_bytes)}", style="bold")
            if result.refused:
                preview.append("\n\nThis will be REFUSED:\n", style="red")
                for path, reason in result.refused.items():
                    preview.append(f"  ✗ {_safe(path)} — {_safe(reason)}\n", style="red")
            # One orphan can group many paths (stale artifacts are reported a
            # class at a time), and removing many paths is a bulk delete: it
            # needs the typed phrase, never a plain Enter.
            removes = len(result.removed)
            screen = ConfirmDeleteScreen(
                heading="Permanently delete this orphan?",
                preview=preview,
                typed_phrase=f"DELETE {removes}" if removes > 1 else None,
                caveat=self._orphan_caveat(orphans),
            )
        else:
            # Dry-run every orphan so the totals reflect what the guard will
            # actually allow, rather than the orphans' own reported sizes.
            previews = [(o, self._delete_service.preview_orphan(o)) for o in orphans]
            deletable = [(o, r) for o, r in previews if not r.failed]
            refused = [(o, r) for o, r in previews if r.failed]
            total = sum(r.freed_bytes for _, r in deletable)
            listed, unlisted = deletable[:_ORPHANS_LISTED], deletable[_ORPHANS_LISTED:]

            preview = Text()
            preview.append(f"Delete ALL {len(orphans)} orphan(s)\n", style="bold")
            preview.append(
                f"{len(deletable)} deletable · {human_size(total)} reclaimable\n",
                style="dim",
            )
            for _, result in listed:
                for path in result.removed:
                    preview.append(f"  • {_safe(path)}\n")
                    preview.append(_contents_summary(path), style="dim")
            if unlisted:
                line = (
                    f"  … and {len(unlisted)} more deletable orphan(s) not shown: "
                    f"{sum(len(r.removed) for _, r in unlisted)} path(s) to delete"
                )
                kept = sum(len(r.refused) for _, r in unlisted)
                if kept:
                    line += f", {kept} refused"
                preview.append(f"{line}\n", style="dim")
            # A deletable orphan can still have paths the delete refuses.
            stays = [item for _, result in listed for item in result.refused.items()]
            if stays:
                preview.append(
                    f"\n{len(stays)} path(s) of the orphans listed will be REFUSED and "
                    "left on disk:\n",
                    style="red",
                )
                for path, reason in stays:
                    preview.append(f"  ✗ {_safe(path)} — {_safe(reason)}\n", style="red")
            if refused:
                preview.append(
                    f"\n{len(refused)} orphan(s) will be REFUSED and left on disk:\n",
                    style="red",
                )
                for orphan, result in refused[:5]:
                    reason = next(iter(result.refused.values()), "refused")
                    preview.append(
                        f"  ✗ {_safe(orphan.reason)} — {_safe(reason)}\n", style="red"
                    )
                if len(refused) > 5:
                    preview.append(
                        f"  … and {len(refused) - 5} more refused orphan(s) not shown\n",
                        style="red",
                    )
            screen = ConfirmDeleteScreen(
                heading="Permanently delete every orphan?",
                preview=preview,
                typed_phrase=f"DELETE {len(orphans)}",
                caveat=self._orphan_caveat(orphans),
            )
        if await self.push_screen_wait(screen) is None:
            return None
        # Delete only what the dialog showed as removable. A path it showed as
        # refused stays refused even if the reason has lapsed since — the quiet
        # window ran out while the phrase was being typed, say — and goes to
        # the delete with its reason, for the op-log and the outcome. An orphan
        # keeps its own paths rather than the dialog's list: a path the adapter
        # deletes around what it keeps (an Oh My Pi bucket, emptied around its
        # lock files) is listed as what is inside it, and must reach the
        # delete whole, for the checks made on it at the moment of deletion.
        return [
            (replace(o, paths=[p for p in o.paths if p not in r.skipped]), dict(r.refused))
            for o, r in previews
        ]

    def action_show_orphans(self) -> None:
        """Open the orphan-cleanup screen."""
        if self._delete_service is None:
            self.notify("Delete service unavailable.", severity="error")
            return
        orphans = self._collect_orphans()
        service = self._delete_service

        def delete_and_report(orphan: Orphan, **kwargs) -> DeleteResult:
            """Delete one orphan, and say so when the op-log could not record it.

            Args:
                orphan: The orphan to delete.
                **kwargs: Passed on to :meth:`DeleteService.delete_orphan`.

            Returns:
                The delete's result.
            """
            result = service.delete_orphan(orphan, **kwargs)
            if service.last_log_error:
                self.notify(
                    f"Deletion happened but was NOT logged — "
                    f"{self._toast_safe(service.last_log_error)}",
                    severity="warning",
                    timeout=10,
                )
            return result

        screen = OrphanScreen(
            orphans=orphans,
            on_delete=delete_and_report,
            on_delete_all=delete_and_report,
            confirm=self._confirm_orphan,
        )
        self.push_screen(screen, lambda _result: self._after_orphans())

    def _after_orphans(self) -> None:
        """Refresh data after returning from the orphan screen."""
        self._reload_data()
        self._populate_tree()


def run_app(check_updates: bool = False) -> int:
    """Launch the interactive TUI.

    Args:
        check_updates: If True, check GitHub for a newer release after launch.

    Returns:
        Process exit code (always ``0`` after a clean exit).
    """
    SxApp(check_updates=check_updates).run()
    return 0
